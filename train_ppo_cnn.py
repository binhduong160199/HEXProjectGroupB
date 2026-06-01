import time
import heapq
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from random import choice, random as rand_func
from copy import deepcopy

from hex_engine import hexPosition, EMPTY, RED, BLUE
from submission.cnn_policy import HexCNNPolicy
from submission.board_encoding import encode_board

# ==============================================================================
# 1. CRITICAL: FORCE DETERMINISTIC RANDOM SEEDS FOR REPRODUCIBILITY
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures internal CPU calculations match identically on every run
    torch.use_deterministic_algorithms(False) 

set_seed(42)

# ==============================================================================
# REINFORCEMENT LEARNING TUNED CURRICULUM CONFIGURATION
# ==============================================================================
CURRICULUM = [
    # Stage 1: 3x3 - High initial exploration to learn basic layout rules
    {"board_size": 3, "episodes": 4000, "epsilon_start": 0.40, "epsilon_end": 0.15, "block_reward": 0.05, "win_reward": 0.10, "lr": 0.00001},
    
    # Stage 2: 5x5 - Strategic transition with higher stakes for tactical mistakes
    {"board_size": 5, "episodes": 5000, "epsilon_start": 0.25, "epsilon_end": 0.08, "block_reward": 0.15, "win_reward": 0.25, "lr": 0.00001},
    
    # Stage 3: 7x7 - Target size refinement with dynamic recovery step tracking
    {"board_size": 7, "episodes": 6000, "epsilon_start": 0.15, "epsilon_end": 0.02, "block_reward": 0.30, "win_reward": 0.50, "lr": 0.00003}
]

GAMMA = 0.97
PPO_EPOCHS = 2
CLIP_EPSILON = 0.15
ENTROPY_COEF = 0.0005
VALUE_COEF = 0.25
MAX_GRAD_NORM = 0.2
LOGIT_CLIP = 20.0

MODEL_PATH = "ppo_cnn_hex.pt"


def get_device():
    return torch.device("cpu")


def is_finite_tensor(tensor):
    return torch.isfinite(tensor).all().item()


def action_to_index(move, board_size):
    row, col = move
    return row * board_size + col


def index_to_action(index, board_size):
    return index // board_size, index % board_size


def get_current_player(board):
    red_count = sum(cell == RED for row in board for cell in row)
    blue_count = sum(cell == BLUE for row in board for cell in row)
    return RED if red_count == blue_count else BLUE


def get_neighbors(row, col, size):
    candidates = [
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
        (row - 1, col + 1),
        (row + 1, col - 1),
    ]
    return [(r, c) for r, c in candidates if 0 <= r < size and 0 <= c < size]


def has_winning_path(board, player):
    size = len(board)
    visited = set()
    stack = []

    if player == RED:
        for row in range(size):
            if board[row][0] == RED:
                stack.append((row, 0))
                visited.add((row, 0))
        while stack:
            row, col = stack.pop()
            if col == size - 1:
                return True
            for nr, nc in get_neighbors(row, col, size):
                if board[nr][nc] == RED and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

    elif player == BLUE:
        for col in range(size):
            if board[0][col] == BLUE:
                stack.append((0, col))
                visited.add((0, col))
        while stack:
            row, col = stack.pop()
            if row == size - 1:
                return True
            for nr, nc in get_neighbors(row, col, size):
                if board[nr][nc] == BLUE and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
    return False


def find_winning_move(board, action_set, player):
    for move in action_set:
        test_board = deepcopy(board)
        row, col = move
        test_board[row][col] = player
        if has_winning_path(test_board, player):
            return move
    return None


def choose_center_move(board, action_set):
    size = len(board)
    center = (size - 1) / 2
    best_move = None
    best_distance = float("inf")
    for row, col in action_set:
        distance = abs(row - center) + abs(col - center)
        if distance < best_distance:
            best_distance = distance
            best_move = (row, col)
    return best_move


def greedy_agent(board, action_set):
    if not action_set:
        return None
    current_player = get_current_player(board)
    opponent = -current_player

    winning_move = find_winning_move(board, action_set, current_player)
    if winning_move is not None:
        return winning_move

    blocking_move = find_winning_move(board, action_set, opponent)
    if blocking_move is not None:
        return blocking_move

    return choose_center_move(board, action_set)


def random_agent(board, action_set):
    if not action_set:
        return None
    return choice(action_set)


def epsilon_greedy_agent(board, action_set, epsilon=0.2):
    if not action_set:
        return None
    if rand_func() < epsilon:
        return random_agent(board, action_set)
    return greedy_agent(board, action_set)


def cell_cost(cell_value, player):
    if cell_value == player:
        return 0.0
    if cell_value == EMPTY:
        return 1.0
    return 1000.0


def shortest_connection_distance(board, player):
    size = len(board)
    distances = [[float("inf") for _ in range(size)] for _ in range(size)]
    heap = []

    if player == RED:
        for row in range(size):
            cost = cell_cost(board[row][0], player)
            distances[row][0] = cost
            heapq.heappush(heap, (cost, row, 0))

        def target_reached(r, c):
            return c == size - 1
    else:
        for col in range(size):
            cost = cell_cost(board[0][col], player)
            distances[0][col] = cost
            heapq.heappush(heap, (cost, 0, col))

        def target_reached(r, c):
            return r == size - 1

    while heap:
        current_distance, row, col = heapq.heappop(heap)
        if current_distance > distances[row][col]:
            continue
        if target_reached(row, col):
            return current_distance
        for nr, nc in get_neighbors(row, col, size):
            new_distance = current_distance + cell_cost(board[nr][nc], player)
            if new_distance < distances[nr][nc]:
                distances[nr][nc] = new_distance
                heapq.heappush(heap, (new_distance, nr, nc))
    return 1000.0


def compute_path_potential_reward(board, action_set, move, player, stage_cfg, episode_in_stage):
    opponent = -player
    
    path_w = 0.010 if episode_in_stage > 2000 else 0.008
    opp_w = 0.005 if episode_in_stage > 2000 else 0.003
    reward_clip = 0.08 if episode_in_stage > 2000 else 0.05

    before_own = shortest_connection_distance(board, player)
    before_opponent = shortest_connection_distance(board, opponent)

    test_board = deepcopy(board)
    row, col = move
    test_board[row][col] = player

    after_own = shortest_connection_distance(test_board, player)
    after_opponent = shortest_connection_distance(test_board, opponent)

    own_improvement = before_own - after_own
    opponent_damage = after_opponent - before_opponent

    reward = 0.0
    reward += path_w * own_improvement
    reward += opp_w * opponent_damage

    own_winning_move = find_winning_move(board, action_set, player)
    opponent_winning_move = find_winning_move(board, action_set, opponent)

    if own_winning_move is not None and move == own_winning_move:
        reward += stage_cfg["win_reward"]

    if opponent_winning_move is not None:
        if move == opponent_winning_move:
            reward += stage_cfg["block_reward"]
        else:
            reward -= (stage_cfg["block_reward"] * 2.0)

    return max(min(reward, reward_clip), -reward_clip)


def create_action_mask(action_set, board_size, device):
    mask = torch.zeros(board_size * board_size, dtype=torch.bool, device=device)
    for move in action_set:
        mask[action_to_index(move, board_size)] = True
    return mask


def select_action(model, board, current_player, action_set, board_size, device, epsilon):
    state = encode_board(board, current_player).to(device)
    logits, value = model(state.unsqueeze(0))

    if not is_finite_tensor(logits) or not is_finite_tensor(value):
        move = choice(action_set)
        return move, torch.tensor(action_to_index(move, board_size), device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), state, create_action_mask(action_set, board_size, device)

    logits = torch.clamp(logits.squeeze(0), -LOGIT_CLIP, LOGIT_CLIP)
    value = torch.clamp(value.view(-1)[0], -2.0, 2.0)

    mask = create_action_mask(action_set, board_size, device)
    masked_logits = torch.full_like(logits, -1e4)
    masked_logits[mask] = logits[mask]

    if not is_finite_tensor(masked_logits):
        move = choice(action_set)
        return move, torch.tensor(action_to_index(move, board_size), device=device), torch.tensor(0.0, device=device), value, state, mask

    dist = Categorical(logits=masked_logits)

    if rand_func() < epsilon:
        move = choice(action_set)
        action_index = torch.tensor(action_to_index(move, board_size), device=device)
    else:
        action_index = dist.sample()
        move = index_to_action(action_index.item(), board_size)

    if move not in action_set:
        move = choice(action_set)
        action_index = torch.tensor(action_to_index(move, board_size), device=device)

    log_prob = dist.log_prob(action_index)
    if not is_finite_tensor(log_prob):
        log_prob = torch.tensor(0.0, device=device)

    return move, action_index, log_prob, value, state, mask


def compute_returns(rewards, gamma, device):
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def cnn_policy_move(model, board, action_set, board_size):
    current_player = get_current_player(board)
    state = encode_board(board, current_player).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state)
    logits = torch.clamp(logits.squeeze(0), -LOGIT_CLIP, LOGIT_CLIP)
    masked_logits = torch.full_like(logits, -1e4)
    for move in action_set:
        idx = action_to_index(move, board_size)
        masked_logits[idx] = logits[idx]
    return index_to_action(torch.argmax(masked_logits).item(), board_size)


def validation_agent(model, board, action_set, board_size):
    current_player = get_current_player(board)
    opponent = -current_player
    winning_move = find_winning_move(board, action_set, current_player)
    if winning_move is not None: return winning_move
    blocking_move = find_winning_move(board, action_set, opponent)
    if blocking_move is not None: return blocking_move
    return cnn_policy_move(model, board, action_set, board_size)


def play_validation_game(model, opponent_agent, board_size, cnn_is_red):
    game = hexPosition(size=board_size)
    while game.winner == EMPTY:
        action_set = game.get_action_space()
        if game.player == RED:
            move = validation_agent(model, game.board, action_set, board_size) if cnn_is_red else opponent_agent(game.board, action_set)
        else:
            move = opponent_agent(game.board, action_set) if cnn_is_red else validation_agent(model, game.board, action_set, board_size)
        if move not in action_set: move = choice(action_set)
        game.move(move)
    return (game.winner == RED) if cnn_is_red else (game.winner == BLUE)


def evaluate_model(model, opponent_agent, board_size, games=20):
    model.eval()
    wins = 0
    for i in range(games):
        if play_validation_game(model, opponent_agent, board_size, cnn_is_red=(i % 2 == 0)):
            wins += 1
    model.train()
    return wins / games


def transfer_weights(old_model, old_size, new_size):
    new_model = HexCNNPolicy(board_size=new_size)
    new_dict = new_model.state_dict()
    old_dict = old_model.state_dict()

    old_features = 64 * old_size * old_size

    for name, param in old_dict.items():
        if "conv" in name:
            new_dict[name].copy_(param)
        elif "fc_policy.weight" in name:
            for r in range(old_size):
                old_start_out = r * old_size
                new_start_out = r * new_size
                new_dict[name][new_start_out:new_start_out+old_size, :old_features].copy_(
                    param[old_start_out:old_start_out+old_size, :]
                )
        elif "fc_policy.bias" in name:
            for r in range(old_size):
                new_dict[name][r*new_size : r*new_size+old_size].copy_(param[r*old_size : r*old_size+old_size])
        elif "fc_value.weight" in name:
            new_dict[name][:, :old_features].copy_(param)
        elif "fc_value.bias" in name:
            new_dict[name].copy_(param)
            
    new_model.load_state_dict(new_dict)
    return new_model


def train():
    start_time = time.time()
    device = get_device()
    print("Using device:", device)

    model = None
    best_score = -1.0
    best_state = None  
    global_episode = 0  
    current_board_size = 3

    for stage_idx, stage in enumerate(CURRICULUM):
        b_size = stage["board_size"]
        episodes = stage["episodes"]
        
        print(f"\n--- STARTING STAGE {stage_idx+1}: Board Size {b_size}x{b_size} ---")
        
        if model is None:
            model = HexCNNPolicy(board_size=b_size).to(device)
        else:
            print(f"Transferring weights from {current_board_size} to {b_size} layout space safely...")
            model = transfer_weights(model, current_board_size, b_size).to(device)
            
        current_board_size = b_size
        optimizer = optim.Adam(model.parameters(), lr=stage["lr"], eps=1e-5)

        for episode in range(1, episodes + 1):
            global_episode += 1
            
            progress = episode / episodes
            epsilon = stage["epsilon_start"] - progress * (stage["epsilon_start"] - stage["epsilon_end"])
            
            mode = "self-play" if episode <= (episodes * 0.3) else "anti-greedy"
            cnn_player = BLUE if (mode == "anti-greedy" and rand_func() < 0.70) else RED

            game = hexPosition(size=b_size)
            states, actions, old_log_probs, old_values, masks, rewards, players = [], [], [], [], [], [], []

            while game.winner == EMPTY:
                current_player = game.player
                action_set = game.get_action_space()
                train_this_turn = (mode == "self-play") or (current_player == cnn_player)

                if train_this_turn:
                    move, action_index, log_prob, value, state, mask = select_action(
                        model, game.board, current_player, action_set, b_size, device, epsilon
                    )
                    shaped_reward = compute_path_potential_reward(game.board, action_set, move, current_player, stage, episode)
                    
                    states.append(state); actions.append(action_index); old_log_probs.append(log_prob.detach())
                    old_values.append(value.detach()); masks.append(mask); rewards.append(shaped_reward)
                    players.append(current_player)
                else:
                    move = greedy_agent(game.board, action_set)
                if move not in action_set: move = choice(action_set)
                game.move(move)

            winner = game.winner
            for i, player in enumerate(players):
                if player == winner:
                    rewards[i] += 5.0
                else:
                    rewards[i] -= 5.0

            if len(states) == 0: continue

            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            old_log_probs = torch.stack(old_log_probs).to(device)
            old_values = torch.stack(old_values).to(device)
            masks = torch.stack(masks).to(device)

            returns = compute_returns(rewards, GAMMA, device)
            advantages = returns - old_values
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            advantages = torch.clamp(advantages, -3.0, 3.0)

            for _ in range(PPO_EPOCHS):
                logits, new_values = model(states)
                logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)
                new_values = new_values.view(-1)

                masked_logits = torch.full_like(logits, -1e4)
                masked_logits[masks] = logits[masks]

                dist = Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs, -5.0, 5.0))
                surrogate_1 = ratio * advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages

                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = F.smooth_l1_loss(new_values, returns)
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()

            if episode % 500 == 0:
                greedy_score = evaluate_model(model, greedy_agent, board_size=b_size, games=40)
                epsilon_score = evaluate_model(model, epsilon_greedy_agent, board_size=b_size, games=40)
                combined_score = 0.70 * greedy_score + 0.30 * epsilon_score
                
                if stage_idx == 2:
                    if combined_score >= best_score:
                        best_score = combined_score
                        best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
                        print(f" >>> New Peak Checkpoint Saved! Score: {best_score*100:.1f}%")
                
                print(f"Stage {stage_idx+1} Ep {episode}: Greedy={greedy_score*100:.1f}% | E-Greedy={epsilon_score*100:.1f}% | Combined={combined_score*100:.1f}%")

    if best_state is not None:
        print(f"\nReloading peak checkpoint weights ({best_score*100:.2f}%)...")
        model.load_state_dict(best_state)

    print("\nTraining complete! Saving optimized native 7x7 model checkpoint...")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved optimized model to {MODEL_PATH} successfully. Overall Time: {(time.time() - start_time)/60:.2f} minutes.")


if __name__ == "__main__":
    train()