import time
import heapq
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from random import choice, random
from copy import deepcopy

from hex_engine import hexPosition, EMPTY, RED, BLUE
from submission.cnn_policy import HexCNNPolicy
from submission.board_encoding import encode_board


BOARD_SIZE = 7
EPISODES = 12000

GAMMA = 0.97
PPO_EPOCHS = 2
CLIP_EPSILON = 0.15
LEARNING_RATE = 0.00001
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

    return [
        (r, c)
        for r, c in candidates
        if 0 <= r < size and 0 <= c < size
    ]


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

    if random() < epsilon:
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


def reward_curriculum(episode):
    if episode <= 4000:
        return {
            "path_weight": 0.008,
            "opponent_weight": 0.003,
            "block_reward": 0.00,
            "miss_block_penalty": 0.00,
            "win_reward": 0.00,
            "reward_clip": 0.05,
        }

    if episode <= 8000:
        return {
            "path_weight": 0.010,
            "opponent_weight": 0.004,
            "block_reward": 0.04,
            "miss_block_penalty": -0.02,
            "win_reward": 0.05,
            "reward_clip": 0.07,
        }

    return {
        "path_weight": 0.010,
        "opponent_weight": 0.005,
        "block_reward": 0.06,
        "miss_block_penalty": -0.03,
        "win_reward": 0.08,
        "reward_clip": 0.08,
    }


def compute_path_potential_reward(board, action_set, move, player, episode):
    opponent = -player
    cfg = reward_curriculum(episode)

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
    reward += cfg["path_weight"] * own_improvement
    reward += cfg["opponent_weight"] * opponent_damage

    own_winning_move = find_winning_move(board, action_set, player)
    opponent_winning_move = find_winning_move(board, action_set, opponent)

    if own_winning_move is not None and move == own_winning_move:
        reward += cfg["win_reward"]

    if opponent_winning_move is not None:
        if move == opponent_winning_move:
            reward += cfg["block_reward"]
        else:
            reward += cfg["miss_block_penalty"]

    reward = max(min(reward, cfg["reward_clip"]), -cfg["reward_clip"])

    return reward


def create_action_mask(action_set, board_size, device):
    mask = torch.zeros(board_size * board_size, dtype=torch.bool, device=device)

    for move in action_set:
        index = action_to_index(move, board_size)
        mask[index] = True

    return mask


def exploration_epsilon(episode):
    if episode <= 4000:
        return 0.35

    if episode <= 8000:
        return 0.20

    return 0.08


def training_mode(episode):
    if episode <= 6000:
        return "self-play"

    return "anti-greedy"


def choose_cnn_player_against_greedy():
    if random() < 0.70:
        return BLUE

    return RED


def select_action(model, board, current_player, action_set, board_size, device, epsilon):
    state = encode_board(board, current_player).to(device)
    state_batch = state.unsqueeze(0)

    logits, value = model(state_batch)

    if not is_finite_tensor(logits) or not is_finite_tensor(value):
        move = choice(action_set)
        action_index = torch.tensor(action_to_index(move, board_size), device=device)
        log_prob = torch.tensor(0.0, device=device)
        mask = create_action_mask(action_set, board_size, device)
        return move, action_index, log_prob, torch.tensor(0.0, device=device), state, mask

    logits = logits.squeeze(0)
    logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)

    value = value.view(-1)[0]
    value = torch.clamp(value, -2.0, 2.0)

    mask = create_action_mask(action_set, board_size, device)

    masked_logits = torch.full_like(logits, -1e4)
    masked_logits[mask] = logits[mask]

    if not is_finite_tensor(masked_logits):
        move = choice(action_set)
        action_index = torch.tensor(action_to_index(move, board_size), device=device)
        log_prob = torch.tensor(0.0, device=device)
        return move, action_index, log_prob, value, state, mask

    dist = Categorical(logits=masked_logits)

    if random() < epsilon:
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


def cnn_policy_move(model, board, action_set):
    board_size = len(board)
    current_player = get_current_player(board)

    state = encode_board(board, current_player).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(state)

    logits = logits.squeeze(0)
    logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)

    masked_logits = torch.full_like(logits, -1e4)

    for move in action_set:
        index = action_to_index(move, board_size)
        masked_logits[index] = logits[index]

    best_index = torch.argmax(masked_logits).item()

    return index_to_action(best_index, board_size)


def validation_agent(model, board, action_set):
    current_player = get_current_player(board)
    opponent = -current_player

    winning_move = find_winning_move(board, action_set, current_player)
    if winning_move is not None:
        return winning_move

    blocking_move = find_winning_move(board, action_set, opponent)
    if blocking_move is not None:
        return blocking_move

    return cnn_policy_move(model, board, action_set)


def play_validation_game(model, opponent_agent, cnn_is_red):
    game = hexPosition(size=BOARD_SIZE)

    while game.winner == EMPTY:
        action_set = game.get_action_space()

        if game.player == RED:
            if cnn_is_red:
                move = validation_agent(model, game.board, action_set)
            else:
                move = opponent_agent(game.board, action_set)
        else:
            if cnn_is_red:
                move = opponent_agent(game.board, action_set)
            else:
                move = validation_agent(model, game.board, action_set)

        if move not in action_set:
            move = choice(action_set)

        game.move(move)

    if cnn_is_red:
        return game.winner == RED

    return game.winner == BLUE


def evaluate_model(model, opponent_agent, games=40):
    model.eval()
    wins = 0

    for i in range(games):
        cnn_is_red = i % 2 == 0

        if play_validation_game(model, opponent_agent, cnn_is_red):
            wins += 1

    model.train()

    return wins / games


def train():
    start_time = time.time()

    device = get_device()
    print("Using device:", device)

    game = hexPosition(size=BOARD_SIZE)
    model = HexCNNPolicy(board_size=BOARD_SIZE).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        eps=1e-5,
    )

    last_loss_value = 0.0
    best_score = -1.0
    best_state = None

    for episode in range(1, EPISODES + 1):
        game.reset()

        epsilon = exploration_epsilon(episode)
        mode = training_mode(episode)

        if mode == "anti-greedy":
            cnn_player = choose_cnn_player_against_greedy()
        else:
            cnn_player = None

        states = []
        actions = []
        old_log_probs = []
        old_values = []
        masks = []
        rewards = []
        players = []

        while game.winner == EMPTY:
            current_player = game.player
            action_set = game.get_action_space()

            if mode == "self-play":
                train_this_turn = True
            else:
                train_this_turn = current_player == cnn_player

            if train_this_turn:
                move, action_index, log_prob, value, state, mask = select_action(
                    model=model,
                    board=game.board,
                    current_player=current_player,
                    action_set=action_set,
                    board_size=BOARD_SIZE,
                    device=device,
                    epsilon=epsilon,
                )

                shaped_reward = compute_path_potential_reward(
                    board=game.board,
                    action_set=action_set,
                    move=move,
                    player=current_player,
                    episode=episode,
                )

                states.append(state)
                actions.append(action_index)
                old_log_probs.append(log_prob.detach())
                old_values.append(value.detach())
                masks.append(mask)
                rewards.append(shaped_reward)
                players.append(current_player)

            else:
                move = greedy_agent(game.board, action_set)

            if move not in action_set:
                move = choice(action_set)

            game.move(move)

        winner = game.winner

        for i, player in enumerate(players):
            if player == winner:
                rewards[i] += 1.0
            else:
                rewards[i] -= 1.0

        if len(states) == 0:
            continue

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        old_log_probs = torch.stack(old_log_probs).to(device)
        old_values = torch.stack(old_values).to(device)
        masks = torch.stack(masks).to(device)

        returns = compute_returns(rewards, GAMMA, device)
        returns = torch.clamp(returns, -2.0, 2.0)

        advantages = returns - old_values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        advantages = torch.clamp(advantages, -3.0, 3.0)

        if (
            not is_finite_tensor(states)
            or not is_finite_tensor(actions.float())
            or not is_finite_tensor(old_log_probs)
            or not is_finite_tensor(old_values)
            or not is_finite_tensor(returns)
            or not is_finite_tensor(advantages)
        ):
            print("Non-finite rollout data. Skipping episode update.")
            continue

        for _ in range(PPO_EPOCHS):
            logits, new_values = model(states)

            if not is_finite_tensor(logits) or not is_finite_tensor(new_values):
                print("Non-finite model output. Skipping this PPO update.")
                continue

            logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)
            new_values = new_values.view(-1)
            new_values = torch.clamp(new_values, -2.0, 2.0)

            masked_logits = torch.full_like(logits, -1e4)
            masked_logits[masks] = logits[masks]

            if not is_finite_tensor(masked_logits):
                print("Non-finite logits. Skipping this PPO update.")
                continue

            dist = Categorical(logits=masked_logits)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            if not is_finite_tensor(new_log_probs) or not is_finite_tensor(entropy):
                print("Non-finite distribution values. Skipping this PPO update.")
                continue

            log_ratio = new_log_probs - old_log_probs
            log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
            ratio = torch.exp(log_ratio)

            surrogate_1 = ratio * advantages
            surrogate_2 = torch.clamp(
                ratio,
                1.0 - CLIP_EPSILON,
                1.0 + CLIP_EPSILON,
            ) * advantages

            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

            value_loss = F.smooth_l1_loss(
                new_values.view(-1),
                returns.view(-1),
            )

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            if not is_finite_tensor(loss):
                print("Non-finite loss. Skipping this PPO update.")
                continue

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )

            if not torch.isfinite(grad_norm):
                print("Non-finite gradient detected. Skipping optimizer step.")
                optimizer.zero_grad()
                continue

            optimizer.step()

            last_loss_value = loss.item()

        if episode % 500 == 0:
            greedy_score = evaluate_model(model, greedy_agent, games=40)
            epsilon_score = evaluate_model(model, epsilon_greedy_agent, games=40)

            combined_score = 0.70 * greedy_score + 0.30 * epsilon_score

            if combined_score > best_score:
                best_score = combined_score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            print(
                f"Validation episode {episode}: "
                f"greedy = {greedy_score * 100:.2f}% | "
                f"epsilon-greedy = {epsilon_score * 100:.2f}% | "
                f"combined = {combined_score * 100:.2f}% | "
                f"best = {best_score * 100:.2f}%"
            )

        if episode % 100 == 0:
            winner_name = "RED" if winner == RED else "BLUE"
            cfg = reward_curriculum(episode)

            if mode == "self-play":
                cnn_info = "BOTH"
            else:
                cnn_info = "RED" if cnn_player == RED else "BLUE"

            print(
                f"Episode {episode}/{EPISODES} | "
                f"Mode: {mode} | "
                f"CNN: {cnn_info} | "
                f"Epsilon: {epsilon:.2f} | "
                f"Reward clip: {cfg['reward_clip']:.2f} | "
                f"Winner: {winner_name} | "
                f"Loss: {last_loss_value:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_PATH)

    total_time = time.time() - start_time

    print(f"Saved best model to {MODEL_PATH}")
    print(f"Best combined validation score: {best_score * 100:.2f}%")
    print(f"Training time: {total_time:.2f} seconds")
    print(f"Training time: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    train()