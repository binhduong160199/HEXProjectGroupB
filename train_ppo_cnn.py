import torch
import torch.optim as optim
from torch.distributions import Categorical
import heapq
import os
from random import choice, random

from hex_engine import hexPosition, EMPTY, RED, BLUE
from submission.cnn_policy import HexCNNPolicy
from submission.board_encoding import encode_board


EPISODES = 10000
OLD_SELF_DIR = "checkpoints/old_self"
OLD_SELF_INTERVAL = 500
OLD_SELF_START_EPISODE = 4000
MAX_OLD_SELF_MODELS = 6

CURRICULUM_PHASES = [
    (0.10, 5, "random", 0.0, 0.08),
    (0.25, 7, "epsilon_greedy", 0.3, 0.06),
    (0.40, 9, "greedy", 0.0, 0.04),
    (1.00, 11, "league", 0.0, 0.03),
]

GAMMA = 0.99
PPO_EPOCHS = 4
CLIP_EPSILON = 0.2
LEARNING_RATE = 0.0003
MAX_SHAPING_REWARD = 0.2

MODEL_PATH = "ppo_cnn_hex.pt"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_existing_model(model, device):
    if not os.path.exists(MODEL_PATH):
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded existing model from {MODEL_PATH}")
    except RuntimeError:
        print(
            f"Could not load {MODEL_PATH}. The checkpoint is not compatible "
            "with the current model, so training starts from scratch."
        )


def create_frozen_model_from_state(state_dict, device):
    model = HexCNNPolicy().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def save_old_self_snapshot(model, episode):
    os.makedirs(OLD_SELF_DIR, exist_ok=True)

    snapshot_path = os.path.join(OLD_SELF_DIR, f"ppo_cnn_episode_{episode}.pt")
    torch.save(model.state_dict(), snapshot_path)

    snapshots = sorted(
        os.path.join(OLD_SELF_DIR, file_name)
        for file_name in os.listdir(OLD_SELF_DIR)
        if file_name.endswith(".pt")
    )

    while len(snapshots) > MAX_OLD_SELF_MODELS:
        os.remove(snapshots.pop(0))

    print(f"Saved old-self snapshot to {snapshot_path}")


def load_old_self_pool(device):
    if not os.path.isdir(OLD_SELF_DIR):
        return []

    snapshot_paths = sorted(
        os.path.join(OLD_SELF_DIR, file_name)
        for file_name in os.listdir(OLD_SELF_DIR)
        if file_name.endswith(".pt")
    )[-MAX_OLD_SELF_MODELS:]

    old_self_pool = []

    for snapshot_path in snapshot_paths:
        try:
            state_dict = torch.load(snapshot_path, map_location=device)
            old_self_pool.append(create_frozen_model_from_state(state_dict, device))
        except RuntimeError:
            print(f"Skipped incompatible old-self snapshot: {snapshot_path}")

    if old_self_pool:
        print(f"Loaded {len(old_self_pool)} old-self opponents")

    return old_self_pool


def action_to_index(move, board_size):
    row, col = move
    return row * board_size + col


def index_to_action(index, board_size):
    row = index // board_size
    col = index % board_size
    return row, col


def create_action_mask(action_set, board_size, device):
    mask = torch.zeros(board_size * board_size, dtype=torch.bool, device=device)

    for move in action_set:
        index = action_to_index(move, board_size)
        mask[index] = True

    return mask


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
        test_board = [row[:] for row in board]
        row, col = move
        test_board[row][col] = player

        if has_winning_path(test_board, player):
            return move

    return None


def choose_center_move(board, action_set):
    size = len(board)
    center = (size - 1) / 2

    return min(
        action_set,
        key=lambda move: abs(move[0] - center) + abs(move[1] - center)
    )


def greedy_opponent_move(board, action_set, player):
    winning_move = find_winning_move(board, action_set, player)
    if winning_move is not None:
        return winning_move

    blocking_move = find_winning_move(board, action_set, -player)
    if blocking_move is not None:
        return blocking_move

    return choose_center_move(board, action_set)


def select_opponent_move(board, action_set, player, opponent_type, epsilon):
    if opponent_type == "random":
        return choice(action_set)

    if opponent_type == "epsilon_greedy" and random() < epsilon:
        return choice(action_set)

    return greedy_opponent_move(board, action_set, player)


def select_model_move(model, board, current_player, action_set, board_size, device):
    state = encode_board(board, current_player).to(device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(state)

    logits = logits.squeeze(0)
    masked_logits = torch.full_like(logits, -1e9)

    for move in action_set:
        masked_logits[action_to_index(move, board_size)] = logits[
            action_to_index(move, board_size)
        ]

    action_index = torch.argmax(masked_logits).item()

    return index_to_action(action_index, board_size)


def connection_cell_cost(cell, player):
    if cell == player:
        return 0.0

    if cell == EMPTY:
        return 1.0

    return float("inf")


def shortest_connection_distance(board, player):
    size = len(board)
    blocked_distance = float(size * size + 1)
    distances = {}
    heap = []

    if player == RED:
        start_cells = [(row, 0) for row in range(size)]
    else:
        start_cells = [(0, col) for col in range(size)]

    for row, col in start_cells:
        cost = connection_cell_cost(board[row][col], player)

        if cost < float("inf"):
            distances[(row, col)] = cost
            heapq.heappush(heap, (cost, row, col))

    while heap:
        distance, row, col = heapq.heappop(heap)

        if distance != distances[(row, col)]:
            continue

        if player == RED and col == size - 1:
            return distance

        if player == BLUE and row == size - 1:
            return distance

        for nr, nc in get_neighbors(row, col, size):
            cost = connection_cell_cost(board[nr][nc], player)

            if cost == float("inf"):
                continue

            next_distance = distance + cost

            if next_distance < distances.get((nr, nc), float("inf")):
                distances[(nr, nc)] = next_distance
                heapq.heappush(heap, (next_distance, nr, nc))

    return blocked_distance


def connection_advantage(board, player):
    own_distance = shortest_connection_distance(board, player)
    opponent_distance = shortest_connection_distance(board, -player)

    return opponent_distance - own_distance


def clip_reward(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def get_curriculum_settings(episode):
    progress = episode / EPISODES

    for phase_end, board_size, opponent_type, epsilon, shaping_scale in CURRICULUM_PHASES:
        if progress <= phase_end:
            if opponent_type == "league":
                return choose_league_curriculum_settings(board_size, shaping_scale)

            return board_size, opponent_type, epsilon, shaping_scale

    _, board_size, opponent_type, epsilon, shaping_scale = CURRICULUM_PHASES[-1]

    if opponent_type == "league":
        return choose_league_curriculum_settings(board_size, shaping_scale)

    return board_size, opponent_type, epsilon, shaping_scale


def choose_league_curriculum_settings(board_size, shaping_scale):
    roll = random()

    if roll < 0.25:
        return board_size, "old_self", 0.0, shaping_scale

    if roll < 0.40:
        return board_size, "self_play", 0.0, shaping_scale

    if roll < 0.55:
        return board_size, "greedy", 0.0, shaping_scale

    return board_size, "epsilon_greedy", 0.2, shaping_scale


def select_action(model, board, current_player, action_set, board_size, device):
    state = encode_board(board, current_player).to(device)
    state_batch = state.unsqueeze(0)

    logits, value = model(state_batch)
    logits = logits.squeeze(0)

    mask = create_action_mask(action_set, board_size, device)
    masked_logits = torch.full_like(logits, -1e9)
    masked_logits[mask] = logits[mask]

    dist = Categorical(logits=masked_logits)

    action_index = dist.sample()
    log_prob = dist.log_prob(action_index)

    move = index_to_action(action_index.item(), board_size)

    return move, action_index, log_prob, value.squeeze(0), state, mask


def compute_returns(rewards, gamma, device):
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    return torch.tensor(returns, dtype=torch.float32, device=device)


def train():
    device = get_device()
    print("Using device:", device)

    model = HexCNNPolicy().to(device)
    load_existing_model(model, device)
    old_self_pool = load_old_self_pool(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for episode in range(1, EPISODES + 1):
        board_size, opponent_type, opponent_epsilon, shaping_scale = (
            get_curriculum_settings(episode)
        )

        if opponent_type == "old_self" and not old_self_pool:
            opponent_type = "self_play"

        game = hexPosition(size=board_size)
        model_player = RED if episode % 2 == 1 else BLUE
        old_self_opponent = choice(old_self_pool) if opponent_type == "old_self" else None

        states = []
        actions = []
        old_log_probs = []
        old_values = []
        masks = []
        players = []
        rewards = []

        while game.winner == EMPTY:
            current_player = game.player
            action_set = game.get_action_space()

            model_turn = (
                opponent_type == "self_play"
                or current_player == model_player
            )

            if not model_turn:
                if opponent_type == "old_self":
                    move = select_model_move(
                        model=old_self_opponent,
                        board=game.board,
                        current_player=current_player,
                        action_set=action_set,
                        board_size=board_size,
                        device=device
                    )
                else:
                    move = select_opponent_move(
                        board=game.board,
                        action_set=action_set,
                        player=current_player,
                        opponent_type=opponent_type,
                        epsilon=opponent_epsilon
                    )

                game.move(move)
                continue

            old_advantage = connection_advantage(game.board, current_player)

            move, action_index, log_prob, value, state, mask = select_action(
                model=model,
                board=game.board,
                current_player=current_player,
                action_set=action_set,
                board_size=board_size,
                device=device
            )

            states.append(state)
            actions.append(action_index)
            old_log_probs.append(log_prob.detach())
            old_values.append(value.detach())
            masks.append(mask)
            players.append(current_player)

            game.move(move)

            new_advantage = connection_advantage(game.board, current_player)
            shaping_reward = shaping_scale * (new_advantage - old_advantage)
            shaping_reward = clip_reward(
                shaping_reward,
                -MAX_SHAPING_REWARD,
                MAX_SHAPING_REWARD
            )
            rewards.append(shaping_reward)

        winner = game.winner

        for index, player in enumerate(players):
            if player == winner:
                rewards[index] += 1.0
            else:
                rewards[index] -= 1.0

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        old_log_probs = torch.stack(old_log_probs).to(device)
        old_values = torch.stack(old_values).to(device)
        masks = torch.stack(masks).to(device)

        returns = compute_returns(rewards, GAMMA, device)
        advantages = returns - old_values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            logits, new_values = model(states)

            masked_logits = torch.full_like(logits, -1e9)
            masked_logits[masks] = logits[masks]

            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surrogate_1 = ratio * advantages
            surrogate_2 = torch.clamp(
                ratio,
                1.0 - CLIP_EPSILON,
                1.0 + CLIP_EPSILON
            ) * advantages

            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = (returns - new_values).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if episode % 100 == 0:
            winner_name = "RED" if winner == RED else "BLUE"
            print(
                f"Episode {episode}/{EPISODES} | "
                f"Board: {board_size}x{board_size} | "
                f"Opponent: {opponent_type} | "
                f"Winner: {winner_name} | "
                f"Loss: {loss.item():.4f}"
            )

        if (
            episode >= OLD_SELF_START_EPISODE
            and episode % OLD_SELF_INTERVAL == 0
        ):
            snapshot_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            old_self_pool.append(create_frozen_model_from_state(snapshot_state, device))

            if len(old_self_pool) > MAX_OLD_SELF_MODELS:
                old_self_pool.pop(0)

            save_old_self_snapshot(model, episode)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
