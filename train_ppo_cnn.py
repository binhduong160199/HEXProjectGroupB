import torch
import torch.optim as optim
from torch.distributions import Categorical
import heapq

from hex_engine import hexPosition, EMPTY, RED, BLUE
from submission.cnn_policy import HexCNNPolicy
from submission.board_encoding import encode_board


EPISODES = 3000

CURRICULUM_PHASES = [
    (0.20, 5),
    (0.40, 5),
    (0.60, 7),
    (0.80, 9),
    (1.00, 11),
]

GAMMA = 0.99
PPO_EPOCHS = 4
CLIP_EPSILON = 0.2
LEARNING_RATE = 0.0003
SHAPING_REWARD_SCALE = 0.05
MAX_SHAPING_REWARD = 0.2

MODEL_PATH = "ppo_cnn_hex.pt"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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


def get_curriculum_board_size(episode):
    progress = episode / EPISODES

    for phase_end, board_size in CURRICULUM_PHASES:
        if progress <= phase_end:
            return board_size

    return CURRICULUM_PHASES[-1][1]


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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for episode in range(1, EPISODES + 1):
        board_size = get_curriculum_board_size(episode)
        game = hexPosition(size=board_size)

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
            shaping_reward = SHAPING_REWARD_SCALE * (new_advantage - old_advantage)
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
                f"Winner: {winner_name} | "
                f"Loss: {loss.item():.4f}"
            )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
