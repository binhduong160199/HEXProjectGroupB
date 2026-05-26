import torch
import torch.optim as optim
from torch.distributions import Categorical

from hex_engine import hexPosition, EMPTY, RED, BLUE
from submission.cnn_policy import HexCNNPolicy
from submission.board_encoding import encode_board


BOARD_SIZE = 7
EPISODES = 3000

GAMMA = 0.99
PPO_EPOCHS = 4
CLIP_EPSILON = 0.2
LEARNING_RATE = 0.0003

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

    game = hexPosition(size=BOARD_SIZE)
    model = HexCNNPolicy(board_size=BOARD_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for episode in range(1, EPISODES + 1):
        game.reset()

        states = []
        actions = []
        old_log_probs = []
        old_values = []
        masks = []
        players = []

        while game.winner == EMPTY:
            current_player = game.player
            action_set = game.get_action_space()

            move, action_index, log_prob, value, state, mask = select_action(
                model=model,
                board=game.board,
                current_player=current_player,
                action_set=action_set,
                board_size=BOARD_SIZE,
                device=device
            )

            states.append(state)
            actions.append(action_index)
            old_log_probs.append(log_prob.detach())
            old_values.append(value.detach())
            masks.append(mask)
            players.append(current_player)

            game.move(move)

        winner = game.winner

        rewards = []
        for player in players:
            if player == winner:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)

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
                f"Winner: {winner_name} | "
                f"Loss: {loss.item():.4f}"
            )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()