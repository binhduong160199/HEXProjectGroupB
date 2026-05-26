import os
import torch
from copy import deepcopy

from submission.cnn_policy import HexCNNPolicy
from submission.board_encoding import encode_board, get_current_player


EMPTY = 0
RED = 1
BLUE = -1

MODEL_PATH = "ppo_cnn_hex.pt"

# Cache model to avoid reloading every time the agent is called
_loaded_model = None 
_loaded_board_size = None

# Convert to index for model input/output
def action_to_index(move, board_size):
    row, col = move
    return row * board_size + col

# Convert back from index to board coordinates
def index_to_action(index, board_size):
    row = index // board_size
    col = index % board_size
    return row, col


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

# Check if the given player has a winning path on the board using DFS
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

# Check if placing a piece for the given player in any of the action_set moves results in an immediate win
def find_winning_move(board, action_set, player):
    for move in action_set:
        test_board = deepcopy(board)
        row, col = move
        test_board[row][col] = player

        if has_winning_path(test_board, player):
            return move

    return None

# If no winning/blocking move is found, choose the move closest to the center
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


def load_model(board_size):
    global _loaded_model, _loaded_board_size

    if _loaded_model is not None and _loaded_board_size == board_size:
        return _loaded_model

    model = HexCNNPolicy(board_size)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
    else:
        print("Warning: ppo_cnn_hex.pt not found. Using center move fallback.")

    _loaded_model = model
    _loaded_board_size = board_size

    return model


def cnn_ppo_agent(board, action_set):
    if not action_set:
        return None

    board_size = len(board)
    current_player = get_current_player(board)
    opponent = -current_player

    # 1. Win immediately if possible
    winning_move = find_winning_move(board, action_set, current_player)
    if winning_move is not None:
        return winning_move

    # 2. Block opponent's immediate win
    blocking_move = find_winning_move(board, action_set, opponent)
    if blocking_move is not None:
        return blocking_move

    # 3. Use CNN-PPO
    if not os.path.exists(MODEL_PATH):
        return choose_center_move(board, action_set)

    model = load_model(board_size)
    state = encode_board(board, current_player).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(state)

    logits = logits.squeeze(0)
    masked_logits = torch.full_like(logits, -1e9)

    for move in action_set:
        action_index = action_to_index(move, board_size)
        masked_logits[action_index] = logits[action_index]

    best_action = torch.argmax(masked_logits).item()

    return index_to_action(best_action, board_size)