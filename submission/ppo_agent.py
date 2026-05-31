import os
import heapq
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


def count_bridge_connections(board, player):
    size = len(board)
    stones = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if board[row][col] == player
    ]
    bridge_count = 0

    for index, first_cell in enumerate(stones):
        first_neighbors = set(get_neighbors(first_cell[0], first_cell[1], size))

        for second_cell in stones[index + 1:]:
            if second_cell in first_neighbors:
                continue

            second_neighbors = set(get_neighbors(second_cell[0], second_cell[1], size))
            shared_neighbors = first_neighbors & second_neighbors

            if len(shared_neighbors) != 2:
                continue

            if all(board[row][col] == EMPTY for row, col in shared_neighbors):
                bridge_count += 1

    return bridge_count


def connection_score(board, player):
    own_distance = shortest_connection_distance(board, player)
    opponent_distance = shortest_connection_distance(board, -player)
    own_bridges = count_bridge_connections(board, player)
    opponent_bridges = count_bridge_connections(board, -player)

    return 10.0 * (opponent_distance - own_distance) + own_bridges - opponent_bridges


def candidate_positional_moves(board, action_set):
    size = len(board)
    occupied_cells = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if board[row][col] != EMPTY
    ]

    if not occupied_cells:
        return [choose_center_move(board, action_set)]

    candidates = set()
    action_lookup = set(action_set)

    for row, col in occupied_cells:
        for candidate in get_neighbors(row, col, size):
            if candidate in action_lookup:
                candidates.add(candidate)

    if not candidates:
        return action_set

    return list(candidates)


def choose_positional_move(board, action_set, player):
    size = len(board)
    center = (size - 1) / 2

    def score_move(move):
        test_board = deepcopy(board)
        row, col = move
        test_board[row][col] = player

        center_score = -(abs(row - center) + abs(col - center))

        return connection_score(test_board, player), center_score

    return max(candidate_positional_moves(board, action_set), key=score_move)


def load_model(board_size):
    global _loaded_model

    if _loaded_model is not None:
        return _loaded_model

    model = HexCNNPolicy(board_size)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            print(
                "Warning: ppo_cnn_hex.pt is not compatible with the current "
                "fully convolutional model. Retrain to create a new checkpoint."
            )
            return None
        model.eval()
    else:
        print("Warning: ppo_cnn_hex.pt not found. Using center move fallback.")
        return None

    _loaded_model = model

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
        return choose_positional_move(board, action_set, current_player)

    model = load_model(board_size)
    if model is None:
        return choose_positional_move(board, action_set, current_player)

    state = encode_board(board, current_player).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(state)

    logits = logits.squeeze(0)
    masked_logits = torch.full_like(logits, -1e9)

    for move in action_set:
        action_index = action_to_index(move, board_size)
        masked_logits[action_index] = logits[action_index]

    best_action = torch.argmax(masked_logits).item()

    model_move = index_to_action(best_action, board_size)
    positional_move = choose_positional_move(board, action_set, current_player)

    test_model_board = deepcopy(board)
    test_model_board[model_move[0]][model_move[1]] = current_player

    test_positional_board = deepcopy(board)
    test_positional_board[positional_move[0]][positional_move[1]] = current_player

    model_score = connection_score(test_model_board, current_player)
    positional_score = connection_score(test_positional_board, current_player)

    if positional_score >= model_score:
        return positional_move

    return model_move
