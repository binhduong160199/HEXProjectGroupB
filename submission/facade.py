from random import choice, random
from copy import deepcopy
from submission.ppo_agent import cnn_ppo_agent

# Board encoding
EMPTY = 0
RED = 1
BLUE = -1


# ============================================================
# RANDOM AGENT
# ============================================================
# Chooses a completely random valid move.
# Used as the simplest baseline agent.
def random_agent(board, action_set):
    if not action_set:
        return None

    return choice(action_set)


# ============================================================
# GREEDY AGENT
# ============================================================
# Uses simple heuristic rules:
# 1. Win immediately if possible
# 2. Block opponent's immediate win
# 3. Otherwise play near the center
def greedy_agent(board, action_set):
    if not action_set:
        return None

    current_player = get_current_player(board)

    # --------------------------------------------------------
    # Rule 1:
    # Try every move and check whether it wins immediately.
    # --------------------------------------------------------
    for move in action_set:
        test_board = deepcopy(board)

        row, col = move
        test_board[row][col] = current_player

        # DFS path search
        if has_winning_path(test_board, current_player):
            return move

    opponent = -current_player

    # --------------------------------------------------------
    # Rule 2:
    # Check whether opponent could win next move.
    # If yes -> block that move.
    # --------------------------------------------------------
    for move in action_set:
        test_board = deepcopy(board)

        row, col = move
        test_board[row][col] = opponent

        if has_winning_path(test_board, opponent):
            return move

    # --------------------------------------------------------
    # Rule 3:
    # If no immediate win/block exists,
    # choose move closest to center.
    # --------------------------------------------------------
    return choose_center_move(board, action_set)


# ============================================================
# EPSILON-GREEDY AGENT
# ============================================================
# Mostly greedy, but sometimes random.
# Adds exploration.
def epsilon_greedy_agent(board, action_set, epsilon=0.2):
    if not action_set:
        return None

    # Exploration
    if random() < epsilon:
        return random_agent(board, action_set)

    # Exploitation
    return greedy_agent(board, action_set)


# ============================================================
# MAIN EXPORTED AGENT
# ============================================================
# This is the function used by the Hex engine.
# Switch the return line to test different agents.
def agent(board, action_set):

    return cnn_ppo_agent(board, action_set)

    # return random_agent(board, action_set)

    # return greedy_agent(board, action_set)

    # return epsilon_greedy_agent(board, action_set, epsilon=0.2)


# ============================================================
# DETERMINE CURRENT PLAYER
# ============================================================
# Infer whose turn it is from the number of stones.
# Red always starts first.
def get_current_player(board):
    red_count = 0
    blue_count = 0

    for row in board:
        for cell in row:
            if cell == RED:
                red_count += 1

            elif cell == BLUE:
                blue_count += 1

    if red_count == blue_count:
        return RED

    return BLUE


# ============================================================
# CENTER HEURISTIC
# ============================================================
# Hex center positions are usually strategically stronger.
# This function chooses the move closest to center.
def choose_center_move(board, action_set):
    size = len(board)

    center = (size - 1) / 2

    best_move = None
    best_distance = float("inf")

    for row, col in action_set:

        # Manhattan distance to center
        distance = abs(row - center) + abs(col - center)

        if distance < best_distance:
            best_distance = distance
            best_move = (row, col)

    return best_move


# ============================================================
# GET NEIGHBORING HEX CELLS
# ============================================================
# Each hex cell has up to 6 neighbors.
def get_neighbors(row, col, size):

    candidates = [
        (row - 1, col),     # up
        (row + 1, col),     # down
        (row, col - 1),     # left
        (row, col + 1),     # right
        (row - 1, col + 1), # up-right
        (row + 1, col - 1), # down-left
    ]

    # Keep only valid board coordinates
    return [
        (r, c)
        for r, c in candidates
        if 0 <= r < size and 0 <= c < size
    ]


# ============================================================
# DFS WIN DETECTION
# ============================================================
# Checks whether a player has connected their target sides.
#
# RED:
# left -> right
#
# BLUE:
# top -> bottom
#
# Uses DFS graph traversal.
def has_winning_path(board, player):

    size = len(board)

    visited = set()

    # DFS stack
    stack = []

    # ========================================================
    # RED SEARCH
    # ========================================================
    if player == RED:

        # Start DFS from all RED cells on left edge
        for row in range(size):

            if board[row][0] == RED:

                stack.append((row, 0))

                visited.add((row, 0))

        # DFS traversal
        while stack:

            row, col = stack.pop()

            # RED wins if reaching right edge
            if col == size - 1:
                return True

            # Explore neighbors
            for nr, nc in get_neighbors(row, col, size):

                if board[nr][nc] == RED and (nr, nc) not in visited:

                    visited.add((nr, nc))

                    stack.append((nr, nc))

    # ========================================================
    # BLUE SEARCH
    # ========================================================
    elif player == BLUE:

        # Start DFS from all BLUE cells on top edge
        for col in range(size):

            if board[0][col] == BLUE:

                stack.append((0, col))

                visited.add((0, col))

        while stack:

            row, col = stack.pop()

            # BLUE wins if reaching bottom edge
            if row == size - 1:
                return True

            for nr, nc in get_neighbors(row, col, size):

                if board[nr][nc] == BLUE and (nr, nc) not in visited:

                    visited.add((nr, nc))

                    stack.append((nr, nc))

    return False