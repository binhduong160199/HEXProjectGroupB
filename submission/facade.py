from random import choice, random


EMPTY = 0
RED = 1
BLUE = -1
EPSILON = 0.0
APPROACH = "greedy"  # "random", "greedy", or "epsilon_greedy"


def infer_player(board):
    """Infer whose turn it is from the number of stones on the board."""
    red_count = 0
    blue_count = 0

    for row in board:
        for cell in row:
            if cell == RED:
                red_count += 1
            elif cell == BLUE:
                blue_count += 1

    return RED if red_count == blue_count else BLUE


def neighbors(size, row, col):
    """Return valid neighboring coordinates for a given cell."""
    candidates = [
        (row - 1, col), # up
        (row + 1, col), # down
        (row, col - 1), # left
        (row, col + 1), # right
        (row - 1, col + 1), # up-right
        (row + 1, col - 1), # down-left
    ]

    return [
        (r, c)
        for r, c in candidates
        if 0 <= r < size and 0 <= c < size
    ]


def count_friendly_neighbors(board, move, player):
    """Count how many neighboring cells contain the player's stones."""
    size = len(board)
    row, col = move
    score = 0

    for n_row, n_col in neighbors(size, row, col):
        if board[n_row][n_col] == player:
            score += 1

    return score


def count_opponent_neighbors(board, move, player):
    """Count how many neighboring cells contain the opponent's stones."""
    size = len(board)
    row, col = move
    opponent = -player
    score = 0

    for n_row, n_col in neighbors(size, row, col):
        if board[n_row][n_col] == opponent:
            score += 1

    return score


def bridge_score(board, move, player):
    """
    Reward moves that form Hex bridge-like structures.
    A bridge is an indirect but strong virtual connection between two friendly stones.
    """
    size = len(board)
    row, col = move
    score = 0

    # Candidate bridge partner positions
    bridge_offsets = [
        (-1, -1),
        (-2, 1),
        (-1, 2),
        (1, 1),
        (2, -1),
        (1, -2),
    ]

    for dr, dc in bridge_offsets:
        r, c = row + dr, col + dc

        if not (0 <= r < size and 0 <= c < size):
            continue

        if board[r][c] == player:
            score += 1

    return score


def edge_progress_score(move, player, size):
    """Score based on how close the move is to the player's target edge.
    This gives higher values near the middle of the player’s connection direction, and lower values at the far edges."""
    row, col = move

    if player == RED:
        return min(col + 1, size - col)

    return min(row + 1, size - row)


def center_score(move, size):
    """Score based on proximity to the center of the board.
    gives a score based on how close the move is to the center of the board. 
    Moves closer to the center get higher scores, while moves near the edges get lower scores."""
    row, col = move
    center = (size - 1) / 2

    return size - (abs(row - center) + abs(col - center))


def heuristic_score(board, move, player):
    """
    Simple baseline heuristic: combine above factors with weights to score the move.
    the agent cares most about connecting to its own stones, 
    then about playing near opponent stones, 
    then about directional progress, then about staying near the center.
    """
    size = len(board)

    return (
        3 * count_friendly_neighbors(board, move, player)
        + 3 * count_opponent_neighbors(board, move, player)
        + edge_progress_score(move, player, size)
        + 4 * bridge_score(board, move, player)
        + 0.5 * center_score(move, size)
    )

def random_agent(board, action_set):
    """Choose a random legal move."""
    return choice(action_set)   

def greedy_agent(board, action_set):
    """Choose the legal move with the best immediate heuristic score."""
    player = infer_player(board)

    return max(
        action_set,
        key=lambda move: heuristic_score(board, move, player),
    )


def epsilon_greedy_agent(board, action_set, epsilon=EPSILON):
    """
    Mostly follow the greedy policy, but explore randomly with probability epsilon.
    """
    if random() < epsilon:
        return random_agent(board, action_set)

    return greedy_agent(board, action_set)



def agent(board, action_set):
    """
    Default exported agent used by the provided Hex runners.
    """
    if APPROACH == "random":
        return random_agent(board, action_set)

    if APPROACH == "greedy":
        return greedy_agent(board, action_set)

    if APPROACH == "epsilon_greedy":
        return epsilon_greedy_agent(board, action_set)







