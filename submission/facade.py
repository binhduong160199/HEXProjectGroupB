#trivial solution
#def agent (board, action_set):
#    return action_set[0]

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.

import heapq
from copy import deepcopy

EMPTY = 0

HEX_DIRECTIONS = [
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
]


def agent(board, action_set):
    """
    Greedy Hex agent compatible with machine_vs_machine().
    """

    # Player bestimmen
    player = current_player(board)

    best_score = float("-inf")
    best_move = action_set[0]

    for move in action_set:

        new_board = deepcopy(board)

        r, c = move
        new_board[r][c] = player

        my_dist = connection_distance(new_board, player)
        opp_dist = connection_distance(new_board, -player)

        score = opp_dist - my_dist

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def current_player(board):
    """
    Bestimmt wer am Zug ist.
    """

    flat = [cell for row in board for cell in row]

    red_count = flat.count(1)
    blue_count = flat.count(-1)

    return 1 if red_count == blue_count else -1


def connection_distance(board, player):

    n = len(board)

    pq = []
    dist = {}

    if player == 1:
        # RED: top -> bottom
        for c in range(n):
            cost = cell_cost(board[0][c], player)

            heapq.heappush(pq, (cost, 0, c))
            dist[(0, c)] = cost

        target = lambda r, c: r == n - 1

    else:
        # BLUE: left -> right
        for r in range(n):
            cost = cell_cost(board[r][0], player)

            heapq.heappush(pq, (cost, r, 0))
            dist[(r, 0)] = cost

        target = lambda r, c: c == n - 1

    while pq:

        current_cost, r, c = heapq.heappop(pq)

        if target(r, c):
            return current_cost

        for dr, dc in HEX_DIRECTIONS:

            nr = r + dr
            nc = c + dc

            if not (0 <= nr < n and 0 <= nc < n):
                continue

            new_cost = current_cost + cell_cost(board[nr][nc], player)

            if (nr, nc) not in dist or new_cost < dist[(nr, nc)]:

                dist[(nr, nc)] = new_cost

                heapq.heappush(
                    pq,
                    (new_cost, nr, nc)
                )

    return float("inf")


def cell_cost(cell, player):

    if cell == player:
        return 0

    if cell == EMPTY:
        return 1

    return 999