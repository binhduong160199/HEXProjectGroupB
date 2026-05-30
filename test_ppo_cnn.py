from random import choice, random
from copy import deepcopy

from hex_engine import hexPosition, EMPTY, RED, BLUE
from submission.ppo_agent import cnn_ppo_agent

def get_current_player(board):
    red_count = sum(cell == RED for row in board for cell in row)
    blue_count = sum(cell == BLUE for row in board for cell in row)

    if red_count == blue_count:
        return RED

    return BLUE


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

    if player == BLUE:
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


def random_agent(board, action_set):
    return choice(action_set)


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
    current_player = get_current_player(board)

    for move in action_set:
        test_board = deepcopy(board)
        row, col = move
        test_board[row][col] = current_player

        if has_winning_path(test_board, current_player):
            return move

    opponent = -current_player

    for move in action_set:
        test_board = deepcopy(board)
        row, col = move
        test_board[row][col] = opponent

        if has_winning_path(test_board, opponent):
            return move

    return choose_center_move(board, action_set)

def epsilon_greedy_agent(board, action_set, epsilon=0.2):
    if random() < epsilon:
        return random_agent(board, action_set)

    return greedy_agent(board, action_set)

def play_game(board_size, red_agent, blue_agent):
    game = hexPosition(size=board_size)

    while game.winner == EMPTY:
        action_set = game.get_action_space()

        if game.player == RED:
            move = red_agent(game.board, action_set)
        else:
            move = blue_agent(game.board, action_set)

        if move not in action_set:
            move = choice(action_set)

        game.move(move)

    return game.winner


def evaluate(board_size, opponent_agent, games=100):
    wins = 0

    for i in range(games):
        if i % 2 == 0:
            winner = play_game(
                board_size=board_size,
                red_agent=cnn_ppo_agent,
                blue_agent=opponent_agent
            )

            if winner == RED:
                wins += 1

        else:
            winner = play_game(
                board_size=board_size,
                red_agent=opponent_agent,
                blue_agent=cnn_ppo_agent
            )

            if winner == BLUE:
                wins += 1

    return wins / games


def main():
    board_size = 11
    games = 500

    win_rate_random = evaluate(board_size, random_agent, games)
    win_rate_greedy = evaluate(board_size, greedy_agent, games)
    win_rate_epsilon = evaluate(board_size, epsilon_greedy_agent, games)

    print(f"CNN-PPO win rate vs epsilon-greedy: {win_rate_epsilon * 100:.2f}%")
    print(f"CNN-PPO win rate vs random: {win_rate_random * 100:.2f}%")
    print(f"CNN-PPO win rate vs greedy: {win_rate_greedy * 100:.2f}%")


if __name__ == "__main__":
    main()
