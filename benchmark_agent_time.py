import argparse
import statistics
import time
from random import choice

from hex_engine import BLUE, EMPTY, RED, hexPosition
from submission.facade import agent


def random_agent(board, action_set):
    return choice(action_set)


def timed_agent(board, action_set, timings):
    start = time.perf_counter()
    move = agent(board, action_set)
    timings.append(time.perf_counter() - start)
    return move


def play_game(board_size, timings):
    game = hexPosition(size=board_size)

    while game.winner == EMPTY:
        action_set = game.get_action_space()

        if game.player == RED:
            move = timed_agent(game.board, action_set, timings)
        else:
            move = random_agent(game.board, action_set)

        if move not in action_set:
            move = choice(action_set)

        game.move(move)

    return game.winner


def print_summary(timings):
    if not timings:
        print("No agent moves were timed.")
        return

    first_move = timings[0]
    later_moves = timings[1:]
    measured = later_moves if later_moves else timings

    print(f"Timed moves: {len(timings)}")
    print(f"First move: {first_move:.6f} seconds = {first_move / 60:.8f} minutes")
    print(f"Average move: {statistics.mean(measured):.6f} seconds = {statistics.mean(measured) / 60:.8f} minutes")
    print(f"Fastest move: {min(measured):.6f} seconds = {min(measured) / 60:.8f} minutes")
    print(f"Slowest move: {max(measured):.6f} seconds = {max(measured) / 60:.8f} minutes")

    if later_moves:
        print("Average excludes the first move because it may include model loading.")


def main():
    parser = argparse.ArgumentParser(description="Measure PPO CNN agent move time.")
    parser.add_argument("--board-size", type=int, default=11)
    parser.add_argument("--games", type=int, default=10)
    args = parser.parse_args()

    timings = []
    wins = 0

    for _ in range(args.games):
        winner = play_game(args.board_size, timings)
        if winner == RED:
            wins += 1

    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Games: {args.games}")
    print(f"Agent wins as RED vs random: {wins}/{args.games}")
    print_summary(timings)


if __name__ == "__main__":
    main()
