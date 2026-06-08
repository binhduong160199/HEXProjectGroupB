import argparse
import statistics
import time
from random import choice

from hex_engine import BLUE, EMPTY, RED, hexPosition
from submission.facade import agent, epsilon_greedy_agent, greedy_agent


def random_agent(board, action_set):
    return choice(action_set)


def timed_agent(board, action_set, timings):
    start = time.perf_counter()
    move = agent(board, action_set)
    timings.append(time.perf_counter() - start)
    return move


def timed_blue_agent(board, action_set, timings):
    return timed_agent(board, action_set, timings)


def get_opponent(name, timings):
    if name == "random":
        return random_agent

    if name == "greedy":
        return greedy_agent

    if name == "epsilon":
        return epsilon_greedy_agent

    if name == "self":
        return lambda board, action_set: timed_blue_agent(board, action_set, timings)

    raise ValueError(f"Unknown opponent: {name}")


def play_game(board_size, opponent, timings):
    game = hexPosition(size=board_size)

    while game.winner == EMPTY:
        action_set = game.get_action_space()

        if game.player == RED:
            move = timed_agent(game.board, action_set, timings)
        else:
            move = opponent(game.board, action_set)

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
    parser.add_argument(
        "--opponent",
        choices=["random", "greedy", "epsilon", "self"],
        default="random",
        help="Opponent to play against. 'self' times the agent for both players.",
    )
    args = parser.parse_args()

    timings = []
    wins = 0
    opponent = get_opponent(args.opponent, timings)

    for _ in range(args.games):
        winner = play_game(args.board_size, opponent, timings)
        if winner == RED:
            wins += 1

    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Games: {args.games}")
    print(f"Opponent: {args.opponent}")
    print(f"Agent wins as RED: {wins}/{args.games}")
    print_summary(timings)


if __name__ == "__main__":
    main()
