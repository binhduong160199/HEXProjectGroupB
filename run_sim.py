from hex_engine import hexPosition
from submission.facade import greedy_agent, random_agent


def run_games(n_games=50, size=7, red_agent=greedy_agent, blue_agent=random_agent):
    wins = {1: 0, -1: 0}

    for i in range(n_games):
        game = hexPosition(size)
        game.machine_vs_machine(red_agent, blue_agent)
        wins[game.winner] = wins.get(game.winner, 0) + 1

    print(f"Ran {n_games} games (size={size})")
    print(f"Red wins: {wins.get(1,0)}")
    print(f"Blue wins: {wins.get(-1,0)}")


if __name__ == '__main__':
    run_games(n_games=50, size=7)
