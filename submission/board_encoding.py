import torch

EMPTY = 0
RED = 1
BLUE = -1


def get_current_player(board):
    red_count = sum(cell == RED for row in board for cell in row)
    blue_count = sum(cell == BLUE for row in board for cell in row)

    if red_count == blue_count:
        return RED

    return BLUE


def encode_board(board, current_player):
    current_player_layer = []
    opponent_layer = []
    empty_layer = []

    opponent = -current_player

    for row in board:
        current_row = []
        opponent_row = []
        empty_row = []

        for cell in row:
            current_row.append(1.0 if cell == current_player else 0.0)
            opponent_row.append(1.0 if cell == opponent else 0.0)
            empty_row.append(1.0 if cell == EMPTY else 0.0)

        current_player_layer.append(current_row)
        opponent_layer.append(opponent_row)
        empty_layer.append(empty_row)

    return torch.tensor(
        [current_player_layer, opponent_layer, empty_layer],
        dtype=torch.float32
    )
