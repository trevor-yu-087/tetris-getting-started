from random import random
from typing import Sequence, Union

import numpy as np
from tetris import Action, BaseAgent, Board, main
from tetris.constants import BOARD_HEIGHT, BOARD_WIDTH

#################################################################
#   Modify the Agent class below to implement your own agent.   #
#   You may define additional methods as you see fit.           #
#################################################################


TRANSLATIONS = [
    [],
    [Action.MOVE_LEFT],
    [Action.MOVE_LEFT] * 2,
    [Action.MOVE_LEFT] * 3,
    [Action.MOVE_LEFT] * 4,
    [Action.MOVE_LEFT] * 5,
    [Action.MOVE_RIGHT],
    [Action.MOVE_RIGHT] * 2,
    [Action.MOVE_RIGHT] * 3,
    [Action.MOVE_RIGHT] * 4,
    [Action.MOVE_RIGHT] * 5,
]

ROTATIONS = [
    [],
    [Action.ROTATE_ANTICLOCKWISE],
    [Action.ROTATE_CLOCKWISE],
    [Action.ROTATE_CLOCKWISE, Action.ROTATE_CLOCKWISE],
]

MOVES = [
            translation + rotation + [Action.HARD_DROP]
            for translation in TRANSLATIONS
            for rotation in ROTATIONS
        ] + [
            rotation + translation + [Action.HARD_DROP]
            for translation in TRANSLATIONS
            for rotation in ROTATIONS
        ]


def calculate_heights(board):
    heights = []

    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[y][x] is not None:
                heights.append(BOARD_HEIGHT - y)
                break
        else:
            heights.append(0)

    return heights


def count_holes(board, heights):
    holes = 0
    for x, h in enumerate(heights):
        if h <= 1:
            continue

        for y in range(BOARD_HEIGHT - h, BOARD_HEIGHT):
            if board[y][x] is None:
                holes += 1

    return holes


class Agent(BaseAgent):
    def __init__(self, weights=[]) -> None:
        super().__init__()

        if len(weights) == 0:
            self.weights = np.array([38, 44, -3, 67, 7])
        else:
            self.weights = weights

    async def play_move(self, board: Board) -> Union[Action, Sequence[Action]]:
        """Makes at least one Tetris move.

        If a sequence of moves is returned, they are made in order
        until the piece lands (with any remaining moves discarded).

        Args:
            board (Board): The Tetris board.

        Returns:
            Union[Action, Sequence[Action]]: The action(s) to perform.
        """

        def score_moves(m):
            b = board.with_moves(m)
            if clearable_lines := b.find_lines_to_clear():
                score_increase = b.clear_lines(clearable_lines)

            # column heights
            column_heights = calculate_heights(b)

            # gaps under blocks
            hole_count = count_holes(b, column_heights)

            # spikiness
            spikiness = [abs(x - y) for x, y in zip(column_heights, column_heights[1:])]
            total_spikiness = sum(spikiness)

            # total number of blocks
            total_blocks = sum(BOARD_WIDTH - line.count(None) for line in b)

            feature_vector = np.array(
                [
                    -len(clearable_lines) + 4,
                    sum(column_heights),
                    hole_count,
                    total_blocks,
                    total_spikiness,
                ]
            )

            return np.dot(self.weights, feature_vector)

        return min(MOVES, key=score_moves)


SelectedAgent = Agent

#####################################################################
#   This runs your agent and communicates with DOXA over stdio,     #
#   so please do not touch these lines unless you are comfortable   #
#   with how DOXA works, otherwise your agent may not run.          #
#####################################################################

if __name__ == "__main__":
    # w = [0.76191086, 0.26673502, 0.26709552, 0.97607697, 0.06224954]
    # w = [2.31305573, 6.29873695, 6.99020742, 2.69680968, 2.44185946]  # GA on 3 seeds [87, 42, 101]
    # w = [2.61429848, 4.1819328, 4.32769645, 8.01483004, 1.43792327]  # GA on 5 seeds [73, 42, 101, 69, 987]
    # w = [9.95, 8.97483004, 5.4419328, 7.85534227, 2.50481351]  # GA on 5 seeds [73, 42, 101, 69, 987]
    w = [7.9735285, 0.21784989, 1.52792327, 1.2061363, 0.34622345]
    agent = SelectedAgent(w)

    main(agent)
