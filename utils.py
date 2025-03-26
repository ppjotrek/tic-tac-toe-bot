import easyAI as eai
import numpy as np
from easyAI import TwoPlayerGame
import random
import time

class TicTacToe(TwoPlayerGame):

    def __init__(self, players, starting_player=1, probabilistic=False):
        self.players = players
        self.board = [0 for i in range(9)]
        self.current_player = starting_player
        self.probabilistic = probabilistic

    def possible_moves(self):
        return [i + 1 for i, e in enumerate(self.board) if e == 0]

    def make_move(self, move):
        self.board[int(move) - 1] = self.current_player

    def unmake_move(self, move):  # optional method (speeds up the AI)
        self.board[int(move) - 1] = 0

    def lose(self):
        """ Has the opponent "three in line ?" """
        return any(
            [
                all([(self.board[c - 1] == self.opponent_index) for c in line])
                for line in [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],  # horiz.
                    [1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9],  # vertical
                    [1, 5, 9],
                    [3, 5, 7],
                ]
            ]
        )  # diagonal

    def is_over(self):
        return (self.possible_moves() == []) or self.lose()

    def show(self):
        pass

    def scoring(self):
        return -100 if self.lose() else 0

    def play(self, decision_times):
        """Override the play method to add probabilistic skipping of moves and measure decision times."""
        while not self.is_over():
            if self.probabilistic and random.random() < 0.2:  #20% chance of skipping
                self.switch_player()
                continue

            start_time = time.time()
            move = self.players[self.current_player - 1].ask_move(self)
            end_time = time.time()
            decision_times[self.current_player - 1].append(end_time - start_time)

            self.make_move(move)
            self.switch_player()


def run_game(number_of_games=100, negamax_1=3, negamax_2=3, probabilistic=False, win_score=float("infinity"), algorithm="negamax"):
    """
    Run a series of games between two AI players.

    :param number_of_games: Number of games to play.
    :param negamax_1: Depth for the first AI player (if using Negamax).
    :param negamax_2: Depth for the second AI player (if using Negamax).
    :param probabilistic: Whether to enable probabilistic skipping of moves.
    :param win_score: The score for a winning move.
    :param algorithm: The algorithm to use ("negamax" or "expectiminimax").
    """
    if algorithm == "negamax":
        ai1 = eai.AI_Player(eai.Negamax(negamax_1, win_score=win_score))
        ai2 = eai.AI_Player(eai.Negamax(negamax_2, win_score=win_score))
    elif algorithm == "expectiminimax":
        ai1 = eai.AI_Player(ExpectiMinimax(negamax_1))
        ai2 = eai.AI_Player(ExpectiMinimax(negamax_2))
    elif algorithm == "mixed":
        ai1 = eai.AI_Player(eai.Negamax(negamax_1, win_score=win_score))
        ai2 = eai.AI_Player(ExpectiMinimax(negamax_2))
    else:
        raise ValueError("Invalid algorithm. Choose 'negamax' or 'expectiminimax'.")

    results = {"AI 1 wins": 0, "AI 2 wins": 0, "Draws": 0}
    decision_times = [[], []]  #decision times container

    for _ in range(number_of_games):
        game = TicTacToe([ai1, ai2], starting_player=random.choice([1, 2]), probabilistic=probabilistic)
        game.play(decision_times)

        if game.lose():
            if game.current_player == 2:
                results["AI 1 wins"] += 1
            else:
                results["AI 2 wins"] += 1
        else:
            results["Draws"] += 1

    #mean decision time
    avg_time_ai1 = sum(decision_times[0]) / len(decision_times[0]) if decision_times[0] else 0
    avg_time_ai2 = sum(decision_times[1]) / len(decision_times[1]) if decision_times[1] else 0

    print("Summary after", number_of_games, "games:")
    print("AI 1 wins:", results["AI 1 wins"])
    print("AI 2 wins:", results["AI 2 wins"])
    print("Draws:", results["Draws"])
    print("Average decision time for AI 1:", round(avg_time_ai1, 4), "seconds")
    print("Average decision time for AI 2:", round(avg_time_ai2, 4), "seconds")


class ExpectiMinimax:
    def __init__(self, depth, alpha=float("-inf"), beta=float("inf")):
        """
        :param depth: max depth of the search tree
        :param alpha: initial alpha value (default: -infinity)
        :param beta: initial beta value (default: infinity)
        """
        self.depth = depth
        self.alpha = alpha
        self.beta = beta

    def __call__(self, game):
        _, move = self.expectiminimax(game, self.depth, self.alpha, self.beta, maximizing=True)
        return move

    def expectiminimax(self, game, depth, alpha, beta, maximizing):
        """
        ExpectiMinimax algorithm with alpha-beta pruning.
        :param game: Current game state.
        :param depth: Remaining depth to search.
        :param alpha: Best value for the maximizing player.
        :param beta: Best value for the minimizing player.
        :param maximizing: True if the current player is maximizing, False otherwise.
        :return: Tuple (evaluation, best_move).
        """
        if depth == 0 or game.is_over():
            return game.scoring(), None

        if maximizing:
            max_eval = float("-inf")
            best_move = None
            for move in game.possible_moves():
                game.make_move(move)
                eval, _ = self.expectiminimax(game, depth - 1, alpha, beta, maximizing=False)
                game.unmake_move(move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:  #prune
                    break
            return max_eval, best_move
        else:
            #random node
            total_eval = 0
            moves = game.possible_moves()
            for move in moves:
                game.make_move(move)
                eval, _ = self.expectiminimax(game, depth - 1, alpha, beta, maximizing=True)
                game.unmake_move(move)
                total_eval += eval
                beta = min(beta, eval)
                if beta <= alpha:  #prune
                    break
            return total_eval / len(moves), None

if __name__ == "__main__":
    run_game(number_of_games=100, negamax_1=3, negamax_2=3, probabilistic=True)