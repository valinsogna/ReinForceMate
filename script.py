import argparse
import pandas as pd
import numpy as np
import torch
from src import Q_Learning

def save_results(gamma, lr, iter):
    r = Q_Learning(capture=True, gamma=gamma, lr=lr, verbose=0)
    game = r.learn(iter)

    # Save the game in a pgn file
    with open(f"game_{gamma}_{lr}.pgn", "w", encoding="utf-8") as new_pgn:
        new_pgn.write(str(game)) 

    reward_smooth = pd.DataFrame(r.reward_trace)
    reward_smooth.rolling(window=10, min_periods=0).mean().plot(figsize=(16,9),title='average performance over the last 125 steps')
    reward_smooth.to_csv(f"reward_trace_{gamma}_{lr}.csv")

    r.env.reset()
    bl = r.env.layer_board
    bl[6, :, :] = 1/10  # Assume we are on move 10

    bl_tensor = torch.from_numpy(bl).unsqueeze(0).float()  # Convert bl to a torch.Tensor

    av = r.agent.get_action_values(bl_tensor)
    av = av.reshape((64, 64))

    # Convert av to a NumPy array
    av_np = av.detach().cpu().numpy()

    white_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
    black_piece = ['_', 'p', 'n', 'b', 'r', 'q', 'k']

    df = pd.DataFrame(np.zeros((6, 7)))

    df.index = white_pieces
    df.columns = black_piece

    for from_square in range(16):
        for to_square in range(30, 64):
            from_piece = r.env.board.piece_at(from_square).symbol()
            to_piece = r.env.board.piece_at(to_square)
            if to_piece:
                to_piece = to_piece.symbol()
            else:
                to_piece = '_'
            df.loc[from_piece, to_piece] = av_np[from_square, to_square]

    df[['_','p','n','b','r','q']].to_csv(f"piece_values_{gamma}_{lr}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", help="Discount factor", type=float, default=0.95)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.01)
    parser.add_argument("--iter", help="Iterations", type=int, default=30)
    args = parser.parse_args()

    save_results(args.gamma, args.lr, args.iter)

