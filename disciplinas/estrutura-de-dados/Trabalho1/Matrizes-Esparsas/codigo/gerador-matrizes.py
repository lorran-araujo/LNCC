"""
Little script to generate a random giant sparse matrix 

How to use:

    $python giant_matrix 

"""


import sys

import numpy as np


if __name__ == "__main__":
    print("Giant Matrix Constructor! \nArguments:")
    print(sys.argv[1:])
    print()

    if len(sys.argv) != 5:
        print("4 Arguments needed: n_lines, n_cols, random seed, zero_prob")
        exit(1)
    else:
        n_lines, n_cols, seed = [int(i) for i in sys.argv[1: -1]]
        zero_prob = float(sys.argv[-1])
        if zero_prob >= 1.0:
            print("Zero prob greater than 1 will render a matrix full of zeros")
            exit(1)

    np.random.seed(seed)

    arr = (
        np.random.choice( [i for i in range(11)], size = n_lines*n_cols, p = [zero_prob]+[(1-zero_prob)/10 for _ in range(10)])
        .reshape(n_lines, n_cols)
    )

    lines, cols = np.where(arr != 0)
    vals = arr[(lines, cols)]

    lines += 1
    cols += 1

    print(f"{n_lines} {n_cols}")
    for l, c, v in zip(lines, cols, vals):
        print(f"{l} {c} {v:.5f}")
    print("0")



