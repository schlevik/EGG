import argparse
import random

import wandb

from egg import core


def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments controlling the game type
    # arguments concerning the input data and how they are processed
    parser.add_argument(
        "--out", type=str, default=None, help="Path to the train data"
    )
    parser.add_argument(
        "--unseen_ratio", type=float, default=0, help="How many examples to remove."
    )
    parser.add_argument(
        '--max_summand', type=int, default=10, help="Max n that a summand can be."
    )
    parser.add_argument(
        '--size', type=int, default=0, help="How big the resulting dataset will be."
    )
    parser.add_argument(
        '--seed', type=int, default=42, help="Random seed."
    )
    parser.add_argument(
        '--positional', action='store_true', default=False,
        help="If not set, keeps (x,y) examples in eval set if"
             "x != any _x but x == _y for some _x,_y in train set. Vice versa with y."
    )
    parser.add_argument(
        '--sum_seen', action='store_true', default=False,
        help="Discard examples from eval set if their sum is not seen in training set."
    )
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    random.seed(opts.seed)
    print(opts, flush=True)
    full = {(x, y) for x in range(opts.max_summand) for y in range(opts.max_summand)}
    if opts.size:
        full = set(random.sample(full, opts.size))
    if opts.unseen_ratio:
        to_remove = round(len(full) * opts.unseen_ratio)
        eval_set = random.sample(full, to_remove)
        train_set = full.difference(eval_set)
        final_eval_set = []
        for x, y in eval_set:
            x_seen = any(x == _x for _x, _y in train_set) if opts.positional else \
                any(x == _x or x == _y for _x, _y in train_set)
            y_seen = any(y == _x for _x, _y in train_set) if opts.positional else \
                any(y == _x or y == _y for _x, _y in train_set)
            sum_seen = (x + y in map(sum, train_set) or (not opts.sum_seen))

            if x_seen and y_seen and sum_seen:
                final_eval_set.append((x, y))
            else:
                print(f"Discarding {(x, y)}: sum seen: {sum_seen}, x seen: {x_seen}, y seen: {y_seen}")

        print(len(final_eval_set), "eval examples.")
        for t in final_eval_set:
            assert t not in train_set
        with open(f'{opts.out}-eval-l{len(eval_set)}-r{opts.unseen_ratio}-s{opts.seed}.txt', 'w+') as f:
            f.write('\n'.join(f"{x} {y}" for x, y in final_eval_set) + '\n')
    else:
        train_set = full

    print(len(train_set), "train examples.")
    with open(f'{opts.out}-train-l{len(train_set)}-r{opts.unseen_ratio}-s{opts.seed}.txt', 'w+') as f:
        f.write('\n'.join(f"{x} {y}" for x, y in train_set) + '\n')


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
