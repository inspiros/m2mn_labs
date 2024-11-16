import argparse
import os

from entropy_estimator import EntropyEstimator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default=os.path.join(os.path.dirname(__file__), 'resources/Declaration1789.txt'))
    return parser.parse_args()


def main():
    args = parse_args()
    # load data
    with open(args.input, 'r') as f:
        txt_data = ''.join(f.readlines())

    ent = EntropyEstimator(txt_data)
    print(f'Zero order entropy: {ent.entropy()}')
    print(f'First order entropy: {ent.entropy1()}')


if __name__ == '__main__':
    main()
