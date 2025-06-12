import argparse
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser(description="Generate all combinations of hyperparameters.")
    parser.add_argument('--output_file', type=str, default='params.txt',
                        help='Output file to write parameter combinations to')
    # Parse known args first (e.g., --output_file), then the rest as raw args
    known_args, unknown_args = parser.parse_known_args()

    # Parse hyperparameter flags and their values
    hyperparams = {}
    key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg
            hyperparams[key] = []
        else:
            if key is None:
                raise ValueError(f"Value {arg} provided without a corresponding flag.")
            hyperparams[key].append(arg)

    return known_args.output_file, hyperparams

def main():
    output_file, hyperparams = parse_args()

    # Create all combinations
    keys = list(hyperparams.keys())
    values = list(product(*(hyperparams[key] for key in keys)))

    with open(output_file, 'w') as f:
        for combination in values:
            line = ' '.join(f'{key} {val}' for key, val in zip(keys, combination))
            f.write(line + '\n')

if __name__ == '__main__':
    main()


