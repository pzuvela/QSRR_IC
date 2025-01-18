import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Command Line App to train QSRR IC models"
    )
    parser.add_argument(
        "-f", "--filename",
        type=str,
        required=True,
        help="Path to configuration file (JSON)"
    )
    args = parser.parse_args()
    return args
