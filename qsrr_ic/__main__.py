from qsrr_ic.arg_parser import get_args
from qsrr_ic import main

if __name__ == "__main__":
    args = get_args()
    main(args.filename)
