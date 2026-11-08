"""``python -m forgemind.cli <command>``.

    bench     whole-training-step benchmark (tokens/s, MFU, memory)
    kbench    per-kernel A/B/C: eager vs inductor vs the packed kernels
    matrix    parallel-strategy matrix

``python -m forgemind.cli.bench`` still works (an explicit module wins over a subcommand).
"""
import sys

_COMMANDS = ("bench", "kbench", "matrix")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv and argv[0] in _COMMANDS else "bench"
    rest = argv[1:] if argv and argv[0] in _COMMANDS else argv
    if cmd == "kbench":
        from .kbench import main as run
    elif cmd == "matrix":
        from .matrix import main as run
    else:
        from .bench import main as run
    return run(rest)


if __name__ == "__main__":
    sys.exit(main())
