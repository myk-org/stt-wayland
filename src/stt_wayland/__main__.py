"""Entry point for stt-daemon command."""

import argparse

from .daemon import run


def main() -> None:
    """Run the STT daemon."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text daemon for Wayland",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        default=False,
        help="Enable AI-based typo and grammar correction on transcribed text (default: disabled)",
    )

    args = parser.parse_args()
    run(refine=args.refine)


if __name__ == "__main__":
    main()
