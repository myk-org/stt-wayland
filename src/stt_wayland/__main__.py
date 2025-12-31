"""Entry point for stt-daemon command."""

from .daemon import run


def main() -> None:
    """Run the STT daemon."""
    run()


if __name__ == "__main__":
    main()
