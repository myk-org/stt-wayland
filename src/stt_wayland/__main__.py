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
    parser.add_argument(
        "--instruction-keyword",
        type=str,
        default=None,
        help="Keyword to separate content from AI instructions (e.g., 'boom'). If not set, feature is disabled.",
    )
    parser.add_argument(
        "--ask-keyword",
        type=str,
        default=None,
        help="Keyword at start of speech triggers AI query mode - AI's answer is typed instead of transcription",
    )

    args = parser.parse_args()
    run(refine=args.refine, instruction_keyword=args.instruction_keyword, ask_keyword=args.ask_keyword)


if __name__ == "__main__":
    main()
