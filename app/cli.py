
"""Command-line interface for the Support Response Generator."""
import argparse
from pathlib import Path

from app.core import build_model


def _print_suggestions(prompt: str, count: int, words: int,
                       save_path: str | None = None) -> None:
    model = build_model(n=3)
    suggestions = model.predict(prompt, top_k=count)

    print(f"\nInput: {prompt}")
    print("Next-word suggestions:")

    for word, probability in suggestions:
        print(f"  {word:<15} probability: {probability:.3f}")

    generated = model.generate(prompt, max_words=words)
    print(f"\nSuggested continuation: {generated}")

    if save_path:
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(generated + "\n", encoding="utf-8")
        print(f"Saved response to: {output_file}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="support-response-generator",
        description="Suggest customer-support response continuations using a smoothed n-gram model.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    suggest = sub.add_parser("suggest", help="suggest next words for a phrase")
    suggest.add_argument("text", help="the beginning of a support response")
    suggest.add_argument("--count", type=int, default=3,
                         help="number of next-word suggestions (default: 3)")
    suggest.add_argument("--words", type=int, default=8,
                         help="maximum generated continuation length (default: 8)")
    suggest.add_argument("--save", metavar="FILE",
                         help="save the generated response to a text file")

    sub.add_parser("interactive",
                   help="enter phrases interactively; type 'quit' to exit")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Support Response Generator is ready. Open README.md for your milestones.")

    elif args.command == "suggest":
        if args.count < 1 or args.words < 0:
            parser.error("--count must be positive and --words cannot be negative")

        _print_suggestions(args.text, args.count, args.words, args.save)

    elif args.command == "interactive":
        print("Support Response Generator (type 'quit' to exit)")

        while True:
            try:
                phrase = input("\nStart your response: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if phrase.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break

            if phrase:
                _print_suggestions(phrase, count=3, words=8)


if __name__ == "__main__":
    main()
