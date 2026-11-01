import argparse
import json
from pathlib import Path

from .config import ChatbotConfig
from .explainer import ReportExplainer

EXAMPLES = Path(__file__).parent / "examples"


def main():
    parser = argparse.ArgumentParser(description="Explain report wording using a local Ollama model")
    parser.add_argument("--config", default="chatbot_config.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--report", type=Path, help="UTF-8 plain-text report; use fictional examples initially")
    source.add_argument("--example", choices=["negation", "uncertainty", "follow_up"], default=None)
    parser.add_argument("--question", default="Please explain this report in plain language.")
    parser.add_argument("--chat", action="store_true", help="Ask follow-up questions; /quit ends the session")
    parser.add_argument("--dry-run", action="store_true", help="Print the request messages without contacting Ollama")
    args = parser.parse_args()
    try:
        config = ChatbotConfig.load(args.config)
        report_path = args.report or EXAMPLES / f"{args.example or 'uncertainty'}.txt"
        session = ReportExplainer(report_path.read_text(encoding="utf-8-sig"), config)
        if args.dry_run:
            print(json.dumps(session.messages_for(args.question), indent=2, ensure_ascii=False))
            return
        print(f"My Report, Explained | Local model: {config.model}")
        print("Educational prototype. Answers can be wrong; check them against the source report.")
        print("The app does not save reports or chat history to disk.")
        print("\n" + session.ask(args.question))
        if args.chat:
            print("\nAsk a follow-up question, or type /quit.")
            while True:
                try:
                    question = input("You: ").strip()
                except EOFError:
                    break
                if question.lower() in {"/quit", "/exit"}:
                    break
                if not question:
                    continue
                print("\nAssistant: " + session.ask(question) + "\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        print("\nSession ended.")


if __name__ == "__main__":
    main()
