import argparse


def main():
    parser = argparse.ArgumentParser(description="Start the local Clarity X-ray interface")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--chat-config", default="chatbot_config.json")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        import uvicorn
        from .server import create_app
    except ModuleNotFoundError as error:
        raise SystemExit("Install the interface dependencies: python -m pip install -r requirements-ui.txt") from error
    uvicorn.run(create_app(args.config, args.chat_config), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
