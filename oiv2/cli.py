import argparse
import sys

from .cli_utils import Text
from .interpreter import Interpreter


def main():
    # Init interpreter
    interpreter = Interpreter()

    print(Text(text="\nOpen-Interpreter v2 (type 'exit' to quit)\n", color="green"))

    while True:
        try:
            user_input = input(Text(text="You: ", color="blue"))
            if user_input.strip().lower() in {"exit", "quit"}:
                print("Goodbye!")
                return

            print(Text(text="Assistant: ", color="green"), end="", flush=True)
            answer = interpreter.respond(user_input)
            print(answer)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            return
        except Exception as e:
            # Log the error and keep the REPL running.
            print(Text(text="[error] ", color="red") + str(e), file=sys.stderr)


if __name__ == "__main__":
    main()
