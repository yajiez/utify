# -*- coding: utf-8 -*-
import sys
import argparse

commands = {}

parser = argparse.ArgumentParser(
    description="Collection of Utility Functions in Python",
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    "command",
    help="The command to be executed. Use utify [command] -h for help." +
         ("\nCurrently available commands:\n\t + {}".format('\n\t + '.join(commands.keys()))
          if len(commands) > 0 else "\nNo available commands.")
)


def main():
    args = parser.parse_args(sys.argv[1:2])
    if not args.command:
        parser.print_help()
    elif args.command not in commands:
        parser.print_help()
        print(f"Unrecognized command: {args.command}. Please see the above help.")
    else:
        commands[args.command]()


if __name__ == '__main__':
    main()
