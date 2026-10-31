import os
import sys
from loguru import logger
from argparse import ArgumentParser

from marco.tasks import *

def main():
    # Leave task-specific --help for the selected task's parser.
    init_parser = ArgumentParser(add_help=False)
    init_parser.add_argument('-m', '--main', type=str, help='The main function to run (required)')
    init_parser.add_argument('--verbose', type=str, default='INFO', choices=['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'], help='The log level')
    init_args, init_extras = init_parser.parse_known_args()

    if init_args.main is None:
        if '-h' in init_extras or '--help' in init_extras:
            init_parser.print_help()
            return
        init_parser.error('the following arguments are required: -m/--main')

    logger.remove()
    terminal_level = init_args.verbose if init_args.verbose in ['DEBUG', 'TRACE'] else 'SUCCESS'
    logger.add(sys.stderr, level=terminal_level)
    os.environ['MARCO_LOG_LEVEL'] = init_args.verbose
    os.makedirs('logs', exist_ok=True)

    try:
        task = eval(init_args.main + 'Task')()
    except NameError:
        logger.error('No such task!')
        return
    task.launch(global_args=vars(init_args), argv=init_extras)

if __name__ == '__main__':
    main()
