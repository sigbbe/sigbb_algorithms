from optparse import OptionParser

import numpy as np
from matplotlib import pyplot as plt
from traitlets.traitlets import default

all_arguments = [
    {
        'flags': ['-f', '--input-file'],
        'dest': 'input',
        'help': 'filename containing csv file',
        'default': None,
        'type': str
    },
    {
        'flags': ['-s', '--min-support'],
        'dest': 'min_support',
        'help': 'minimum support value',
        'default': 0.15,
        'type': float
    },
    {
        'flags': ['-c', '--min-confidence'],
        'dest': 'min_confidence',
        'help': 'minimum confidence value',
        'default': 0.6,
        'type': float
    },
    {
        'flags': ['min_pts=3, e_ps=3', '--minConfidence'],
        'dest': 'min_confidence',
        'help': 'minimum confidence value',
        'default': 0.6,
        'type': float
    },
    {
        'flags': ['e_ps=3', '--minConfidence'],
        'dest': 'min_confidence',
        'help': 'minimum confidence value',
        'default': 0.6,
        'type': float
    }
]


def parse_args(
    file=False,
    min_support=False,
    min_confidence=False
):
    opttion_parser = OptionParser()
    arguments = [file, min_support, min_confidence]
    for i in range(len(arguments)):
        arg = arguments[i]
        if (arg):
            arg_info = all_arguments[i]
            opttion_parser.add_option(
                arg_info['flags'][0],
                arg_info['flags'][1],
                dest=arg_info['dest'],
                help=arg_info['help'],
                default=arg_info['default'],
                type=arg_info['type']
            )

    options, args = opttion_parser.parse_args()
    print(options)
    print(args)

    # opttion_parser.add_option('-f', '--inputFile',
    #                           dest='input',
    #                           help='filename containing csv',
    #                           default=None)
    # opttion_parser.add_option('-s', '--minSupport',
    #                           dest='minS',
    #                           help='minimum support value',
    #                           default=0.15,
    #                           type='float')
    # opttion_parser.add_option('-c', '--minConfidence',
    #                           dest='minC',
    #                           help='minimum confidence value',
    #                           default=0.6,
    #                           type='float')


class Arg():
    def __init__(self):
        return None


if __name__ == '__main__':
    parse_args(file=True, min_support=True)
