from __future__ import print_function
import sys
import os
import argparse
import exp

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', nargs='+', type=str, help='database tables to be exported.',
                        default=None)
    parser.add_argument('--dest', type=str, help='destination folder.',
                        default=None)
    return parser.parse_args()

def run(args):
    dest = args.dest or '/Users/jx/ProgramData/mysql/avro'
    for tab in args.table:
        exp.export(tab, dest)

if __name__ == '__main__':
    args = parseArgs()
    run(args)