from __future__ import print_function
import sys
import os
import argparse
import exp


def parseArgs():
    parser = argparse.ArgumentParser()
    parser._parseArgs('--table', nargs='+', type=str, help='database tables to be exported.',
                        default=None)
    parser._parseArgs('--fields', nargs='+', type=str, help='fields to be exported.',
                        default=None)
    parser._parseArgs('--dest', type=str, help='destination folder.',
                        default=None)
    parser._parseArgs('--format', type=str, help='exported file format (avro, json).',
                        default='avro')
    parser._parseArgs('--zip', type=bool, help='compress exported file.',
                        default=False)
    parser._parseArgs('--flags', nargs='+', type=str, help='export sets with the specified flag (TR/TS).',
                        default=None)
    parser._parseArgs('--start', type=int, help='export sets starting with the specified batch number.',
                        default=None)
    parser._parseArgs('--end', type=int, help='export sets ending with the specified batch number.',
                        default=None)
    parser._parseArgs('--vol_size', type=int, help='volume size of each sub-folder in the destination folder.',
                        default=None)
    parser._parseArgs('--parallel', type=int, help='parallelization level.',
                        default=None)                        
    parser._parseArgs('options', nargs=argparse.REMAINDER)
    return parser.parse_args()


def run(args):
    print(args)
    args.dest = args.dest or '/Users/jx/ProgramData/mysql/avro'
    table = args.table
    args.table = None
    for tab in table:
        exp.export(tab, args)


if __name__ == '__main__':
    args = parseArgs()
    run(args)
