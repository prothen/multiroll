#!/bin/env python

import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--display', type=str2bool, dest='display_active', help='Enable visualisation', default=True)
parser.add_argument('--step', type=str2bool, dest='step_active', help='Enable visualisation', default=True)
