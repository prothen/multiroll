#!/bin/env python

import time

S2MS = 1.e3

timestamp = time.time()


def timeme_reset():
    global timestamp
    timestamp = time.time()


def timeme(message):
    global timestamp
    global S2MS
    print(message, '\n\t({:3.2f}ms)'.format((time.time() - timestamp)*S2MS))
    timestamp = time.time()

