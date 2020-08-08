#!/bin/bash
NAME="profiling_report"
FILEPATH="./figures/$NAME"
DESTPATH="./resources/$NAME"
python -m cProfile -o $DESTPATH ./testbed.py --display False >/dev/null 2>&1
gprof2dot -f pstats  $DESTPATH | dot -Tsvg -o $FILEPATH.svg
# inkscape -z -w 2048 -h 1920 $FILEPATH.svg -e $DESTPATH.png
eog $DESTPATH.svg

