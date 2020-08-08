#!/bin/bash
FILEPATH="../profiling_report"
python -m cProfile -o $FILEPATH ./testbed.py>/dev/null 2>&1
gprof2dot -f pstats  $FILEPATH | dot -Tsvg -o $FILEPATH.svg
# inkscape -z -w 2048 -h 1920 $FILEPATH.svg -e $FILEPATH.png
eog $FILEPATH.svg

