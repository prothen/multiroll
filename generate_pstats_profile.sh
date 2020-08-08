#!/bin/bash
NAME="profiling_report"
python -m cProfile -o $NAME testbed.py>/dev/null 2>&1
gprof2dot -f pstats  $NAME | dot -Tsvg -o $NAME.svg
#inkscape -z -w 2048 -h 1920 $NAME.svg -e $NAME.png
#rm $NAME.svg
#eog $NAME.png
eog $NAME.svg

