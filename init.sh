#!/bin/bash

CWD="$(readlink -m "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )")"
TBA=()
TBA_multiroll_DIR="export PYTHONPATH=${CWD}:\$PYTHONPATH"
echo -e "\nExpose to PYTHONPATH\n\t $TBA_multiroll_DIR"
TBA+=("$TBA_multiroll_DIR")

for i in "${TBA[@]}"
do
    if grep -Fxq "$i" ~/.profile; then echo -e "\nSkip entry: \n $i";
    else echo "$i" >> ~/.profile; echo -e  "\nAdded entry: \n $i"; fi
done
