#!/bin/bash
CW="$(readlink -m "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )")"
if ! [ -d flatland/flatland ]; then
	git submodule update --init --recursive
else
	echo "Already clone repository"
fi
cd flatland && python setup.py install 
		
