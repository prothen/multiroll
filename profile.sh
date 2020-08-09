#!/bin/bash
if ! [ -d ./resources/profiler ]; then 
    echo "Submodules 'Profiler' does not exist. (Probably no access rights)"
else
    echo "Execute"
    ./resources/profiler/generate_pstats_profile.sh
fi
