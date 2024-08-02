#!/bin/bash
workload=$1
split=$2
# @fcx
# port=${3:-5432}
port=${3:-5432}

# # clear cache
# echo "python clear_cache.py -w JOB -p $port"

# python clear_cache.py -w JOB -p $port

# echo "python clear_cache.py -w STACK -p $port"

# python clear_cache.py -w STACK -p $port

# echo "python clear_cache.py -w TPCDS -p $port"

# python clear_cache.py -w TPCDS -p $port

# pre warm
echo "python pre_warm.py -w $workload -p $port"

python pre_warm.py -w $workload -p $port

# run workload

if test $workload = "JOB"; then
    echo "python run_workload.py -w $workload -s $split -m test -p $port"
    python run_workload.py -w $workload -s $split -m test -p $port
else
    echo "python run_workload.py -w $workload -m test -p $port"
    python run_workload.py -w $workload -m test -p $port
fi
