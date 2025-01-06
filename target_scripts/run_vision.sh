#!/usr/bin/env bash

cd "$(dirname $0)"

./start.sh 0 $3

./trace.sh $1_$2 &

sleep 1s

DIR_NAME=../benchmarks/$1_$2
if [ -d $DIR_NAME ] ; then
    echo $DIR_NAME
else
    mkdir -p $DIR_NAME
fi

../vision/$1_$2/$1 -d 0 -p 0 -c 0 -t 30 -l 2 -o $DIR_NAME/timing.csv -b ../vision/$1_$2

sudo kill -s SIGINT $(pidof membw_ctrl)

echo "Run benchmark with rising bandwidths"
./rising_bw.sh $1 $2 $3

touch $DIR_NAME/finished
