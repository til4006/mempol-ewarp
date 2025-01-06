#!/usr/bin/env bash
cd "$(dirname $0)"

DIR_NAME=../benchmarks/$1
if [ -d $DIR_NAME ] ; then
    echo $DIR_NAME
else
    mkdir -p $DIR_NAME
fi

sudo taskset 0x8 ~/membw_ctrl --platform imx8m trace
sudo taskset 0x8 ~/membw_ctrl --platform imx8m twrite $DIR_NAME/trace.trc
