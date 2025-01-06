#!/usr/bin/env bash

cd "$(dirname $0)"

DIR_NAME=../benchmarks/$1_$2/rising
if [ -d $DIR_NAME ] ; then
    echo $DIR_NAME
else
    mkdir -p $DIR_NAME
fi

if [ -d $DIR_NAME/mg ] ; then
	echo "Memory guard directory exists"
else
	mkdir -p $DIR_NAME/mg
fi

for((i = 100; i <= 4000; i = i + 100));
do
	cachelines=$(expr $i \* 1000 / 64)
	if [ "$3" = "0x19" ]; then
		cachelines=$(expr $cachelines \* 4)
	fi

	echo -ne "BW: ($i/4000) cachelines=$cachelines\r"
	sudo jailhouse memguard 0 1000 $cachelines $3
	sleep 1s
	NUMBER=$(printf "%08d" $i)
	../vision/$1_$2/$1 -d 0 -p 0 -c 1 -t 1 -l 2 -o $DIR_NAME/mg/timing_$NUMBER.csv -b ../vision/$1_$2
done
echo -ne "\n"
head -n 1 $DIR_NAME/mg/timing_00000100.csv > $DIR_NAME/mg/rising_mg_$3.csv && tail -n+2 -q $DIR_NAME/mg/timing*.csv >> $DIR_NAME/mg/rising_mg_$3.csv

rm $DIR_NAME/mg/timing*
exit 0
