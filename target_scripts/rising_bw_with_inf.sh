#!/usr/bin/env bash

cd "$(dirname $0)"

DIR_NAME=../benchmarks/$1_$2/rising
if [ -d $DIR_NAME ] ; then
    echo $DIR_NAME
else
    mkdir -p $DIR_NAME
fi

for((j = 0; j <= 4000; j = j + 500));
do
	NUMBER_INF=$(printf "%08d" $j)
	mkdir -p $DIR_NAME/inf_$NUMBER_INF
	for((i = 100; i <= 4000; i = i + 100));
	do
		echo -ne "BW: ($i/4000), INF: ($j/4000)\r"
		./start.sh $i $j
		NUMBER_BW=$(printf "%08d" $i)
		../vision/$1_$2/$1 -d 0 -p 0 -c 0 -t 1 -l 2 -o $DIR_NAME/inf_$NUMBER_INF/timing_$NUMBER_BW.csv -b ../vision/$1_$2
	done
	head -n 1 $DIR_NAME/inf_$NUMBER_INF/timing_00000100.csv > $DIR_NAME/inf_$NUMBER_INF/rising.csv && tail -n+2 -q $DIR_NAME/inf_$NUMBER_INF/*.csv >> $DIR_NAME/inf_$NUMBER_INF/rising.csv
	rm $DIR_NAME/inf_$NUMBER_INF/timing*
done

sudo kill $(pidof bench)

echo -ne "\n"

exit 0

