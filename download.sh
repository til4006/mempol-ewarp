#!/usr/bin/env bash
cd "$(dirname $0)"

MAIN_DIR_NAME=benchmarks/

if [ ! -d "$MAIN_DIR_NAME" ]; then
	mkdir "$MAIN_DIR_NAME"
fi

if [ -z "$1" ]; then
	echo "Usage: $0 <subdir_name>"
	echo "Example: $0 2023-10-01_12-00-00"
	exit 1
fi

DIR_NAME=$MAIN_DIR_NAME$1/



echo "Downloading trace files from remote server..."
scp -r timo@frodo:$MAIN_DIR_NAME $DIR_NAME > /dev/null

for SUB_DIR in $DIR_NAME*; do
	if [ ! -d "$SUB_DIR" ]; then
		continue
	fi
	INPUT_FILE=$SUB_DIR/trace.trc
	OUTPUT_FILE=$SUB_DIR/trace
	echo "Parsing trace file: $INPUT_FILE"
	echo "Output files will be:"
	echo "    $OUTPUT_FILE.info"
	echo "    $OUTPUT_FILE.csv"
	echo "    $OUTPUT_FILE.all"
	./membw_ctrl.host --platform imx8m tdump $INPUT_FILE | grep '#' > $OUTPUT_FILE.info
	./membw_ctrl.host --platform imx8m tdump $INPUT_FILE | grep -v '^#' > $OUTPUT_FILE.csv
	./membw_ctrl.host --platform imx8m tdump $INPUT_FILE > $OUTPUT_FILE.all
done

exit 0
