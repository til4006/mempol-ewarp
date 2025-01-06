#!/usr/bin/env bash
cd "$(dirname $0)"

DIR_NAME=benchmarks/

scp -r timo@frodo:$DIR_NAME .

for SUB_DIR in $DIR_NAME*; do
	INPUT_FILE=$SUB_DIR/trace.trc
	OUTPUT_FILE=$SUB_DIR/trace
	echo $INPUT_FILE
	echo $OUTPUT_FILE
	./membw_ctrl.host --platform imx8m tdump $INPUT_FILE | grep '#' > $OUTPUT_FILE.info
	./membw_ctrl.host --platform imx8m tdump $INPUT_FILE | grep -v '^#' > $OUTPUT_FILE.csv
	./membw_ctrl.host --platform imx8m tdump $INPUT_FILE > $OUTPUT_FILE.all
done

exit 0
