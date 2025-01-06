#!/usr/bin/env bash

# echo "Run Benchmark: disparity cif"
# ./run_vision.sh disparity cif 200
# echo "Run Benchmark: disparity vga"
# ./run_vision.sh disparity vga 200
echo "Run Benchmark: tracking cif"
./run_vision.sh tracking cif 200
echo "Run Benchmark: tracking vga"
./run_vision.sh tracking vga 200
echo "Run Benchmark: mser vga"
./run_vision.sh mser vga 200
echo "Run Benchmark: sift vga"
./run_vision.sh sift vga 200

echo "Finished"
exit 0
