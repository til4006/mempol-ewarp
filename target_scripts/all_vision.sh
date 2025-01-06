#!/usr/bin/env bash

# echo "Run Benchmark: disparity cif"
# ./run_vision.sh disparity cif 0
# echo "Run Benchmark: disparity vga"
# ./run_vision.sh disparity vga 0
echo "Run Benchmark: tracking cif"
./run_vision.sh tracking cif 0
echo "Run Benchmark: tracking vga"
./run_vision.sh tracking vga 0
echo "Run Benchmark: mser vga"
./run_vision.sh mser vga 0
echo "Run Benchmark: sift vga"
./run_vision.sh sift vga 0

echo "Finished"
exit 0
