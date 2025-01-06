#!/usr/bin/env bash

# echo "Run Benchmark with interference: disparity cif"
# ./rising_bw.sh disparity cif 1000
# echo "Run Benchmark with interference: disparity vga"
# ./rising_bw.sh disparity vga 1000
echo "Run Benchmark with interference: tracking cif"
./rising_bw.sh tracking cif 1000
echo "Run Benchmark with interference: tracking vga"
./rising_bw.sh tracking vga 1000
echo "Run Benchmark with interference: mser vga"
./rising_bw.sh mser vga 1000
echo "Run Benchmark with interference: sift vga"
./rising_bw.sh sift vga 1000

echo "Finished"
exit 0
