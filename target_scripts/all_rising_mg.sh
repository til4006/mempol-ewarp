#!/usr/bin/env bash

# echo "Run Benchmark with interference: disparity cif"
# ./rising_bw.sh disparity cif 1000
# echo "Run Benchmark with interference: disparity vga"
# ./rising_bw.sh disparity vga 1000
echo "Run Benchmark with interference: tracking cif"
./rising_bw_mg.sh tracking cif 0x17
./rising_bw_mg.sh tracking cif 0x18
./rising_bw_mg.sh tracking cif 0x19
echo "Run Benchmark with interference: tracking vga"
./rising_bw_mg.sh tracking vga 0x17
./rising_bw_mg.sh tracking vga 0x18
./rising_bw_mg.sh tracking vga 0x19
echo "Run Benchmark with interference: mser vga"
./rising_bw_mg.sh mser vga 0x17
./rising_bw_mg.sh mser vga 0x18
./rising_bw_mg.sh mser vga 0x19
echo "Run Benchmark with interference: sift vga"
./rising_bw_mg.sh sift vga 0x17
./rising_bw_mg.sh sift vga 0x18
./rising_bw_mg.sh sift vga 0x19

echo "Finished"
exit 0
