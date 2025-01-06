python benchmark.py disparity_cif -l -s &
python benchmark.py disparity_vga -l -s &
python benchmark.py tracking_cif -l -s &
python benchmark.py tracking_vga -l -s &
python benchmark.py mser_vga -l -s &
# python benchmark.py sift_vga -l -s &
wait
