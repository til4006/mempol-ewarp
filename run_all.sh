
python benchmark.py L2_with_inf/tracking_cif -l -s &
python benchmark.py L2_with_inf/tracking_vga -l -s &
python benchmark.py L2_with_inf/mser_vga -l -s &
python benchmark.py L2_with_inf/sift_vga -l -s &
python benchmark.py L2_no_inf/tracking_cif -l -s &
python benchmark.py L2_no_inf/tracking_vga -l -s &
python benchmark.py L2_no_inf/mser_vga -l -s &
python benchmark.py L2_no_inf/sift_vga -l -s &
wait
