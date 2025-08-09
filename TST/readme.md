TST dir

Images: copy test images here with the naming convention "image0.jpg", "image1.jpg", and so on.
Dataset: Directory used by the python script to store generated .ppm files. The script will convert images in the "Images" directory to .ppm, and then perform the filtering operations on these images to produce known good expected output .ppm files for comparison with those produced by each CUDA kernel. T

Example script usage, with arguments:
python testScript.py ^
  --exe "C:/Users/alec7/source/repos/5770_Hw1/x64/Debug/5770_Hw1.exe" ^
  --images "Images/" ^
  --dataset "Dataset/" ^
  --ops sobel sharpen ^
  --trailing-comma-in-i ^
  --out-plots "Graphs/"

  Or without line-continuation characters so you may press the up arrow and re-execute:

  python testScript.py --exe "C:/Users/alec7/source/repos/5770_Hw1/x64/Debug/5770_Hw1.exe" --images "Images/" --dataset "Dataset/" --ops sobel sharpen --trailing-comma-in-i --out-plots "Graphs/"

  and optionally add --debug print-cmd to see detailed logs from the .exe execution 