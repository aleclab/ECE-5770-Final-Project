TST dir

Images: copy test images here with the naming convention "image0.jpg", "image1.jpg", and so on.
Dataset: Directory used by the python script to store generated .ppm files. The script will convert images in the "Images" directory to .ppm, and then perform the filtering operations on these images to produce known good expected output .ppm files for comparison with those produced by each CUDA kernel. 

python prep_and_run_backends.py ^
  --exe "C:\path\to\5770_Hw1.exe" ^
  --images "C:\data\images" ^
  --dataset "C:\data\dataset" ^
  --ops sobel sharpen gaussian ^
  --backends cuda cpu mt ^
  --threads 0 ^
  --trailing-comma-in-i ^
  --out-plots "C:\data\plots" ^
  --csv "C:\data\plots\runs.csv" ^
  --print-cmd --debug
  --scatter --scatter-out "C:\data\plots\time_vs_area.png"
  --win-high-priority 
  --cuda-include-copies        ^  REM: default, but explicit for clarity
  --cuda-include-scales           REM: optional, include CUDA scale kernels too
  --sobel-variants all ^        CUDA Sobel variants to run: 0 1 2 3, or 'all'
  --skip-prep

NOTE: "--threads 0" = omp_get_max_threads() for multi-threaded workload. Values other than 0 allow optionally selecting number of threads. 

--print-cmd --debug for optional verbose printouts 


Example usage on Alec's computer:
python testScript.py --exe "C:/Users/alec7/source/repos/5770_Hw1/x64/Release/5770_Hw1.exe" --images "Images/" --dataset "Dataset/" --ops sobel sharpen gaussian --backends cuda cpu mt --threads 0 --trailing-comma-in-i --out-plots "Graphs/" --csv "CSV/datum.csv" --scatter --scatter-out "Graphs/Scatters/" --win-high-priority --cuda-include-copies --sobel-variants all --skip-prep


python testScript_npp.py --exe "C:/Users/alec7/source/repos/5770_Hw1/x64/Release/5770_Hw1.exe" --images "synth/areaSweep/" --dataset "E:\ultrasize" --ops sobel sharpen gaussian --backends cuda cpu mt --threads 0 --trailing-comma-in-i --out-plots "Graphs/scats/" --csv "CSV/datum.csv" --scatter --scatter-out "Graphs/Scatters/" --win-high-priority --cuda-include-copies --sobel-variants all --opensource-compare

python testScript_npp.py --exe "C:/Users/alec7/source/repos/5770_Hw1/x64/Release/5770_Hw1.exe" --images "ImagesSubset/" --dataset "Dataset/" --ops sobel sharpen gaussian --backends cuda cpu mt --threads 0 --trailing-comma-in-i --out-plots "Graphs/" --csv "CSV/datum.csv" --scatter --scatter-out "Graphs/Scatters/" --win-high-priority --cuda-include-copies --sobel-variants all --opensource-compare

Debuggin MT:
python testScript.py --exe "C:/Users/alec7/source/repos/5770_Hw1/x64/Release/5770_Hw1.exe" --images "ImagesSubset/" --dataset "Dataset/" --ops sobel --backends mt --threads 0 --trailing-comma-in-i --out-plots "Graphs/" --csv "CSV/datum.csv" --scatter --scatter-out "Graphs/Scatters/"




python testScript_npp.py --exe "C:/Users/alec7/Desktop/Ece5770/5770finalproj/SRC/5770_Hw1.exe" --images "synth/areaSweepSubset/" --dataset "Dataset/" --ops sobel sharpen gaussian --backends cuda cpu mt --threads 0 --trailing-comma-in-i --out-plots "Graphs/scats/" --csv "CSV/datum.csv" --scatter --scatter-out "Graphs/Scatters/" --win-high-priority --cuda-include-copies --sobel-variants all --opensource-compare


PLOT ONLY guide:
C:\Users\alec7\Desktop\Oakland\ECE 5770\FInal Project\5770finalproj\TST>python plotOnly.py --csv CSV/datum.csv --out-plots Graphs/NewGraphs/ --ops sobel sharpen gaussian --scatter --scatter-out Graphs/NewGraphs