# ECE 5770 Final Project

**Issues:**
- npp timing results for sobel not being printed and or captured properly

**TODOs:**
- number of images argument 
- OpenACC implementation for comparison

- include information about both processors in report
- re-run laptop benchmark with threadcount set to something other than default (though it appears to be performing well with 22 threads now?)
- share gitub with prof
- Unique scatter plots for each data set to avoid clutter and confusion, i.e. scatter on a per-workflow basis. j
- apply fixes to logging after desktop benchmark workload is finished
- retest with compute_89,sm_89 (previously was compute_52,sm_52) which are from Maxwell generation (11 year old PTX, 980ti era)
- fix double printing of "CUDA" option in scatterplots - adds to clutter problem

- Professor Feedback:
- Occupancy analysis to improve CUDA performance
- Include a state-of-the-art open source version of one of the kernels in the testing suite
- Create and test with still larger images

**In progress:**
- Further optimzations to CUDA now that we have a robust verification and benchmarking toolchain. Any issues or any improvements achieved would be immediately noticeable. 

- Potentially benchmark python execution during data prep phase for another sequential bulletpoint, but also that might be accelerated under the hood. 
- Execute everything on other NVIDIA enabled computers. 

**Done:**
- introduced variants of Sobel kernel with various optimization techniques employed and/or omitted to investigate performance impact. 
- multithreaded C/C++ implementation with timing benchmarks for comparison. Makes use of OpenMP. 
- graph avg runtimes against pixel counts for all 3x3 datapoints (scatterplot maybe)
- launch with windows high-priority mode to try and limit impact of other Windows apps on timing benchmarks
- Enable maximum optimizations (O2) for host code. Re-compile wb library and .cu with "release" mode, reconfigure project settings & get working. 
- exception catcher for user inputted ctrl+c that finalizes graphs in the event i need to cut the processing short
- include dev2host and host2dev mem transfers in CUDA timing measurements. includes the time taken for transferring data to and from the GPU. 
- option to skip the pre-processing step and reuse original .ppms and such. 
- introduced copies of extremely small and extremely large images in the dataset to tease out any performance differences at extremes 
- extremely large and extremely small images to help with graph 
- Versions of kernels with certain optimization techniques NOT used to demonstrate impact
