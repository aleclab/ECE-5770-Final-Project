# ECE 5770 Final Project

TODOs:
- exception catcher for user inputted ctrl+c that finalizes graphs in the event i need to cut the processing short
- Cache previous "Images/" input argument in a file. Check on execution. If two subsequent launches use identical directories, skip the pre-processing step and reuse original .ppms and such. 
- Further optimzations to CUDA now that we have a robust verification and benchmarking toolchain. Any issues or any improvements achieved would be immediately noticeable. 
- Execute everything on other NVIDIA enabled computers. I've added .exe to repo so this can easily be one. In theory it should just "work" if the computer's NVIDIA drivers are up to date.. right?

Further comparisons: 
- Versions of kernels with certain optimization techniques NOT used to demonstrate impact
- OpenACC approach with compiler directives for comparison
- Potentially benchmark python execution during data prep phase for another sequential bulletpoint, but also that might be accelerated under the hood. 

Done:
- multithreaded C/C++ implementation with timing benchmarks for comparison. Makes use of OpenMP. 
- graph avg runtimes against pixel counts for all 3x3 datapoints (scatterplot maybe)
- launch with windows high-priority mode to try and limit impact of other Windows apps on timing benchmarks
- Enable maximum optimizations (O2) for host code. Re-compile wb library and .cu with "release" mode, reconfigure project settings & get working. 