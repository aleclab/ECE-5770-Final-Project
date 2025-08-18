# ECE 5770 Final Project

**Professor Feedback:**
- Include a state-of-the-art open source version of one of the kernels in the testing suite. Could be issues with our implementation that makes multithread appear performant in comparison.
- **Follow-up:** Incorporated comparable npp (Nvidia Performance Primitives) API calls in testing. Similar performance! 

- Occupancy analysis to improve CUDA performance
- **Follow-up:** Supposedly the npp APIs we tested with handle occupancy calculations automatically, and the timing performance of these APIs for the test image set was comparable to our CUDA. This should rule out occupancy inefficiencies as an explanation for any perceived CUDA performance shortcomings. 

- Create and test with still larger images
- **Follow-up:** Attempted to do this. At around 10^9 pixels, encountered issues creating .jpgs of that size with scripts (very slow, consumes 100% RAM, crash). Tried resolving speed issue with multiple "workers' in Python, but forgot about RAM consumption and it casued Windows to crash. Relatedly, decompressing the extremely large, synthetic .jpg files into .ppm files consumes a massive amount of disk space. I dedicated ~600gb to the images used for Batch1 and Batch2 testing. 

- Explore varying image mask sizes.
- **Follow-up:** Did not explore this. 

*********************************************************************************************
**Issues:**

**TODOs:**
- number of images argument 
- OpenACC implementation for comparison

**In progress:**
- Further optimzations to CUDA now that we have a robust verification and benchmarking toolchain. Any issues or any improvements achieved would be immediately noticeable. 
- Execute everything on other NVIDIA enabled computers. 

**Done:** (non-exaustive) 
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
- Per-filter scatter charts to improve readability and other graphing improvements in plotOnly.py (run after main test - didn't get time to update main script). Line displayed on scatter, now produces linear & log scaled y-axis versions of scatter, and no longer erroneously lists a 5th sobel variants. 
- Retested with compute_89,sm_89 (previously was compute_52,sm_52) which are from Maxwell generation (11 year old PTX, 980ti era). Intended to address a cudaSynch failure on laptop but it did not. Didn't notice a performance difference either. 
