# Custom_DeepStream
## Abstract
This project aims to create a customized pipeline (array input to rtsp output).

## Pipeline Introduction

Generating a Deepstream pipeline is roughly divided into four parts: <br>
1. Create element
2. Set the property of each element
3. Add element into the pipeline
4. Link the required elements in order

## Our customized Pipeline Introduction

- Goal: Implement input array to output rtsp
- Pipeline: Appsrc -> nvvideoconvert 1 -> capsfilter1 -> streammux -> nvvideoconvert2 -> capsfilter2 -> h264encoder -> h264parser -> rtppay -> udpsink

## Pipeline Visualization

![deepstream_test3_pipeline](https://user-images.githubusercontent.com/56544982/170412439-79d36162-675e-432d-b880-f18b481f9193.png)


## Usage
- Run graph_pipeline function inside eric3_final_class.py to visualize the pipeline.
- Run live_demo.py to confirm that the pipeline is established successfully.


## Demo
1. https://www.youtube.com/watch?v=BF-mr3Fs-Hw&ab_channel=TechnologyGoldenRetriever
2. https://www.youtube.com/watch?v=LdA_vVY89II&ab_channel=TechnologyGoldenRetriever

## Reference
1. https://blog.csdn.net/qq_38032876/article/details/109820358
2. https://blog.csdn.net/Tosonw/article/details/104286028
3. https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test1-rtsp-out
4. https://forums.developer.nvidia.com/t/appsrc-with-numpy-input-in-python/120611





