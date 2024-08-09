FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs

RUN pip3 uninstall tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs tensorrt-llm torch

RUN pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git

WORKDIR /TensorRT-LLM

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002 

CMD ["/bin/bash"]
