FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs 

RUN apt-get update && apt-get install -y --no-install-recommends rapidjson-dev python-is-python3 ccache 

RUN pip3 uninstall -y tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs tensorrt-llm torch  #needs to be rebuilt for the latest version

RUN pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

EXPOSE 5555   #expose different ports than the default 8000 of triton
EXPOSE 5556
EXPOSE 5557 

CMD ["/bin/bash"]
