podman run --rm -it --net host --ulimit memlock=-1 --ulimit stack=67108864  \
                --security-opt=label=disable --security-opt seccomp=unconfined \
                --gpus=all --device nvidia.com/gpu=all \
                --ipc=host \
                --name triton_08_08 \
                --tmpfs /tmp:exec \
                -v ./mistral_nemo/Mistral-Nemo-Instruct-2407:/models/mistral-nemo \
                -v ./mistral_nemo/engines:/engines \
                -v ./mistral-nemo-trt/cudaDev/tensorrtllm_backend:/trtback \
                --user root \
                nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
