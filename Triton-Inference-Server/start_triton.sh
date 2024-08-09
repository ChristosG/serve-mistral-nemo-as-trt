podman run --rm -it --net host --ulimit memlock=-1 --ulimit stack=67108864  \
                --security-opt=label=disable --security-opt seccomp=unconfined \
                --gpus=all --device nvidia.com/gpu=all \
                --ipc=host \
                --name triton_1008 \
                --tmpfs /tmp:exec \
                -v ./mistral_nemo/Mistral-Nemo-Instruct-2407:/models/mistral-nemo \
                -v ./mistral_nemo/engines:/engines \
                -v ./tensorrtllm_backend:/trtback \
                --user root \
                triton_10_08 /bin/bash
