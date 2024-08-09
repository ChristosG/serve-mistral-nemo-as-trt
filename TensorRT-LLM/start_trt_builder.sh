podman run  --rm -it --net host --ulimit memlock=-1 --ulimit stack=67108864  \
            --security-opt=label=disable --security-opt seccomp=unconfined \
            --gpus=all --device nvidia.com/gpu=all \
            --ipc=host \
            --name cudaDev_08 \
            --tmpfs /tmp:exec \
            -v ./mistral_nemo/Mistral-Nemo-Instruct-2407:/models/mistral-nemo \
            -v ./mistral_nemo/engines:/engines \
            --user root \
            trt_builder_08_08   #this the <desired_container_name> specified on podman build -t <desired_container_name> .
