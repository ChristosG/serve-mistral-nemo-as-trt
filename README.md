# Guide to Compiling Mistral-Nemo-Instruct-2407 into a TensorRT-LLM Engine and Serving it on Triton Inference Server

This guide provides step-by-step instructions on how to compile the Mistral-Nemo-Instruct-2407 model into a TensorRT-LLM engine and deploy it on the Triton Inference Server.

## Prerequisites

1. **Model Access:**  
   Begin by accessing and downloading the model from the [Mistral-Nemo-Instruct-2407 repository](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) on Hugging Face.

2. **Environment Setup:**  
   Ensure you have the Dockerfile in your working directory. In the same directory, build your container by executing the following command:
   ```bash
   podman build -t <desired_container_name> .  


## Important Notes

- **Base Image:**  
  The Docker image `nvidia/cuda:12.4.1-devel-ubuntu22.04` is equipped with the essential dependencies for immediate commencement of the TensorRT engine building process.

- **TensorRT Updates:**  
  Given that Mistral-Nemo is a relatively new model, it is crucial to install the latest TensorRT packages. As TensorRT evolves, ensure that the TensorRT and PyTorch Python modules (wheels) are rebuilt to maintain a successful integration. This is vital for both engine building and Triton serving containers.

- **Version Consistency:**  
  It is imperative that both the TensorRT engine builder container and the Triton Inference Server container operate on the same TensorRT-LLM version. This ensures compatibility and functionality. The required dependencies are automatically set up either by the Dockerfile or by manually running:
  ```bash
  pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

