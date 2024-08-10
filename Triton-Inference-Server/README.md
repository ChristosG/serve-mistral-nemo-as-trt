# Deploying Mistral-Nemo-Instruct-2407 on Triton Inference Server

This guide details the steps to deploy the Mistral-Nemo-Instruct-2407 model using TensorRT-LLM and Triton Inference Server.

## Starting the Triton Inference Server

After successfully creating the container, run the `./start_triton.sh` script to start the Triton Inference Server container. Then, execute the following command to launch the Triton server:

```bash
tritonserver --model-repository=/trtback/all_models/inflight_batcher_llm/ \
             --model-control-mode=explicit \
             --load-model=preprocessing \
             --load-model=postprocessing \
             --load-model=tensorrt_llm \
             --load-model=tensorrt_llm_bls \
             --load-model=ensemble \
             --log-verbose=2 \
             --log-info=1 \
             --log-warning=1 \
             --log-error=1 \
             --http-port=5555 \
             --grpc-port=5556 \
             --metrics-port=5557
```
The Triton server will now be ready to handle requests, such as the following examples:

### Ensemble Model Request

```bash
curl -X POST localhost:5555/v2/models/ensemble/generate -d \
'{
  "text_input": "<s>[INST] write a story about boats [/INST]",
  "parameters": {
    "max_tokens": 500,
    "bad_words":[""],
    "stop_words":[""],
    "temperature": 0.4
  }
}'
```
### TensorRT LLM BLS Model Request

```bash
curl -X POST localhost:5555/v2/models/tensorrt_llm_bls/generate -d \
'{
  "text_input": "<s>[INST] write a story about boats [/INST]",
  "parameters": {
    "max_tokens": 500,
    "bad_words":[""],
    "stop_words":[""],
    "temperature": 0.4
  }
}'
```
## Important Notes

- **Ports:**  
  The ports specified can be modified by exposing them in the Dockerfile and adjusting them accordingly in the `tritonserver` command.

- **Dockerfile Integration:**  
  The `tritonserver` command can be integrated directly into the Dockerfile by configuring the `CMD ["/bin/bash"]` command. Alternatively, include the `tritonserver` command in an `entrypoint.sh` script and copy it into the container. This setup allows the entire Triton infrastructure to start with the `./start_triton.sh` command.

- **Path Configuration:**  
  Ensure that the paths specified in the `.sh` scripts correspond to your directories, especially those starting with `-v`.

## Model Deployment

TensorRT-LLM engines are closely tied to the [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend). Follow the official documentation to download the backend directory.

Within the `/tensorrtllm_backend/all_models/inflight_batcher_llm` directory, there are five models:

- **preprocessing:** Handles the preprocessing of the prompt.  
  `config.pbtxt` requires `tokenizer_dir` and `max_batch_size`.

- **postprocessing:** Converts the model output to text.  
  `config.pbtxt` requires `tokenizer_dir` and `max_batch_size`.

- **tensorrt_llm:** This is the LLM, represented as a TensorRT engine.  
  `config.pbtxt` requires multiple configurations.

The following two models manage HTTP communication, routing user requests through preprocessing, then to the tensorrt_llm, and finally through postprocessing back to the user:

- **ensemble**
- **tensorrt_llm_bls**

### Configuring Models

The `config.pbtxt` for each model can theoretically be populated using the following commands (refer to the [tensorrtllm_backend documentation](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/tools/fill_template.py)):

```bash
python3 tools/fill_template.py -i /trtback/all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:/models/mistral-nemo,tokenizer_type:llama,triton_max_batch_size:16,preprocessing_instance_count:1
python3 tools/fill_template.py -i /trtback/all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:/models/mistral-nemo,tokenizer_type:llama,triton_max_batch_size:16,postprocessing_instance_count:1
python3 tools/fill_template.py -i /trtback/all_models/inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:16,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 tools/fill_template.py -i /trtback/all_models/inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:16
python3 tools/fill_template.py -i /trtback/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt triton_max_batch_size:16,decoupled_mode:False,max_beam_width:1,engine_dir:/engines/mistral-nemo-engine_v01,max_tokens_in_paged_kv_cache:80000,max_attention_window_size:80000,kv_cache_free_gpu_mem_fraction:0.9,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600
```
However, as of the latest version (08.08.2024), these scripts had some bugs, so I manually added the parameters in each `config.pbtxt`. You can find the backend I used here: [tensorrtllm_backend](https://github.com/ChristosG/serve-mistral-nemo-as-trt/tree/main/tensorrtllm_backend).
