# Building the TensorRT-LLM Engine

This guide details the process of building the TensorRT-LLM engine from the Mistral-Nemo model. Follow the steps below to successfully create and deploy the engine.

## Step 1: Start the Engine Builder Container

After successfully creating the container, run the `./start_trt_builder.sh` script to start the engine builder container.

## Step 2: Create a Model Checkpoint

To create a checkpoint of the model, run the following command. The parameters used here are a test configuration, but feel free to adjust them according to your requirements:

```bash
python3 examples/llama/convert_checkpoint.py \
    --model_dir /models/mistral-nemo \
    --output_dir /checkpoints/mistral-nemo-checkpoint_v01 \
    --dtype float16 \
    --per_token \
    --per_channel \
    --use_weight_only \
    --weight_only_precision int4
```

*Note:* This checkpoint uses `int4` quantization for faster inference, with a slight trade-off in accuracy. For more details on available parameters, run:

```bash
python3 examples/llama/convert_checkpoint.py --help
```

## Step 3: Convert the Checkpoint to a TensorRT-LLM Engine

After successfully creating the checkpoint, it's time to convert it to a TensorRT-LLM engine. The parameters specified below are another test configuration:

```bash
trtllm-build \
    --checkpoint_dir /checkpoints/mistral-nemo-checkpoint_v01 \
    --output_dir /engines/mistral-nemo-engine_v01 \
    --gemm_plugin float16 \
    --max_batch_size 16 \
    --context_fmha enable \
    --paged_kv_cache enable \
    --profiling_verbosity none \
    --use_fused_mlp \
    --remove_input_padding enable \
    --gpt_attention_plugin float16 \
    --logits_dtype float16 \
    --paged_state disable \
    --gather_all_token_logits \
    --max_num_tokens 4096 \
    --max_input_len 40000 \
    --max_seq_len 40000 \
    --enable_xqa enable \
    --workers 4 \
    --use_paged_context_fmha enable
```

For more information about the `trtllm-build` parameters, you can refer to the [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llama/README.md).

## Step 4: Test the Built TensorRT Engine

Once the TensorRT engine is successfully built, you can test it by running:

```bash
mpirun -n 1 --allow-run-as-root python3 examples/run.py \
    --engine_dir /engines/mistral-nemo-engine_v01 \
    --tokenizer_dir /models/mistral-nemo \
    --max_output_len 512 \
    --input_text "hi how are you?" \
    --temperature 0.2
```

In this example, `world_size=1` is used because the model fits within a single GPU. You may need to adjust this setting for larger models that cannot fit into a single GPU.

### MPI and GPU Parallelization

MPI is utilized to parallelize the workload across multiple GPUs. In my setup, using L40 GPUs which does not supprot NVLink, the performance decreases when GPUs have to communicate via PCIe while loading a model across multiple GPUs. However, if the model fits inside a single GPU, this configuration provides the best performance. If you have NVLink, it might offer better performance than PCIe for parallelized workloads, but I cannot confirm this from personal experience.

## Final Step: Deploy the Engine on Triton

If everything was successful, the engine will be built inside the `/engines` directory (`rank0.engine`, `config.json`). The engine is now ready to be deployed on Triton Inference Server for inference via HTTP.

