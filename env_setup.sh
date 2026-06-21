#!/bin/bash
# Activate an existing conda env first, then source this file to expose pip
# nvidia CUDA libraries to TensorFlow 2.13.

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="${SITE}/nvidia/cudnn/lib:${SITE}/nvidia/cublas/lib:${SITE}/nvidia/cuda_runtime/lib:${SITE}/nvidia/cufft/lib:${SITE}/nvidia/curand/lib:${SITE}/nvidia/cusolver/lib:${SITE}/nvidia/cusparse/lib:${SITE}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export DDEBACKEND=tensorflow.compat.v1

