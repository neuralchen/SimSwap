ARG CUDA_VERSION=11.1

# onnxruntime-gpu requires cudnn
ARG CUDNN_VER=8

# See possible types: https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated
ARG IMAGE_TYPE=runtime

FROM nvidia/cuda:${CUDA_VERSION}-${IMAGE_TYPE}-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y wget \
    && \
    apt autoremove -y && \
    apt clean -y

# Install Miniconda See possible versions: https://repo.anaconda.com/miniconda/
ARG CONDA_VERSION=3-py38_4.9.2

ARG CONDA_DIR=/opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget -q -O ./miniconda.sh http://repo.continuum.io/miniconda/Miniconda${CONDA_VERSION}-Linux-x86_64.sh \
    && sh ./miniconda.sh -bfp $CONDA_DIR \
    && rm ./miniconda.sh

WORKDIR /home

COPY ./environment.yml ./environment.yml 

RUN conda env update -n base --prune --file ./environment.yml && \
    conda clean -ay && rm ./environment.yml

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VER}-${IMAGE_TYPE}-ubuntu20.04

ARG CONDA_DIR=/opt/conda

# Copy conda installation
COPY --from=builder /opt/conda $CONDA_DIR

ENV PATH=$CONDA_DIR/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y software-properties-common && \
    # the latest version of Git
    add-apt-repository ppa:git-core/ppa && \
    apt update && \
    apt install -y --no-install-recommends \
    git ca-certificates wget unzip \
    libgl1-mesa-glx \
    && \
    apt autoremove -y && \
    apt clean -y

WORKDIR /home

RUN git clone --single-branch https://github.com/neuralchen/SimSwap.git && \
    cd ./SimSwap && \
    rm -rf .git && \
    ./download-weights.sh && \
    python test_video_swapsingle.py \
        --isTrain false --use_mask --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --pic_a_path ./demo_file/Iron_man.jpg \
        --video_path ./demo_file/multi_people_1080p.mp4 \
        --output_path ./output/multi_test_swapsingle.mp4 \
        --temp_path ./temp_results

WORKDIR /home/SimSwap

