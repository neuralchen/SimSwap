FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && \
    apt-get install -y wget && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    apt-get update && \
    apt-get install -y git && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

WORKDIR /
COPY ./requirements.txt /SimSwap/requirements.txt
RUN pip install -r /SimSwap/requirements.txt


# Use external volume for data
ENV NVIDIA_VISIBLE_DEVICES 1
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''","--NotebookApp.password=''", "--allow-root"]
