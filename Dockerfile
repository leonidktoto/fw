# Базовый образ с поддержкой CUDA 12 и cuDNN для Ubuntu 20.04
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=nointeractive
# Устанавливаем необходимые системные зависимости, Python 3.11 и ffmpeg
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip для Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Устанавливаем переменную окружения для корректной загрузки библиотек CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# Set the working directory inside the container
WORKDIR /app

# Install the dependencies
RUN pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
RUN pip install --force-reinstall ctranslate2
RUN ln -s /usr/bin/python3.11 /usr/bin/python 
RUN apt-get update && apt-get install -y libcudnn8 libcudnn8-dev

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
# Copy the rest of the application code to the container
COPY . .

# Set the command to run your script
CMD ["python3.11", "main.py"]
