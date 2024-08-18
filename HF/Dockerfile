FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
  build-essential \
  python3.9 \
  python3-pip \
  git \
  ffmpeg \
  libcudnn8 \
  libcudnn8-dev \
  sudo

# Set up a new user named "user" with user ID 1099
RUN useradd -m -u 1099 user

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# Set up CUDA libraries
RUN export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`:$LD_LIBRARY_PATH

RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Create necessary directories and set permissions
RUN mkdir -p /app/Results/Audio_Processing /app/Results/Video_Downloads && \
    chown -R user:user /app

# Switch to the "user" user
USER user

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces

# Command to run the application
CMD ["python3", "app.py"]
