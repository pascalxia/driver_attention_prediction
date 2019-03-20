FROM tensorflow/tensorflow:1.5.0-gpu-py3

RUN apt-get update && apt-get install -y \
    ffmpeg
    
RUN pip3 install \
    feather-format \
    keras==2.1.5 \
    moviepy \
    opencv-python==3.2.0.8 \
    pandas \
    tqdm