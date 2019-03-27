# Driver Attention Prediction Model

## Downloading Dataset:

The Berkeley DeepDrive Attention dataset can be downloaded here: https://bdd-data.berkeley.edu/. Click on the "Download Dataset" to get to the user portal and then you will find the BDD-Attention dataset listed together with other Berkeley DeepDrive video datasets.

## Project Introduction:

This project accompanies the paper **_Training a network to attend like human drivers saves it from common but misleading loss functions_** (https://arxiv.org/abs/1711.06406)

## Demo:
![Demo image](wiki_images/demo.jpg)

### Video demos 
<a href="http://www.youtube.com/watch?feature=player_embedded&v=IcAtJ0CvYuQ" target="_blank">
     <img src="http://img.youtube.com/vi/IcAtJ0CvYuQ/0.jpg" alt="Video demo cover" width="480" height="270" border="10" />
</a>

### Model structure
![Model structure image](wiki_images/structure2.png)

## Using Our Code:
### Dependencies
The code was written with Tensorflow 1.5, Keras 2.1.5 and some other common packages. A Docker image (blindgrandpa/tf150_kr215) was prepared for running the code. The Dockerfile of that Docker image is at `./docker_images/tf150_kr215/` in this repo. The Dockerfile lists all the dependencies. In order to use this Docker image to run our code, you need to have nvidia-docker installed.


### Use our model to do inference on your videos
If you want to use our model to generate predicted driver attention maps for your videos, please follow the steps below. 

1. Put your videos into the directory `./data/inference/camera_videos/`

2. Parse your videos into frames. 
Please run the following command to parse your videos into frames. `video_suffix` should be the suffix of your video files. `sample_rate` is for how many frames per second you want to have predicted attention maps. 3 Hz is recommended. We assume that there is no underscore in the names of your video files. 
```bash
python parse_videos.py \
--video_dir=data/inference/camera_videos \
--image_dir=data/inference/camera_images \
--sample_rate=3 \
--video_suffix=.mp4
```


3. Convert video frames to Tensorflow tfrecords files. All the video frames will be divided into `n_divides` tfrecords files.
The frames of each video will be divided into groups that have at most `longest_seq` frames. 
You can change `longest_seq` according to the memory size of your computer.
```bash
python write_tfrecords_for_inference.py \
--data_dir=data/inference \
--n_divides=2 \
--longest_seq=35
```


4. Download the pre-trained weights. Download [this zip file](https://drive.google.com/file/d/1q_CgyX73wrYTAsZjDF9aMXNPURcUmWVy/view?usp=sharing) and unzip it to `./`

5. Download the pre-trained weights of Alexnet. Downlaod [bvlc_alexnet.npy](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put it at `./`


6. Predict driver attention maps by runnning the following command. The predicted attention maps will be at `./pretrained_models/prediction_iter_0/`. The files will be named in the pattern "VideoName_TimeInMilliseconds.jpg".
```bash
python infer.py \
--data_dir=data \
--model_dir=pretrained_models
```


   
