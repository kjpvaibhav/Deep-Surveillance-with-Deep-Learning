# Deep-Surveillance-with-Deep-Learning

Surveillance security is a very tedious and time-consuming job. In this tutorial, we will build a system to automate the task of analyzing video surveillance. We will analyze the video feed in real-time and identify any abnormal activities like violence or theft.

There is a lot of research going on in the industry about video surveillance among them; the role of CCTV videos has overgrown. CCTV cameras are placed all over the places for surveillance and security.

In the last decade, there have been advancements in deep learning algorithms for deep surveillance. These advancements have shown an essential trend in deep surveillance and promise a drastic efficiency gain. The typical applications of deep surveillance are theft identification, violence detection, and detection of the chances of explosion.

## Network architecture:
We have generally seen deep neural networks for computer vision, image classification, and object detection tasks. In this project, we have to extend deep neural networks to 3-dimensional for learning spatio-temporal features of the video feed.

For this video surveillance project, we will introduce a spatio temporal autoencoder, which is based on a 3D convolution network. The encoder part extracts the spatial and temporal information, and then the decoder reconstructs the frames. The abnormal events are identified by computing the reconstruction loss using Euclidean distance between original and reconstructed batch.

## The dataset for abnormal event detection in video surveillance:
Following are the comprehensive datasets that are used to train models for anomaly detection tasks.

### CUHK Avenue Dataset:
This dataset contains 16 training and 21 testing video clips. The video contains 30652 frames in total.

The training videos contain video with normal situations. The testing videos contain videos with both standard and abnormal events.

Dataset Download Link: Avenue Dataset for Abnormal Event Detection - http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

### UCSD pedestrian Dataset:
This dataset contains videos with pedestrians. It includes groups of people walking towards, away, and parallel to the camera. The abnormal event includes:

Non-pedestrian entities
Anomalous pedestrian motion patterns
Dataset Download Link: UCSD Anomaly Detection Dataset - http://www.svcl.ucsd.edu/projects/anomaly/dataset.html

## Project Source Code
Before proceeding ahead, please download the source code which we used in this deep learning project: Video Surveillance Project Code - https://data-flair.s3.ap-south-1.amazonaws.com/Data-Science-Code/video-surviellance-project-code.zip

## Video Surveillance – Anomaly Even Detection Code:
First, download any one of the above datasets and put in a directory named “train”.

Make a new python file train.py and paste the code described in following steps:

1. Imports:
2. Initialize directory path variable and describe a function to process and store video frames.
3. Extract frames from video and call store function.
4. Store the store_image list in a numpy file “training.npy”.
5. Create spatial autoencoder architecture.
6. Train the autoencoder on the “training.npy” file and save the model with name “saved_model.h5”.
   Run this script to train and save the autoencoder model.

Now make another python file “test.py” and observe the results of abnormal event detection on any custom video.
Now, run this script and observe the results of video surveillance, it will highlight the abnormal events.

## Summary:
In this deep learning project, we train an autoencoder for abnormal event detection. We train the autoencoder on normal videos. We identify the abnormal events based on the euclidean distance of the custom video feed and the frames predicted by the autoencoder.
We set a threshold value for abnormal events. In this project, it is 0.0068; you can vary this threshold to experiment getting better results.

