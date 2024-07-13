
# Tennis System

## Introduction
This project analyzes Tennis players in a video to measure their speed, ball shot speed and number of shots. This project will detect players and the tennis ball using YOLO and also utilizes CNNs to extract court keypoints. For more details, please refer to [here](https://roubin.me/computer-vision-based-tennis-match-analysis-system/). 

## Output Videos
Here is a screenshot from one of the output videos:

![Screenshot](result/tennis_final.png)

## Models Used
* YOLO v8x for player detection
* Fine Tuned YOLOv5l6u for tennis ball detection
* Court Key point extraction

* Trained YOLOV5 model: https://drive.google.com/file/d/1-qyQgPXbwp9TSjqczIvpoFntWuPruUvS/view?usp=sharing
* Trained tennis court key point model: https://drive.google.com/file/d/1lboeE2gtjOMISs6qub-krFzfNapkB-qR/view?usp=sharing

## Training
* Tennis ball detetcor with YOLO: train/yolov5_tennis.ipynb
* Tennis court keypoint with Pytorch: train/tennis_court_keypoints.ipynb

## Requirements
* python3.8
* ultralytics
* pytroch
* pandas
* numpy 
* opencv

