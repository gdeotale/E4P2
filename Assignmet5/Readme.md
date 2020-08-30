#Data

1.1 Pose Estimation on COCO
The COCO train, validation, and test sets contain more than 200k images and 250k person instances labeled with keypoints. 
150k instances of them are publicly available for training and validation. Our models are only trained on all COCO train2017 dataset 
(includes 57K images and 150K person instances) no extra data involved, ablation are studied on the val2017 set and finally we report 
the final results on test-dev2017 set to make a fair comparison with the public state-of-the-art results

1.2 Pose Estimation and Tracking on PoseTrack
PoseTrack dataset is a large-scale benchmark for multi-person pose estimation and tracking in videos. It requires not only pose estimation 
in single frames, but also temporal tracking across frames. It contains 514 videos including 66,374 frames in total, split into 300, 50 and 
208 videos for training, validation and test set respectively. For training videos, 30 frames from the center are annotated. For
validation and test videos, besides 30 frames from the center, every fourth frame is also annotated for evaluating long range articulated
tracking.
