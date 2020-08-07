Following is the Pytorch based implementation to use pretrained Mobilenetv2 model and train it over four classes namely Flying Birds, Winged_drones, Large Quadcopters, Small Quadcopters. The images for these labels can be found out at
https://drive.google.com/file/d/133nsp1_PJXUpKOLzcYu9JivzHlGKMr8x/view?usp=sharing

We added following two fully connected model on top of existing pretrained model. 
# Mobilenet Model Addition to suit new class addition
![](Readme_images/Model_add.png)

Main colab file is kept at

New Generated model is kept at 

Training/Testing medhod is kept at

Image Augmentation and data loading is kept at

The model has been trained over 50 epochs and we are able to achive 85% as top test accuracy.
# Plots
![](Readme_images/Plots.png)
# LR vs Epochs
![](Readme_images/lr_vs_epoch.png)
# Misclassified Images
1. Flying Birds
![](Readme_images/flying_birds.png)
2. Large Quadcopters
![](Readme_images/large_Quadcopters.png)
3. Small Quadcopters
![](Readme_images/small_quadcopter.png)
4.Winged Drone
![](Readme_images/winged_drone.png)
# Cloud Watch Logs
![](Readme_images/CloudWatch.png)
# Api Gateway
![](Readme_images/ApiGateway.png)
