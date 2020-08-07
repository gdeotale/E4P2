## EVA 4 Phase 2 Assignment 2 Deploy Trained Mobilenet_v2 on AWS
------------------------------------------------------------------------------------------------------------

## Group : 
1. Abhijit Mali
2. Gunjan Deotale
3. Sanket Maheshwari
4. Pratik Jain

----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------

# what model did you train?
Following is the Pytorch based implementation to use pretrained Mobilenetv2 model and train it over four classes namely Flying Birds, Winged_drones, Large Quadcopters, Small Quadcopters. The images for these labels can be found out at
https://drive.google.com/file/d/133nsp1_PJXUpKOLzcYu9JivzHlGKMr8x/view?usp=sharing

The model is deployed on aws and working is tested on Insomnia using url
https://9nnncm80a9.execute-api.ap-south-1.amazonaws.com/dev/classify

# Resizing Strategy:
Although the images were of different sizes, we have used image resizing in albumentation to resize to 256x256 and then we used image cropping in albumentation to crop it to 224x224 as this size is need as input to Mobilenet v2.

# Explain the code?
We added following two fully connected model on top of existing pretrained model. For training we have freezed all existing layers until average pool and unfreezed newly added last 2 layers.
# Mobilenet Model Addition to suit new class addition
![](Readme_images/Model_add.png)

Main colab file is kept at
https://github.com/gdeotale/E4P2/blob/master/Assignment2/Mobilenet_Training/Main.ipynb

New Generated model is kept at 
https://github.com/gdeotale/E4P2/blob/master/Assignment2/Mobilenet_Training/Generated_models/Modeljit.pt

Training/Testing method is kept at
https://github.com/gdeotale/E4P2/blob/master/Assignment2/Mobilenet_Training/Train_Test_utils/

We have used Albumentation as method of augmentation, we tried image resizing, Image cropping, Cut Out and Image Normalization as methods in Augmentation
Image Augmentation and Dataloader is kept at
https://github.com/gdeotale/E4P2/blob/master/Assignment2/Mobilenet_Training/Main.ipynb

We have segregated the data in train test folder in ratio of 70:30 classwise.

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
