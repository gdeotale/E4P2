## EVA 4 Phase 2 Assignment 4 Face Recognition of 10 new classes trained with LFW dataset using transfer learning
------------------------------------------------------------------------------------------------------------

## Group : 
1. Abhijit Mali
2. Gunjan Deotale
3. Sanket Maheshwari
4. Pratik Jain

----------------------
## Dataset creation and training strategy :

Took images of the following 10 well-know celebrities not present in the LFW_funneled dataset.
1. Aamir Khan
2. Ajay Devgn
3. Akshay Kumar
4. Anushka Sharma
5. Deepika Padukone
6. MS Dhoni
7. Sonu Sood
8. Sushant Singh
9. Sharukh Khan
10. Virat Kohli

*  For creating train and test set we considered classes which has image count > 3 in the LFW funneled dataset.
*  So total there are 520 classes from the dataset and 10 custom classes which we added. There are total 620 classes in the dataset. We split the data 70:30 in train and test set.
*  We used cyclic learning rate policy with annealing. We achieved max train accuracy = 100% and test accuracy = 98.34%. We used pretrained model InceptionResnetV1 which was trained on the VGGFace dataset.

#### Train vs Test accuracy and loss plots

![](https://github.com/gdeotale/E4P2/blob/master/Assignment4/ReadmeImages/Accuracyplot.png)

#### LR vs epoch plot

![](https://github.com/gdeotale/E4P2/blob/master/Assignment4/ReadmeImages/lrvsepoch.png)

#### Misclassified Images

![](https://github.com/gdeotale/E4P2/blob/master/Assignment4/ReadmeImages/missclassified.jpg)

#### Correct classified Images

![](https://github.com/gdeotale/E4P2/blob/master/Assignment4/ReadmeImages/correct_classified.jpg)

---------------------------------------------------------------------------------------------------------------------------
## Following examples are implemented as AWS apps:
1. Resnet image classification
2. Mobilenet flying object classification
3. Face Alignment
4. Face Swap
5. Face Recognition

Link : https://session3--face-alignment-face-swap.s3.ap-south-1.amazonaws.com/doctype.html

----------------------------------------------------------------------------------------------------------------------------
## AWS API Link:

https://tf06hxpp42.execute-api.ap-south-1.amazonaws.com/dev/recognize

-----------------------------------------------------------------------------------------------------------------------------
## Code links
**Face Recognition**

https://github.com/gdeotale/E4P2/tree/master/Assignment4

**HTML and js**

https://github.com/gdeotale/E4P2/blob/master/Assignment4/Datahtml/doctype.html
