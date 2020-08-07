import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.notebook import tqdm

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_acc_loss(train_acc, test_acc, trainloss_, testloss_):
  fig, axs = plt.subplots(2,2,figsize=(10,10))
  axs[0,0].plot(train_acc)
  axs[0,0].set_title("Training Accuracy")
  axs[0,0].set_xlabel("Batch")
  axs[0,0].set_ylabel("Accuracy")
  axs[0,1].plot(test_acc) 
  axs[0,1].set_title("Test Accuracy")
  axs[0,1].set_xlabel("Batch")
  axs[0,1].set_ylabel("Accuracy")
  axs[1,0].plot(trainloss_)
  axs[1,0].set_title("Training Loss")
  axs[1,0].set_xlabel("Batch")
  axs[1,0].set_ylabel("Loss")
  axs[1,1].plot(testloss_) 
  axs[1,1].set_title("Test Loss")
  axs[1,1].set_xlabel("Batch")
  axs[1,1].set_ylabel("Loss")