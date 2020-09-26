import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2
from tqdm.notebook import tqdm

# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def getmeanstd(path):
  count = 0
  sum_mean = [0, 0, 0]; sum_std = [0, 0, 0]
  for i in tqdm(glob.glob(path)):
    img = cv2.imread(i)
    if img is None:
      print(i)
      continue
    r = img[:,:,2]/255
    g = img[:,:,1]/255
    b = img[:,:,0]/255
    sum_mean[0] += np.mean(r)
    sum_std[0] += np.std(r)
    sum_mean[1] += np.mean(g)
    sum_std[1] += np.std(g)
    sum_mean[2] += np.mean(b)
    sum_std[2] += np.std(b)
    count=count+1
    if(count>100000):
      break
    
  sum_mean[0] = sum_mean[0]/(count)
  sum_std[0] = sum_std[0]/count
  sum_mean[1] = sum_mean[1]/count
  sum_std[1] = sum_std[1]/count
  sum_mean[2] = sum_mean[2]/count
  sum_std[2] = sum_std[2]/count
  print("Mean: -",sum_mean)
  print("stdDev: -", sum_std)
  return sum_mean, sum_std


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