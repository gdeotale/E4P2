from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():
  def __init__(self):
    self.train_transform = Compose([
      Resize(160, 160, interpolation=1, always_apply=False, p=1),
      HorizontalFlip(),
      Cutout(num_holes=4, max_h_size=8, max_w_size=8, always_apply=False, p=1.0),
      
      Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
      ),
      ToTensor()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img