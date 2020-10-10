import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensor
from torch.utils.data.dataset import Dataset

class TestAugmentation():
  def __init__(self):
     self.upscale_factor = 4
     self.train_hr_transform = Compose([
        Resize(256, 256, interpolation=1, always_apply=False, p=1),
     ])

     self.train_lr_transform = Compose([
        Resize(64, 64, interpolation=1, always_apply=False, p=1),
    ])

     self.train_normalize = Compose([
        Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
      ),
      ToTensor(),
    ])

  def __call__(self, img):
     img = np.array(img) 
     hr_image = self.train_hr_transform(image = img)['image']
     lr_image = self.train_lr_transform(image = hr_image)['image']
     hr_restore_img = self.train_hr_transform(image = lr_image)['image']
     hrn = self.train_normalize(image = hr_image)['image']
     lrn = self.train_normalize(image = lr_image)['image']
     hrn_restore = self.train_normalize(image = hr_restore_img)['image']
     return (lrn), (hrn), (hrn_restore)