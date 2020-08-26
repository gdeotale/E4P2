import cv2
from tqdm import tqdm
import os
from PIL.Image import Image


class DatasetHelper:

    def __init__(self):
        self.output_folder = ''
        self.dictionaryFileName = ''

    def get_train_test_data(self, images_folder, train_split=0.7):

        bg_ext = '.jpg'

        train_imgs = []
        test_imgs = []

        train_labels = []
        test_labels = []
        path_data = os.walk(images_folder)

        for path in path_data:
            for directory in path[1]:
                print(directory)
                from src.utils import Utils
                file_paths, filenames = Utils.get_all_file_paths(os.path.join(images_folder, directory))
                total_count = len(filenames)
                count = 0

                train_count = total_count * train_split

                for file_path in file_paths:
                    if count < train_count:
                        train_imgs.append(file_path)
                        train_labels.append(directory)
                    else:
                        test_imgs.append(file_path)
                        test_labels.append(directory)
                    count += 1

        return train_imgs, train_labels, test_imgs, test_labels

    def resize(self, img):
        scale_percent = 29  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized
