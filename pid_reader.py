import image_pb2
import traceback
import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from skimage import io
import image_pb2
import cv2

class PIDReader(Dataset):
    #image_count = amount of images per epoch per phase (train/valid)
    def __init__(self, dataset_path, data_transforms, csv_path, image_count, format, dataset_depth):
        self.dataset_path = dataset_path
        self.data_transforms = data_transforms
        self.image_count = image_count
        print("csv_path: ", csv_path)
        self.csv_df = pd.read_csv(csv_path, header=0, names=['id', 'name', 'class'],
                              dtype={'id': str, 'name': str, 'class': str})
        self.chosen_images, self.num_classes = self.select_images(self.csv_df, self.image_count)
        self.format = format
        self.dataset_depth = dataset_depth


    def get_image(self, klasse, image):
        for class_file in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_file)
            if os.path.isfile(class_path) and class_file == klasse + ".pid":
                f = open(class_path, "rb")
                p = image_pb2.Person()
                p.ParseFromString(f.read())
                f.close()
                for i in p.images:
                    if i.name == image:
                        return i
        raise Exception("pid file not found. Variables: image: ", image, ", class: ", klasse)

    # selects images from the csv at random, where the chance of each class being chosen is equal, no matter how many images a class has.
    def select_images(self, csv_df, image_count):
        # dictionary to keep track of which images belong to which class:
        def make_dictionary_for_class(df):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            image_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in image_classes:
                    image_classes[label] = []
                image_classes[label].append(df.iloc[idx, 0])
            return image_classes
        selected_images = []
        classes = csv_df['class'].unique()
        image_classes = make_dictionary_for_class(csv_df)

        for _ in range(image_count):
            klasse = np.random.choice(classes)
            name = csv_df.loc[csv_df['class'] == klasse, 'name'].values[0]
            img = np.random.randint(0, len(image_classes[klasse]))
            selected_images.append([image_classes[klasse][img], klasse, name])
        return selected_images, len(classes)

    def __len__(self):
        
        return len(self.chosen_images)
        
    def __getitem__(self, idx):
        try:
            img_id, img_class, img_name = self.chosen_images[idx]
            # img = os.path.join(self.dataset_dir, str(img_name), str(img_id) + self.format)
            img_id = str(img_id) + self.format
            img = self.get_image(img_name, img_id).contents
            img = io.imread(img, plugin='imageio')
            if self.dataset_depth==1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_class = torch.from_numpy(np.array(img_class).astype('long'))
            sample = {"image": img, "class": img_class}
            if self.data_transforms:
                sample['image'] = self.data_transforms(sample['image'])
        except:
            print("traceback: ", traceback.format_exc())
            sample = {'exception': True}
        # return tuple(tensor[index] for tensor in self.tensors)
        return sample


