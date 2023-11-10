import numpy as np
import torch
from matplotlib import pyplot as plt
import random


class DataManager:
    def __init__(self, filename):
        self.data = np.load(filename)
        self.images = self.data['images']
        self.poses = self.data['poses']
        self.focal = self.data['focal']

        self.num_of_images = self.images.shape[0]
        self.image_height = self.images.shape[1]
        self.image_width = self.images.shape[2]

        self.print_data_info()
        self.print_example_image_data()

    def print_data_info(self):
        print(f'Images shape: {self.images.shape}')
        print(f'Poses shape: {self.poses.shape}')
        print(f'Focal length: {self.focal}')

    def print_example_image_data(self):
        rand_ind = random.randint(0, self.num_of_images)
        example_img = self.images[rand_ind]
        example_pose = self.poses[rand_ind]
        plt.imshow(example_img)
        print('Pose')
        print(example_pose)
