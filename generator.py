import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # Load labels from JSON file
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)
        self.n_samples = len(self.labels)
        self.epoch = 0
        self.batch_num = 0

        # Adjust batch size if it is greater than samples size or if it is zero
        if self.n_samples < self.batch_size or self.batch_size == 0:
            self.batch_size = self.n_samples

        # Create a mapping array for shuffling
        self.mapping = np.arange(self.n_samples)

        # Dictionary to map class indices to class names
        self.class_dict = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method
        if self.batch_num * self.batch_size >= self.n_samples:
            self.epoch += 1
            self.batch_num = 0

        # Shuffle mapping array
        if self.batch_num == 0 and self.shuffle:
            np.random.shuffle(self.mapping)

        # Empty arrays for images and labels
        images = np.zeros((self.batch_size, *tuple(self.image_size)))
        labels = np.zeros(self.batch_size, dtype=int)

        start = self.batch_num * self.batch_size
        end = start + self.batch_size

        # Cases where data is not divisible by batch size (reusing images from the beginning of dataset)
        if end <= self.n_samples:
            indices = self.mapping[start:end]
        else:
            indices = np.concatenate((self.mapping[start:], self.mapping[:end-self.n_samples]))

        # Load and augment images
        for i, index in enumerate(indices):
            img = np.load(f"{self.file_path}/{index}.npy")
            images[i] = self.augment(img)
            labels[i] = self.labels[str(index)]

        self.batch_num += 1

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        if img.shape != self.image_size:
            img = np.resize(img, self.image_size)

        if self.mirroring:
            if np.random.choice((True, False)):
                img = np.fliplr(img)

        if self.rotation:
            n_times = np.random.choice((0, 1, 2, 3))
            img = np.rot90(img, n_times)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        imgs, labs = self.next()
        fig = plt.figure(figsize=(10, 10))
        cols = 3
        rows = self.batch_size // 3 + (1 if self.batch_size % 3 != 0 else 0)
        for i in range(1, self.batch_size + 1):
            img = imgs[i - 1]
            lab = self.class_dict[labs[i - 1]]
            fig.add_subplot(rows, cols, i)
            plt.imshow(img.astype('uint8'))
            plt.xticks([])
            plt.yticks([])
            plt.title(lab)
        plt.show()
