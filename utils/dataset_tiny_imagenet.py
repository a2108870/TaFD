import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        TinyImageNet Dataset.

        Args:
            root (string): Root directory of dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from validation set. Default: True
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version.
        """
        self.root = root
        self.train = train
        self.transform = transform

        # Load class info
        wnids_path = os.path.join(root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.wnids = [line.strip() for line in f.readlines()]

        # Create class to index mapping
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        # Load default transforms if none provided
        if self.transform is None:
            if self.train:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor()
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std=[0.229, 0.224, 0.225])
                ])

        # Load image paths and labels
        self.images = []
        self.labels = []

        if self.train:
            # Load training data
            for class_id in self.wnids:
                class_dir = os.path.join(root, 'train', class_id, 'images')
                class_idx = self.class_to_idx[class_id]

                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
        else:
            # Load validation data
            val_annotations_path = os.path.join(root, 'val', 'val_annotations.txt')
            val_img_dir = os.path.join(root, 'val', 'images')

            with open(val_annotations_path, 'r') as f:
                for line in f:
                    img_name, class_id = line.split()[:2]
                    if class_id in self.class_to_idx:  # Check if class is in training set
                        img_path = os.path.join(val_img_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_id])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target) where target is class_index of target class.
        """
        img_path = self.images[idx]
        target = self.labels[idx]

        # Load image
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random valid index instead
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_class_name(self, idx):
        """
        Get class name from class index.

        Args:
            idx (int): Class index
        Returns:
            str: Class name (WordNet ID)
        """
        return self.wnids[idx]
