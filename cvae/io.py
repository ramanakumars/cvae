import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import ImageReadMode, read_image


class ZooniverseLabelGenerator(Dataset):
    AUGMENTATION = T.Resize(96)

    def __init__(self, imgfolder, label_file, indices=None):
        super().__init__()
        self.imgfolder = imgfolder
        self.labels = np.load(label_file)

        self.imgs = np.asarray(sorted(glob.glob(self.imgfolder + "*.png")))
        self.subject_ids = np.asarray(
            [int(os.path.splitext(os.path.basename(e))[0]) for e in self.imgs]
        )

        assert len(self.labels) == len(self.imgs), (
            f"List of imgs ({len(self.imgs)}) and labels ({len(self.labels)})should match!"
        )

        if indices is not None:
            self.indices = indices
            self.ndata = len(indices)
        else:
            self.ndata = len(self.imgs)
            self.indices = np.arange(self.ndata)

        print(f"Found data with {self.ndata} images")

    def __len__(self):
        return self.ndata

    def __getitem__(self, index):
        subject_id = self.subject_ids[index]
        img_fname = self.imgs[index]

        img = self.AUGMENTATION(read_image(img_fname, ImageReadMode.RGB) / 255.0)
        label = torch.Tensor(
            self.labels[self.labels[:, 0] == subject_id, 1:].tolist()[0]
        )

        return img, label


class LabelMeDataset(Dataset):
    AUGMENTATION = T.Resize(96)

    def __init__(self, folder_path: str):
        super().__init__()
        self.imgfolder = folder_path
        self.labels = np.loadtxt(os.path.join(folder_path, "annotation.txt"))

        self.imgs = np.asarray(
            sorted(glob.glob(os.path.join(self.imgfolder, "*/*.jpg")))
        )
        self.img_ids = np.asarray(
            [int(os.path.splitext(os.path.basename(e))[0]) for e in self.imgs]
        )

        assert len(self.labels) == len(self.imgs), (
            f"List of imgs ({len(self.imgs)}) and labels ({len(self.labels)})should match!"
        )

        self.ndata = len(self.img_ids)

        print(f"Found data with {self.ndata} images")

    def __len__(self):
        return self.ndata

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_fname = self.imgs[index]

        img = self.AUGMENTATION(read_image(img_fname, ImageReadMode.RGB) / 255.0)
        # read the labels and normalize between 0 to 1
        label = (
            torch.Tensor(self.labels[self.labels[:, 0] == img_id, 1:].tolist()[0]) + 1
        ) / 2.0

        return img, label
