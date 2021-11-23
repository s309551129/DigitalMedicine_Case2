import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as trns
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, root, info, img_size, mode):
        self.root = root
        self.info = info
        self.mode = mode
        if self.mode == "train":
            self.data_trns = trns.Compose([
                trns.ToTensor(),
                trns.RandomResizedCrop(img_size),
                trns.RandomHorizontalFlip(p=0.5),
                trns.RandomRotation(30),
                #trns.ColorJitter(brightness=0.126, saturation=0.5),
                #trns.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                trns.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
            ])
        else:
            self.data_trns = trns.Compose([
                trns.ToTensor(),
                trns.Resize([img_size, img_size]),
                #trns.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                trns.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
            ])

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        filename, label_idx = self.info[idx]
        img = Image.open(self.root+filename+".jpg")
        img = self.data_trns(img)
        return img, label_idx
