import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as trns
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, root, seq, img_size):
        self.root = root
        self.seq = seq
        self.data_trns = trns.Compose([
            trns.ToTensor(),
            trns.Resize([img_size, img_size]),
            trns.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ])

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        filename = self.seq[idx]
        img = Image.open(self.root+filename+".jpg")
        img = self.data_trns(img)
        return img, filename
