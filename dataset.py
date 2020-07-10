import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms


class MURA(Dataset):
    def __init__(self, directory, csv_files, csv_labels, transform):
        self.directory = directory
        self.files_df = pd.read_csv(os.path.join(directory, csv_files),
                                    header=None)
        self.labeled_df = pd.read_csv(os.path.join(directory, csv_labels),
                                      header=None)
        self.transform = transform

    def __len__(self):
        return len(self.files_df)

    def __getitem__(self, idx):
        filename = self.files_df.iloc[idx, 0]
        path = os.path.join(self.directory, filename)
        img = Image.open(path).convert('RGB')
        img = self.transform(img) if self.transform else img
        label = self.labeled_df[self.labeled_df[0] == os.path.dirname(path) +
                                "/"][1].iloc[0]
        return img, label


def get_dataloader(csv_files,
                   csv_labels,
                   name,
                   batch_size,
                   shuffle,
                   split=None,
                   directory="",
                   num_workers=4):
    data_transforms = {
        "train":
        transforms.Compose((
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        )),
        "valid":
        transforms.Compose((
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        )),
    }
    sampler = None
    dataset = MURA(directory, csv_files, csv_labels, data_transforms[name])
    if split is not None:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        new_dataset_size = int(np.floor(split * dataset_size))
        np.random.shuffle(indices)
        splited = indices[:new_dataset_size]
        sampler = SubsetRandomSampler(splited)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      sampler=sampler)
