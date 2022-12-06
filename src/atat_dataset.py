import glob
import operator
import random
from functools import reduce

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


def get_face_id_finetune_dataset(data_set_type):
    train_img_path_list = []
    test_img_path_list = []
    train_labels = []
    test_labels = []

    for sub_idx in range(1, 41):
        img_path_list = glob.glob(f"data/atat_data/s{sub_idx}/*")
        random.shuffle(img_path_list)

        if data_set_type == "finetune":
            train_img_path_list.append(img_path_list[:8])
            train_labels += [sub_idx - 1]

            test_img_path_list.append(img_path_list[8:])
            test_labels += [sub_idx - 1]

        if data_set_type == "test":
            train_img_path_list += img_path_list[:8]
            train_labels += [sub_idx - 1] * 8

            test_img_path_list += img_path_list[8:]
            test_labels += [sub_idx - 1] * 2

    return train_img_path_list, test_img_path_list, train_labels, test_labels


class ATATFinetuneDataset(IterableDataset):
    def __init__(self, img_path_list, subj_ids, per_subj_imgcount, transform=None):
        self.img_path_list = img_path_list
        self.subj_ids = subj_ids
        self.transform = transform
        self.per_subj_imgcount = per_subj_imgcount

    def __iter__(self):
        batch_imgs = reduce(operator.add,
                            list(map(lambda x: random.sample(x, self.per_subj_imgcount), self.img_path_list)))
        batch_ids = (np.vstack([self.subj_ids] * self.per_subj_imgcount).T).ravel()

        for img_path, subj_id in zip(batch_imgs, batch_ids):
            img = Image.open(img_path) #.convert('L')
            if self.transform:
                img = self.transform(img)

            yield img, subj_id


class ATATTestDataset(Dataset):
    def __init__(self, img_path_list, subj_ids, transform=None):
        self.img_path_list = img_path_list
        self.subj_ids = subj_ids
        self.transform = transform

    def __len__(self):
        return len(self.subj_ids)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        subj_id = self.subj_ids[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, subj_id


def get_face_id_finetune_dataloaders(batchsize, transform, per_subj_imgcount=None):
    train_img_path_list, test_img_path_list, y_train, y_test = get_face_id_finetune_dataset("finetune")
    train_dataset = ATATFinetuneDataset(train_img_path_list, y_train, per_subj_imgcount, transform)
    test_dataset = ATATFinetuneDataset(test_img_path_list, y_test, per_subj_imgcount, transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, num_workers=0, pin_memory=True, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, num_workers=0, pin_memory=True, shuffle=False
    )

    return train_dataloader, test_dataloader


def get_face_id_test_dataloaders(batchsize, transform):
    train_img_path_list, test_img_path_list, y_train, y_test = get_face_id_finetune_dataset("test")
    train_dataset = ATATTestDataset(train_img_path_list, y_train, transform)
    test_dataset = ATATTestDataset(test_img_path_list, y_test, transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, num_workers=0, pin_memory=True, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, num_workers=0, pin_memory=True, shuffle=False
    )

    return train_dataloader, test_dataloader