import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


IMG_DIR = "data/CelebA/img_align_celeba/"
IDENTITY_FILE = "data/CelebA/identity_CelebA.txt"


def get_sub_ids(per_subj_imgcount):
    df_id = pd.read_csv(IDENTITY_FILE, delimiter=" ", names=["img_name", "subject_id"])
    df_id = df_id.sort_values(by="subject_id")

    df_id_sub_count = df_id["subject_id"].value_counts()
    t = df_id_sub_count[(df_id_sub_count <= per_subj_imgcount)].reset_index().rename(
        columns={"subject_id": "count", "index": "subject_id"})
    df_id = df_id.merge(t, how="left")
    df_id = df_id[df_id["count"].isna()].iloc[:, :2]
    df_id.index = range(len(df_id))
    subjects = np.unique(df_id["subject_id"].values)

    df_id["img_name"] = df_id["img_name"].apply(lambda x: [x])
    df_sub_imgs = df_id.groupby("subject_id").sum()

    return df_sub_imgs, subjects


class CelebaDataset(IterableDataset):
    def __init__(self, img_dir, df_subj_imgs, subjects, batch_subj_count, per_subj_imgcount, transform=None):
        self.img_dir = img_dir

        self.df_subj_imgs = df_subj_imgs
        self.subjects = subjects
        self.batch_subj_count = batch_subj_count
        self.per_subj_imgcount = per_subj_imgcount
        self.transform = transform

    def __iter__(self):
        batch_subjects = np.random.choice(self.subjects, self.batch_subj_count, replace=False)

        batch_imgs = self.df_subj_imgs.loc[batch_subjects]["img_name"].apply(
            lambda x: random.sample(x, self.per_subj_imgcount)).sum()

        batch_ids = (np.vstack([batch_subjects] * self.per_subj_imgcount).T).ravel()

        for img_name, sub_id in zip(batch_imgs, batch_ids):
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path) #.convert('L')
            if self.transform:
                img = self.transform(img)

            yield img, sub_id


def get_dataloader(batch_size, train_transform, per_subj_imgcount):
    batch_subj_count = batch_size // per_subj_imgcount
    df_subj_imgs, subjects = get_sub_ids(per_subj_imgcount)

    train_dataset = CelebaDataset(IMG_DIR, df_subj_imgs, subjects, batch_subj_count, per_subj_imgcount, train_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=False
    )

    return train_dataloader
