import glob
import os
import random

import cv2
import numpy as np

from src.settings import IMG_HEIGHT, IMG_WIDTH


def divide_face_imgs_():
    train_img_path_list = []

    for sub_idx in range(1, 41):
        img_path_list = glob.glob(f"data/atat_data/s{sub_idx}/*")
        random.shuffle(img_path_list)
        train_img_path_list += img_path_list[:8]

    test_img_path_list = list(set(glob.glob("data/atat_data/*/*")) - set(train_img_path_list))
    return train_img_path_list, test_img_path_list


def extract_subject(img_path):
    return int(img_path.split(os.path.sep)[-2][1:]) - 1


def read_imgs(img_path_list):
    img_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_list.append(img.ravel())
    return np.vstack(img_list)


def get_face_id_data():
    train_img_path_list, test_img_path_list = divide_face_imgs_()
    X_train = read_imgs(train_img_path_list)
    X_test = read_imgs(test_img_path_list)

    y_train = np.array([int(extract_subject(t)) for t in train_img_path_list])
    y_test = np.array([int(extract_subject(t)) for t in test_img_path_list])

    return X_train, X_test, y_train, y_test
