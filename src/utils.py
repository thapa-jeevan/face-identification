import os
import random

import numpy as np
import torch


def calculate_accuracy(model, X, y):
    y_preds = model.predict(X)
    return sum(y_preds.ravel() == y.ravel()) / len(X)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
