import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from lfw_dataset import get_dataloader
from resnet18 import resnet18
from settings import CHECKPOINT_DIR

BATCH_SIZE = 32


def predict(model, dataloader_):
    model.eval()
    y_embd_ls = []

    for i, inp_tensor in enumerate(dataloader_):
        y_embd = model(inp_tensor.cuda())
        y_embd = torch.nn.functional.normalize(y_embd)
        y_embd_ls.append(y_embd.detach().cpu().numpy())

    return np.vstack(y_embd_ls)


if __name__ == '__main__':
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "face_classification.pt")
    model = resnet18(num_classes=2)
    model.cuda()
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    dataloader, lfw_labels = get_dataloader(BATCH_SIZE)
    y_embd_ls = predict(model, dataloader)

    t = y_embd_ls @ y_embd_ls.T

    preds = (np.diag(t, 1)[::2] > 0.84) * 1

    acc_ = accuracy_score(preds, lfw_labels)
    print(f"Face Verification Accuracy in LFW Dataset:{acc_:.3f}")
