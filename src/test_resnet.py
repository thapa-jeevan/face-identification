import os

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms

from .atat_dataset import get_face_id_test_dataloaders
from src.models.model_resnet import resnet18
from src.settings import CHECKPOINT_DIR
from src.settings import IMG_SHAPE
from src.utils import seed_everything

seed_everything(98123)

test_transform = transforms.Compose([
    transforms.Resize(IMG_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def predict(model, dataloader_):
    model.eval()
    y_embd_ls = []
    y_subj_ls = []

    for i, (inp_tensor, y_true) in enumerate(dataloader_):
        y_embd = model(inp_tensor.cuda())
        y_embd = torch.nn.functional.normalize(y_embd)
        y_subj_ls.append(y_true.numpy())
        y_embd_ls.append(y_embd.detach().cpu().numpy())

    y_embd_ls = np.vstack(y_embd_ls)
    y_subj_ls = np.hstack(y_subj_ls)

    return y_embd_ls, y_subj_ls


if __name__ == '__main__':
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "face_identification.pt")
    model = resnet18(num_classes=256)
    model.cuda()
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    train_dataloader, test_dataloader = get_face_id_test_dataloaders(batchsize=128, transform=test_transform)
    train_embd_ls, train_subj_ls = predict(model, train_dataloader)
    test_embd_ls, test_subj_ls = predict(model, test_dataloader)

    knn_model = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="cosine")
    knn_model.fit(train_embd_ls, train_subj_ls)
    test_preds = knn_model.predict(test_embd_ls)

    acc_ = sum(test_preds == test_subj_ls) / len(test_subj_ls)
    print(f"Test Accuracy: {acc_}")
