import os

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from src.face_identification.atat_dataset import get_face_id_finetune_dataloaders
from src.face_identification.celeba_dataset import get_dataloader as get_pretrain_dataloader
from src.face_identification.loss import contrastive_loss_entropy
from src.models.model_resnet import resnet18
from src.settings import CHECKPOINT_DIR
from src.settings import IMG_SHAPE
from src.utils import seed_everything

seed_everything(98123)

PRETRAIN_BATCH_SIZE = 1024 * 3
FINETUNE_BATCH_SIZE = 80
PER_SUB_IMGCOUNT = 2

TEMPERATURE = 0.01

PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 3
ITER_PER_EPOCH = 100


test_transform = transforms.Compose([
    transforms.Resize(IMG_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_transform = transforms.Compose([
    transforms.Resize(IMG_SHAPE),
    transforms.RandomResizedCrop(IMG_SHAPE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-15, 15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def train_single_epoch(model, train_dataloader):
    train_loss = 0
    model.train()

    for idx in range(ITER_PER_EPOCH):
        inp_tensor, ids = next(iter(train_dataloader))
        optimizer.zero_grad()

        y_embd = model(inp_tensor.cuda())

        loss = criterion((y_embd, ids))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred = np.argmax(y_embd.detach().cpu().numpy(), axis=1)
        print(f"\rEpoch: {epoch} Iteration: {idx + 1}/{ITER_PER_EPOCH} Training Loss: {loss.item():.5f}", end=" ")
    tqdm.write(f"Training Loss: {train_loss / ITER_PER_EPOCH:2.3f}")


if __name__ == '__main__':
    model = resnet18(num_classes=256)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = lambda x: contrastive_loss_entropy(*x, TEMPERATURE)

    pre_checkpoint_path = os.path.join(CHECKPOINT_DIR, "face_identification_pretrain.pt")
    pre_train_dataloader = get_pretrain_dataloader(PRETRAIN_BATCH_SIZE, train_transform, PER_SUB_IMGCOUNT)

    for epoch in tqdm(range(PRETRAIN_EPOCHS)):
        train_single_epoch(model, pre_train_dataloader)

        torch.save({
            'epoch': epoch,
            'stage': "pretrain",
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, pre_checkpoint_path)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "face_identification.pt")
    finetune_dataloader, test_dataloader = get_face_id_finetune_dataloaders(FINETUNE_BATCH_SIZE, train_transform, PER_SUB_IMGCOUNT)

    for epoch in tqdm(range(FINETUNE_EPOCHS)):
        train_single_epoch(model, finetune_dataloader)

        torch.save({
            'epoch': epoch,
            'stage': "finetune",
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)


