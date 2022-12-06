import argparse
import os

import torch
from torchvision import transforms
from tqdm import tqdm

from src.celeba_dataset import get_dataloader as get_pretrain_dataloader
from src.loss import contrastive_loss_entropy
from src.resnet18 import resnet18
from src.settings import CHECKPOINT_DIR
from src.settings import IMG_SHAPE
from src.utils import seed_everything

seed_everything(98123)

PER_SUB_IMGCOUNT = 2

PRETRAIN_EPOCHS = 500
ITER_PER_EPOCH = 500


def args_parse():
    parser = argparse.ArgumentParser(description='Process training arguments for face identification.')
    parser.add_argument('--batch_size', type=int, default=3072,
                        help='Training batch size.')
    parser.add_argument('--img_scale', type=float, default=1.,
                        help='Image scale value.')
    parser.add_argument('--temperature', type=float, default=0.01,
                        help='Temperature to apply for softmax.')
    parser.add_argument('--apply_random_train_aug', type=bool, default=True,
                        help='If true applies random augmentation in training.')
    parser.add_argument('--weight', type=str, default=None,
                        help='File path to the trained checkpoint.')
    parser.add_argument('--cuda', type=str, default='0',
                        help='Cuda Device to use for training.')
    args = parser.parse_args()
    return args


def get_transform(args):
    img_shape = (int(IMG_SHAPE[0] * args.img_scale), int(IMG_SHAPE[1] * args.img_scale))

    test_transform = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_transform = transforms.Compose([
        # transforms.Resize(img_shape),
        transforms.RandomResizedCrop(img_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-15, 15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transform_ = train_transform if args.apply_random_train_aug else test_transform
    return transform_


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
        # y_pred = np.argmax(y_embd.detach().cpu().numpy(), axis=1)
        print(f"\rEpoch: {epoch} Iteration: {idx + 1}/{ITER_PER_EPOCH} Training Loss: {loss.item():.5f}", end=" ")
    tqdm.write(f"Training Loss: {train_loss / ITER_PER_EPOCH:2.3f}")


def get_chekpoint_dir(args):
    _exp = f"batchsize{args.batch_size}_temp{args.temperature}_imgsclae_{args.img_scale}_rnd_trns_{args.apply_random_train_aug}"
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, _exp)
    os.makedirs(checkpoint_dir)
    return checkpoint_dir


if __name__ == '__main__':
    args = args_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    pre_train_dataloader = get_pretrain_dataloader(args.batch_size, get_transform(args), PER_SUB_IMGCOUNT)

    model = resnet18(num_classes=256)
    model.cuda()

    if args.weight:
        model.load_state_dict(torch.load(args.weight)["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    def criterion(x):
        return contrastive_loss_entropy(*x, args.temperature)


    checkpoint_dir = get_chekpoint_dir(args)

    for epoch in tqdm(range(PRETRAIN_EPOCHS)):
        train_single_epoch(model, pre_train_dataloader)

        pre_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")

        torch.save({
            'epoch': epoch,
            'stage': "pretrain",
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, pre_checkpoint_path)
