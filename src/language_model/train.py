import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

def train(dataset, model, args, device):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Currently training using {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--person', type=str, default="phoebe")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sequence-length', type=int, default=15)
    args = parser.parse_args()

    model_dir = f"../../models/language_model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset = Dataset(args, device)
    model = Model(dataset, device)
    model.to(device)

    train(dataset, model, args, device)
    torch.save(model.state_dict(), model_dir + f"{args.person}.language_model.pth")
