import torch
import config
from tqdm import tqdm

def training_func(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        images = data["image"].to(config.DEVICE)
        targets = data["targets"].to(config.DEVICE)
        # for k,v in data.items():
        #     data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(images, targets)
        # _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def evaluation_func(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            images = data["image"].to(config.DEVICE)
            targets = data["targets"].to(config.DEVICE)

            # for k,v in data.items():
            #     data[k] = v.to(config.DEVICE)
            batch_preds, loss = model(images, targets)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)


