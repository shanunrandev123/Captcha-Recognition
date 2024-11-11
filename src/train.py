import os
import glob
import torch
import numpy as np


from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
from model import CaptchaModel
import engine
from pprint import pprint



def decode_preds(pred, encoder):
    pred = pred.permute(1,0,2)
    pred = torch.softmax(pred, 2)
    pred = torch.argmax(pred, 2)
    pred = pred.detach().cpu().numpy()

    captcha_preds = []

    for j in range(pred.shape[0]):
        temp = []
        for k in pred[j, :]:
            k = k - 1
            if k == -1:
                temp.append("-")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        prediction = "".join(temp)
        captcha_preds.append(prediction)

    return captcha_preds


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    label_enc = preprocessing.LabelEncoder()

    label_enc.fit(targets_flat)
    targets_enc = [label_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1
    print(targets_enc)
    print(label_enc.classes_)
    print(len(label_enc.classes_))

    train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    model = CaptchaModel(num_chars=len(label_enc.classes_))
    model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.8,
        mode="max"
    )

    for epoch in range(config.EPOCHS):
        train_loss = engine.training_func(model, train_loader, optimizer)
        test_preds, test_loss = engine.evaluation_func(model, test_loader)
        test_captcha_preds = []
        for p in test_preds:
            current_preds = decode_preds(p, label_enc)
            test_captcha_preds.extend(current_preds)
        pprint(list(zip(test_orig_targets, test_captcha_preds))[6:10])
        print(f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss}")

        # test_preds = np.vstack(test_preds)
        # test_preds = np.argmax(test_preds, axis=2).T
        # print(test_preds)
        #
        # accuracy = metrics.accuracy_score(test_targets.flatten(), test_preds.flatten())
        # print(f"Accuracy={accuracy}")
        #
        # scheduler.step(accuracy)

        # if accuracy > 0.95:
        #     torch.save(model.state_dict(), "./model.bin")
        #     break




if __name__ == "__main__":
    run_training()



