import os
import pickle

import torch
from tqdm import tqdm

from evaluation import evaluate


def train(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_score = 0
    best_epoch = 0
    save_config(config)
    for epoch in range(config.epoch):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, batch in pbar:
            optimizer.zero_grad()
            loss = model(batch, do_train=True)
            if config.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            pbar.set_description("Epoch %d, Loss %.4f" % (epoch, loss.item()))

        save_model(model, str(epoch), config)
        new_score = evaluate(model, val_loader, config)
        if new_score >= best_score:
            best_score = new_score
            best_epoch = epoch
            save_model(model, "best", config)
        print("Epoch %d, Evaluated Score %.4f, Best Score %.4f" % (epoch, new_score, best_score))
    print("best epoch: %d \t F1 = %.2f" % (best_epoch, best_score))


def save_model(model, name, config):
    base_path = './runs'
    model_to_save = model.module if hasattr(
        model, 'module') else model
    torch.save(
        model_to_save.state_dict(),
        os.path.join(base_path, config.name + "_" + name),
    )


def save_config(config):
    base_path = './runs'
    with open(os.path.join(base_path, config.name + "_config"), 'wb') as fb:
        pickle.dump(config, fb)
