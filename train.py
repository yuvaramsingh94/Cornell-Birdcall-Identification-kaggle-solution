import os
import gc
import time
import shutil
import random
import warnings
import typing as tp
from pathlib import Path
from contextlib import contextmanager
from scipy import signal
import yaml
from joblib import delayed, Parallel
import cv2
import librosa
import audioread
import soundfile as sf
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn

# import torch.nn.functional as F
import torch.utils.data as data
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions
import torch.optim as optim
from torch.optim import lr_scheduler

# from warmup_scheduler import GradualWarmupScheduler
import torch.nn.functional as F
import torchvision.models as models
from utils2 import SpectrogramDataset, TrainDataset
from model import YuvNet

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore


@contextmanager
def timer(name: str) -> None:
    """Timer Util"""
    t0 = time.time()
    print("[{}] start".format(name))
    yield
    print("[{}] done in {:.0f} s".format(name, time.time() - t0))


# ROOT = Path.cwd().parent
# INPUT_ROOT = ROOT / "input"
ROOT = Path.cwd().parent
RAW_DATA = ROOT / "data"
TRAIN_RESAMPLED_AUDIO_DIRS = RAW_DATA / "train_audio_resampled"
TRAIN_RESAMPLED_BG_AUDIO_DIRS = RAW_DATA / "combined_audio" / "background"
# train = pd.read_csv(RAW_DATA / "resampled_train.csv")
train = pd.read_csv("train_filtered_v1.csv")

# seed: 1213
settings_str = """
globals:
  seed: 1213
  device: cuda
  num_epochs: 150
  output_dir: training_resnet_18_v1/
  use_fold: 1
  target_sr: 32000

dataset:
  name: SpectrogramDataset
  params:
    img_size: 224
    melspectrogram_parameters:
      n_mels: 155
      fmin: 0
      fmax: 16000
      n_fft: 1024
      hop_length: 256
    
split:
  name: StratifiedKFold
  params:
    n_splits: 5
    random_state: 42
    shuffle: True

loader:
  train:
    batch_size: 32
    shuffle: True
    num_workers: 15
    pin_memory: True
    drop_last: True
  val:
    batch_size: 32
    shuffle: False
    num_workers: 15
    pin_memory: True
    drop_last: False

model:
  gru_hidden_size: 512
  gru_layers: 1
  gru_bidirectional: True
  params:
    pretrained: True
    n_classes: 264

loss:
  name: BCELoss
  params: {}

optimizer:
  name: Adam
  params:
    lr: 3e-4

scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 10
"""
# n_mels= 155,n_fft = 1024, fmin= 0,hop_length = 512, fmax= fmax_val
settings = yaml.safe_load(settings_str)


def get_loaders_for_training(
    args_dataset, args_loader, train_file_list, val_file_list, train_df, bg_file_df
):
    ## make dataset
    train_dataset = TrainDataset(
        train_file_list,
        train_df,
        bg_file_df=bg_file_df,
        augmentation=True,
        **args_dataset
    )
    val_dataset = SpectrogramDataset(val_file_list, augmentation=False, **args_dataset)
    # make dataloader
    train_loader = data.DataLoader(train_dataset, **args_loader["train"])
    val_loader = data.DataLoader(val_dataset, **args_loader["val"])
    return train_loader, val_loader


def get_model(args):
    model = YuvNet(settings["model"])
    return model


def train_loop(
    manager, args, model, device, train_loader, optimizer, scheduler, loss_func
):
    """Run minibatch training loop"""
    cccc = 0
    while not manager.stop_trigger:
        print("it has executed now ", cccc)
        cccc += 1

        model.train()
        print("autopool ", model.autopool.alpha)
        for batch_idx, (data, target) in enumerate(train_loader):
            with manager.run_iteration():
                # print(data.shape)
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                # print('traininig shape ',data.shape)
                # print('target ',target)
                final_output, sigmoid_output = model(data)
                loss = loss_func(final_output, target)
                ppe.reporting.report({"train/loss": loss.item()})
                loss.backward()
                # print('autograd ',model.autopool.alpha)
                optimizer.step()
        scheduler.step()


def eval_for_batch(args, model, device, data, target, loss_func, eval_func_dict={}):
    """
    Run evaliation for valid

    This function is applied to each batch of val loader.
    """
    model.eval()
    data, target = data.to(device), target.to(device)
    # print('eval shape ',data.shape)
    final_output, sigmoid_output = model(data)
    # Final result will be average of averages of the same size
    val_loss = loss_func(final_output, target).item()
    ppe.reporting.report({"val/loss": val_loss})

    for eval_name, eval_func in eval_func_dict.items():
        # prediction.detach().cpu().numpy()

        y_true = target.detach().cpu().numpy()
        print("values of y_true ", y_true)
        # y_pred = torch.sigmoid(final_output).detach().cpu().numpy()
        y_pred = final_output.detach().cpu().numpy()
        print("values of prediction ", final_output)
        if eval_func != f1_score:
            eval_value = eval_func(np.round_(y_true), y_pred, average="micro")
        else:
            eval_value = eval_func(
                np.round_(y_true), np.round_(y_pred), average="micro"
            )
        ppe.reporting.report({"val/{}".format(eval_name): eval_value})


def set_extensions(
    manager, args, model, device, test_loader, optimizer, loss_func, eval_func_dict={}
):
    """set extensions for PPE"""

    my_extensions = [
        # # observe, report
        ppe_extensions.observe_lr(optimizer=optimizer),
        # ppe_extensions.ParameterStatistics(model, prefix='model'),
        # ppe_extensions.VariableStatisticsPlot(model),
        ppe_extensions.LogReport(),
        ppe_extensions.PlotReport(
            ["train/loss", "val/loss"], "epoch", filename="loss.png"
        ),
        ppe_extensions.PlotReport(["val/AUROC"], "epoch", filename="auroc.png"),
        ppe_extensions.PlotReport(["val/F1"], "epoch", filename="F1.png"),
        ppe_extensions.PlotReport(
            [
                "lr",
            ],
            "epoch",
            filename="lr.png",
        ),
        ppe_extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "lr",
                "train/loss",
                "val/loss",
                "val/AUROC",
                "val/F1",
                "elapsed_time",
            ]
        ),
        #         ppe_extensions.ProgressBar(update_interval=100),
        # # evaluation
        (
            ppe_extensions.Evaluator(
                test_loader,
                model,
                eval_func=lambda data, target: eval_for_batch(
                    args, model, device, data, target, loss_func, eval_func_dict
                ),
                progress_bar=True,
            ),
            (1, "epoch"),
        ),
        # # save model snapshot.
        (
            ppe_extensions.snapshot(
                target=model, filename="snapshot_epoch_{.updater.epoch}.pth"
            )
        ),
    ]

    # # set extensions to manager
    for ext in my_extensions:
        if isinstance(ext, tuple):
            manager.extend(ext[0], trigger=ext[1])
        else:
            manager.extend(ext)

    return manager


tmp_list = []

for ebird_d in TRAIN_RESAMPLED_AUDIO_DIRS.iterdir():
    if ebird_d.is_file():
        continue
    for wav_f in ebird_d.iterdir():
        tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])

tmp_bg_list = []
for ebird_d in TRAIN_RESAMPLED_BG_AUDIO_DIRS.iterdir():
    if ebird_d.is_file():
        continue
    for wav_f in ebird_d.iterdir():
        tmp_bg_list.append(["nocall", wav_f.name, float("NaN"), wav_f.as_posix()])

random.shuffle(tmp_bg_list)
train_wav_path_exist_bg = pd.DataFrame(
    tmp_bg_list[:500],
    columns=["ebird_code", "resampled_filename", "background_code", "file_path"],
)


train_wav_path_exist_bg_extra = pd.DataFrame(
    tmp_bg_list[500:],
    columns=["ebird_code", "resampled_filename", "background_code", "file_path"],
)


train_wav_path_exist = pd.DataFrame(
    tmp_list, columns=["ebird_code", "resampled_filename", "file_path"]
)

del tmp_list
train_filtered = train[["ebird_code", "resampled_filename", "background_code"]].copy()
train_all = pd.merge(
    train_filtered,
    train_wav_path_exist,
    on=["ebird_code", "resampled_filename"],
    how="inner",
)

train_all = pd.concat([train_all, train_wav_path_exist_bg]).reset_index()


### have to remove nocall for this
train_all = train_all[~train_all.ebird_code.isin(["nocall"])]


print(train.shape)
print(train_wav_path_exist.shape)
print(train_all.shape)

print(train_all.tail())


skf = StratifiedKFold(**settings["split"]["params"])

train_all["fold"] = -1
for fold_id, (train_index, val_index) in enumerate(
    skf.split(train_all, train_all["ebird_code"])
):
    train_all.iloc[val_index, -1] = fold_id


train_all = pd.read_csv("train.csv")


use_fold = settings["globals"]["use_fold"]
train_file_list = train_all.query("fold != @use_fold")[
    ["file_path", "ebird_code", "background_code"]
].values.tolist()
val_file_list = train_all.query("fold == @use_fold")[
    ["file_path", "ebird_code", "background_code"]
].values.tolist()

print(
    "[fold {}] train: {}, val: {}".format(
        use_fold, len(train_file_list), len(val_file_list)
    )
)


set_seed(settings["globals"]["seed"])
device = torch.device(settings["globals"]["device"])
output_dir = Path(settings["globals"]["output_dir"])


training_df = train_all[train_all["fold"] != use_fold]
print("training df sape ", training_df.shape)
print("bg path ", train_wav_path_exist_bg_extra.shape)
# # # get loader
train_loader, val_loader = get_loaders_for_training(
    settings["dataset"]["params"],
    settings["loader"],
    train_file_list,
    val_file_list,
    train_df=training_df,
    bg_file_df=train_wav_path_exist_bg_extra,
)

# # # get model
model = get_model(settings["model"])
model = model.to(device)

init_lr = 3e-4
warmup_factor = 10

warmup_epo = 1

n_epochs = settings["globals"]["num_epochs"]

optimizer = optim.Adam(model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)


# # # get loss
loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])

# # # create training manager
trigger = None

manager = ppe.training.ExtensionsManager(
    model,
    optimizer,
    settings["globals"]["num_epochs"],
    iters_per_epoch=len(train_loader),
    stop_trigger=trigger,
    out_dir=output_dir,
)

# # # set manager extensions
manager = set_extensions(
    manager,
    settings,
    model,
    device,
    val_loader,
    optimizer,
    loss_func,
    eval_func_dict={"AUROC": roc_auc_score, "F1": f1_score},
)


# print(model)
# # runtraining
train_loop(
    manager, settings, model, device, train_loader, optimizer, scheduler, loss_func
)
