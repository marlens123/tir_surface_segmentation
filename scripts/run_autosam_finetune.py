"""
Inspired by 2023 Xinrong Hu et al. https://github.com/xhu248/AutoSAM/blob/main/scripts/main_autosam_seg.py
"""

import os
import argparse
import time
import warnings
import numpy as np
import json

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import JaccardIndex

from models.AutoSAM.loss_functions.dice_loss import SoftDiceLoss
from models.AutoSAM.models.build_autosam_seg_model import sam_seg_model_registry

from torch.utils.data import DataLoader
from .utils.data import Dataset
from .utils.preprocess_helpers import get_preprocessing
from .utils.train_helpers import compute_class_weights, set_seed
from .utils.losses import FocalLoss
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import wandb

wandb.login()

parser = argparse.ArgumentParser(description="PyTorch AutoSam Training")

# prefix
parser.add_argument(
    "--pref", default="autosam_sa-1b", type=str, help="used for wandb logging"
)

# hyperparameters
parser.add_argument("--path_to_config", default="configs/autosam/sa-1b.json", type=str, help="Path to config file that stores hyperparameter setting.")

# processing
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "--seed", default=84, type=int, help="seed for initializing training."
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")

# data
parser.add_argument(
    "--path_to_X_train", type=str, default="data/training/train_images.npy"
)
parser.add_argument(
    "--path_to_y_train", type=str, default="data/training/train_masks.npy"
)
parser.add_argument(
    "--path_to_X_test", type=str, default="data/training/test_images.npy"
)
parser.add_argument(
    "--path_to_y_test", type=str, default="data/training/test_masks.npy"
)

# wandb monitoring
parser.add_argument(
    '--use_wandb', 
    default=True, 
    action='store_false', 
    help='use wandb for logging'
)

parser.add_argument(
    "--wandb_entity", type=str, default="sea-ice"
)

parser.add_argument(
    "--loss_fn", type=str, default="dice_ce", help="loss function to use", choices=["dice_ce", "focal", "focal_dice_full", "focal_dice_half"]
)

def main():
    args = parser.parse_args()

    set_seed(args.seed)

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s).")
        else:
            print("CUDA is not available.")

    with open(args.path_to_config) as f:
        config = json.load(f)

    main_worker(args, config)

def main_worker(args, config):

    cfg_model = config['model']
    cfg_training = config['training']

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if cfg_model["autosam_size"] == "vit_h":
        model_checkpoint = "pretraining_checkpoints/segment_anything/sam_vit_h_4b8939.pth"
        model = sam_seg_model_registry["vit_h"](
            num_classes=cfg_model["num_classes"], checkpoint=model_checkpoint
        )
    elif cfg_model["autosam_size"] == "vit_l":
        model_checkpoint = "pretraining_checkpoints/segment_anything/sam_vit_l_0b3195.pth"
        model = sam_seg_model_registry["vit_l"](
            num_classes=cfg_model["num_classes"], checkpoint=model_checkpoint
        )
    elif cfg_model["autosam_size"] == "vit_b":
        if cfg_model["pretrain"] == "none":
            model = sam_seg_model_registry["vit_b"](
                num_classes=cfg_model["num_classes"],
                checkpoint=None,
            )
        else:
            model_checkpoint = "pretraining_checkpoints/segment_anything/sam_vit_b_01ec64.pth"
            model = sam_seg_model_registry["vit_b"](
                num_classes=cfg_model["num_classes"],
                checkpoint=model_checkpoint,
            )
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # freeze weights in the image_encoder
    if not cfg_model["encoder_freeze"]:
        for name, param in model.named_parameters():
            if param.requires_grad and "iou" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    else:
        for name, param in model.named_parameters():
            if param.requires_grad and "image_encoder" in name or "iou" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    if cfg_training["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg_training["learning_rate"],
            weight_decay=cfg_training["weight_decay"],
        )
    elif cfg_training["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg_training["learning_rate"],
            momentum=0.9,
            weight_decay=cfg_training["weight_decay"],
        )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

    cudnn.benchmark = True

    # compute class weights
    if cfg_training["use_class_weights"]:
        class_weights = compute_class_weights(args.path_to_y_train)
        # for dice loss component
        class_weights_np = class_weights
        # for cce loss component
        class_weights = torch.from_numpy(class_weights).float().cuda(args.gpu)
    else:
        class_weights = None
        class_weights_np = None

    # Data loading code
    train_dataset = Dataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]))
    test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_training["batch_size"],
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.workers
    )

    save_dir = "runs/" + args.pref

    best_mp_iou = 0.38

    # wandb setup
    if args.use_wandb and args is not None:

        wandb.init(
            entity=args.wandb_entity,
            project='eds',
            name=args.pref,
        )
        wandb.config.update(config)
        wandb.watch(model, log_freq=2)

    for epoch in range(cfg_training["num_epochs"]):
        print("EPOCH {}:".format(epoch + 1))

        # train for one epoch
        train(
            train_loader,
            class_weights,
            model,
            optimizer,
            epoch,
            cfg_model,
            args,
            class_weights_np=class_weights_np,
            loss_fn=args.loss_fn
        )
        _, _, mp_iou, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = validate(test_loader, model, epoch, scheduler, cfg_model, args)

        # save model when melt pond IoU improved
        if mp_iou > best_mp_iou:
            best_mp_iou = mp_iou
            save_path = os.path.join("runs/", args.pref, "weights/")
            os.makedirs(save_path, exist_ok = True)
            torch.save(
                model.state_dict(), os.path.join(save_path, "weights_autosam_epoch_{}.pth".format(epoch))
            )

        if epoch == 145:
            model_weights_path = os.path.join("models/", "weights/", "AutoSAM/")
            os.makedirs(model_weights_path, exist_ok = True)
            torch.save(
                model.state_dict(), os.path.join(model_weights_path, "{0}_epoch_{1}.pth".format(args.pref, epoch))
            )


def train(
    train_loader,
    class_weights,
    model,
    optimizer,
    epoch,
    cfg_model,
    args=None,
    class_weights_np=None,
    loss_fn="dice_ce"
):

    if args is None:
        gpu = 0
        use_wandb = False
    else:
        gpu = args.gpu
        use_wandb = args.use_wandb

    dice_loss = SoftDiceLoss(
        batch_dice=True, do_bg=True, rebalance_weights=None
    )
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')

    # switch to train mode
    model.train()

    end = time.time()
    for i, tup in enumerate(train_loader):

        if gpu is not None:
            img = tup[0].float().cuda(gpu, non_blocking=True)
            label = tup[1].long().cuda(gpu, non_blocking=True)
        else:
            img = tup[0].float()
            label = tup[1].long()
        b, _, h, w = img.shape

        # compute output
        start_time = time.time()
        mask, _ = model(img)
        end_time = time.time()
        print("Time for forward pass autosam: ", end_time - start_time)
        mask = mask.view(b, -1, h, w)

        if loss_fn == "focal":
            loss = focal_loss(mask, label.squeeze(1))
        elif loss_fn == "focal_dice_half":
            dice_weight = 0.5
            assert mask.shape[1] == 3
            pred_softmax = F.softmax(mask, dim=1)
            loss = focal_loss(mask, label.squeeze(1)) + dice_weight * dice_loss(
                pred_softmax, label.squeeze(1)
            )
        elif loss_fn == "focal_dice_full":
            assert mask.shape[1] == 3
            pred_softmax = F.softmax(mask, dim=1)
            loss = focal_loss(mask, label.squeeze(1)) + dice_loss(
                pred_softmax, label.squeeze(1)
            )
        elif loss_fn == "dice_ce":
            assert mask.shape[1] == 3
            pred_softmax = F.softmax(mask, dim=1)
            loss = ce_loss(mask, label.squeeze(1)) + dice_loss(
                pred_softmax, label.squeeze(1)
            )
        print("Time for loss computation autosam: ", time.time() - end_time)

        jaccard = JaccardIndex(task="multiclass", num_classes=cfg_model["num_classes"]).to(
            gpu
        )
        jac = jaccard(torch.argmax(mask, dim=1), label.squeeze(1))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        end = time.time()

    if use_wandb:
        wandb.log({"epoch": epoch, "train_loss": loss})
        wandb.log({"epoch": epoch, "train_jac": jac})


def validate(val_loader, model, epoch, scheduler, cfg_model, args=None):
    loss_list = []
    jac_list_mp = []
    jac_list_si = []
    jac_list_oc = []
    jac_mean = []
    label_coll = []
    prob_coll = []
    pred_coll = []
    tp = [0] * cfg_model["num_classes"]
    fp = [0] * cfg_model["num_classes"]
    fn = [0] * cfg_model["num_classes"]
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=True)
    model.eval()

    if args is None:
        gpu = 0
        use_wandb = False
    else:
        gpu = args.gpu
        use_wandb = args.use_wandb

    with torch.no_grad():
        for i, tup in enumerate(val_loader):
            if gpu is not None:
                img = tup[0].float().cuda(gpu, non_blocking=True)
                label = tup[1].long().cuda(gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]
            b, _, h, w = img.shape

            label_coll.append(label.squeeze(1).cpu())

            # compute output
            mask, iou_pred = model(img)
            mask = mask.view(b, -1, h, w)
            iou_pred = iou_pred.squeeze().view(b, -1)
            iou_pred = torch.mean(iou_pred)

            assert mask.shape[1] == 3
            pred_softmax = F.softmax(mask, dim=1)
            prob_coll.append(pred_softmax.cpu())
            loss = dice_loss(
                pred_softmax, label.squeeze(1)
            )  # self.ce_loss(pred, target.squeeze())
            loss_list.append(loss.item())

            pred = torch.argmax(mask, dim=1)
            pred_coll.append(pred.cpu())

            # compute confusion stats per class
            for c in range(cfg_model["num_classes"]):
                tp[c] += torch.sum((pred == c) & (label.squeeze(1) == c)).item()
                fp[c] += torch.sum((pred == c) & (label.squeeze(1) != c)).item()
                fn[c] += torch.sum((pred != c) & (label.squeeze(1) == c)).item()

            jaccard = JaccardIndex(
                task="multiclass", num_classes=cfg_model["num_classes"], average=None
            ).to(gpu)
            jaccard_mean = JaccardIndex(
                task="multiclass", num_classes=cfg_model["num_classes"]
            ).to(gpu)

            jac = jaccard(pred_softmax, label.squeeze(1))
            jac_m = jaccard_mean(pred_softmax, label.squeeze(1))

            jac_list_mp.append(jac[0].item())
            jac_list_si.append(jac[1].item())
            jac_list_oc.append(jac[2].item())
            jac_mean.append(jac_m.item())

            if use_wandb:
                wandb.log({"epoch": epoch, "val_loss_{}".format(i): loss.item()})
                wandb.log({"epoch": epoch, "val_jac_mp_{}".format(i): jac[0].item()})
                wandb.log({"epoch": epoch, "val_jac_si_{}".format(i): jac[1].item()})
                wandb.log({"epoch": epoch, "val_jac_oc_{}".format(i): jac[2].item()})
                wandb.log({"epoch": epoch, "val_jac_{}".format(i): jac_m.item()})

    if use_wandb:
        wandb.log({"epoch": epoch, "val_loss": np.mean(loss_list)})
        wandb.log({"epoch": epoch, "val_jac_mp": np.mean(jac_list_mp)})
        wandb.log({"epoch": epoch, "val_jac_si": np.mean(jac_list_si)})
        wandb.log({"epoch": epoch, "val_jac_oc": np.mean(jac_list_oc)})
        wandb.log({"epoch": epoch, "val_jac": np.mean(jac_mean)})

    if epoch >= 10:
        scheduler.step(np.mean(loss_list))

    print(
        "Validating: Epoch: %2d Loss: %.4f IoU_pred: %.4f"
        % (epoch, np.mean(loss_list), iou_pred.item())
    )

    # PRECISION / RECALL
    ####################
    tp = torch.tensor(tp)
    fp = torch.tensor(fp)
    fn = torch.tensor(fn)

    tp_total = tp.sum().float()
    fp_total = fp.sum().float()
    fn_total = fn.sum().float()

    # compute precision/recall per class
    precision = tp.float() / (tp + fp).clamp(min=1)
    recall = tp.float() / (tp + fn).clamp(min=1)

    # you can also average (macro) or weight by class frequency (micro)
    precision_macro = precision.mean().item()
    recall_macro = recall.mean().item()

    precision_micro = (tp_total / (tp_total + fp_total).clamp(min=1)).item()
    recall_micro = (tp_total / (tp_total + fn_total).clamp(min=1)).item()

    precision_weighted = (precision * (tp + fn).float() / (tp + fn).float().sum()).sum().item()
    recall_weighted = (recall * (tp + fn).float() / (tp + fn).float().sum()).sum().item()

    if use_wandb:
        wandb.log({"epoch": epoch, "val_precision_mp": precision[0].item()})
        wandb.log({"epoch": epoch, "val_precision_si": precision[1].item()})
        wandb.log({"epoch": epoch, "val_precision_oc": precision[2].item()})
        wandb.log({"epoch": epoch, "val_precision_macro": precision_macro})
        wandb.log({"epoch": epoch, "val_precision_micro": precision_micro})
        wandb.log({"epoch": epoch, "val_recall_mp": recall[0].item()})
        wandb.log({"epoch": epoch, "val_recall_si": recall[1].item()})
        wandb.log({"epoch": epoch, "val_recall_oc": recall[2].item()})
        wandb.log({"epoch": epoch, "val_recall_macro": recall_macro})
        wandb.log({"epoch": epoch, "val_recall_micro": recall_micro})

    ######################
    # ROC AUC
    ######################
    y_scores = np.concatenate(prob_coll, axis=0)
    y_true = np.concatenate(label_coll, axis=0)

    roc_auc_scores = []
    roc_curves = []

    for c in range(cfg_model["num_classes"]):
        y_true_c = (y_true == c).flatten()
        y_scores_c = y_scores[:, c].flatten()
        roc_auc = roc_auc_score(y_true_c, y_scores_c)
        roc_auc_scores.append(roc_auc)
        fpr, tpr, thresholds = roc_curve(y_true_c, y_scores_c)
        roc_curves.append((fpr, tpr, thresholds))

    ######################

    # AUC-PR
    ######################
    auc_pr_scores = []

    for c in range(cfg_model["num_classes"]):
        y_true_c = (y_true == c).flatten()
        y_scores_c = y_scores[:, c].flatten()
        ap = average_precision_score(y_true_c, y_scores_c)
        auc_pr_scores.append(ap)

    return np.mean(loss_list), np.mean(jac_mean), np.mean(jac_list_mp), np.mean(jac_list_oc), np.mean(jac_list_si), label_coll, pred_coll, precision, recall, precision_macro, recall_macro, precision_micro, recall_micro, precision_weighted, recall_weighted, roc_auc_scores, auc_pr_scores, roc_curves


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    print("Time with GPU: ", end_time - start_time)