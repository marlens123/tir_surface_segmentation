# credit the authors
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

from .utils.preprocess_helpers import get_preprocessing

from models.AutoSAM.loss_functions.dice_loss import SoftDiceLoss

from torch.utils.data import DataLoader
from .utils.data import Dataset
from .utils.train_helpers import compute_class_weights, set_seed
from .utils.losses import FocalLoss

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from .run_smp_finetune import validate

import wandb
import numpy as np
import os
import torch
import random
import torch.nn
import argparse
import importlib.util
import wandb
import time
import json
import warnings
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.satlaspretrain_models.model import Weights as SatlasWeights

from models.swin_transformer.model import ImageNetWeights


wandb.login()

parser = argparse.ArgumentParser(description="PyTorch Unet Training")

# prefix
parser.add_argument(
    "--pref", default="default", type=str, help="used for wandb logging"
)

# hyperparameters
parser.add_argument("--path_to_config", default="configs/swinb/imnet.json", type=str, help="Path to config file that stores hyperparameter setting.")

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
parser.add_argument("--gpu", default=0, help="GPU id to use.")

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
    '--wandb_entity', 
    default='sea-ice', 
    type=str, 
    help='wandb entity name'
)

parser.add_argument(
    '--loss_fn',
    default='dice_ce', 
    type=str, 
    choices=['dice_ce', 'focal', 'focal_dice_full', 'focal_dice_half'],
    help='loss function to use'
)

args = parser.parse_args()

if "swin" in args.path_to_config:
    arch = "SwinTransformer"
    swin_backbone = args.path_to_config.split("/")[1]
elif "sam2_unet" in args.path_to_config:
    arch = "SAM2-UNet"

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

    if arch == "SAM2-UNet":
        from models.SAM2_UNet.SAM2UNet import SAM2UNet
        if cfg_model["pretrain"] == "sam2_b+":
            model = SAM2UNet(model_cfg="sam2_hiera_b+.yaml", checkpoint_path="pretraining_checkpoints/SAM_2/sam2_hiera_base_plus.pt")
        elif cfg_model["pretrain"] == "sam2_l":
            model = SAM2UNet(model_cfg="sam2_hiera_l.yaml", checkpoint_path="pretraining_checkpoints/SAM_2/sam2_hiera_large.pt")
        elif cfg_model["pretrain"] == "sam2_s":
            model = SAM2UNet(model_cfg="sam2_hiera_s.yaml", checkpoint_path="pretraining_checkpoints/SAM_2/sam2_hiera_small.pt")
        elif cfg_model["pretrain"] == "sam2_t":
            model = SAM2UNet(model_cfg="sam2_hiera_t.yaml", checkpoint_path="pretraining_checkpoints/SAM_2/sam2_hiera_tiny.pt")
        elif cfg_model["pretrain"] == "none":
            model = SAM2UNet(model_cfg="sam2_hiera_b+.yaml", checkpoint_path=None)

    elif arch == "SwinTransformer":
        if cfg_model["pretrain"] == "satlas":
            from satlaspretrain_models.utils import Head

            # load model weights from satlas
            weights_manager = SatlasWeights()
            model = weights_manager.get_pretrained_model(
                model_identifier="Aerial_SwinB_SI",
                fpn=True,
                head=Head.SEGMENT,
                num_categories=cfg_model["num_classes"],
                device="cpu",
            )

        elif cfg_model["pretrain"] == "imagenet":
            from models.swin_transformer.utils import Head

            # load model weights from imagenet
            weights_manager = ImageNetWeights()
            model = weights_manager.get_pretrained_model(
                backbone=cfg_model["backbone"],
                fpn=True,
                head=Head.SEGMENT,
                num_categories=cfg_model["num_classes"],
                device="cpu",
            )
        elif cfg_model["pretrain"] == "none":
            from models.swin_transformer.utils import Head

            # load model weights from imagenet
            weights_manager = ImageNetWeights()
            model = weights_manager.get_pretrained_model(
                backbone=cfg_model["backbone"],
                fpn=True,
                head=Head.SEGMENT,
                num_categories=cfg_model["num_classes"],
                device="cpu",
                weights=None,
            )
        else:
            raise ValueError(
                "Invalid pretraining dataset. Choose 'imagenet' or 'none'."
            )

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
            if param.requires_grad and "encoder" in name or "iou" in name:
                if "bn" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad:
            print(name, "requires grad")

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

    best_mp_iou = 0.30

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
        _, _, mp_iou, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = validate(test_loader, model, epoch, scheduler, cfg_model, args)

        # save model weights
        if mp_iou > best_mp_iou:
            best_mp_iou = mp_iou
            save_path = os.path.join("runs/", args.pref, "weights/")
            os.makedirs(save_path, exist_ok = True)
            torch.save(
                model.state_dict(), os.path.join(save_path, "weights_{0}_epoch_{1}.pth".format(arch, epoch))
            )

        if epoch == 145:
            model_weights_path = os.path.join("models/", "weights/", arch)
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

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    focal_loss = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    dice_loss = SoftDiceLoss(
        batch_dice=True, do_bg=True, rebalance_weights=None
    )
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    # switch to train mode
    model.train()

    end = time.time()
    for i, tup in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

        if gpu is not None:
            img = tup[0].float().cuda(gpu, non_blocking=True)
            label = tup[1].long().cuda(gpu, non_blocking=True)
        else:
            img = tup[0].float()
            label = tup[1].long()

        # compute output
        start_time = time.time()

        if arch == "SAM2-UNet":
            preds = model.forward(img)

            # interpolate to original size
            preds = [F.interpolate(pred, size=(cfg_model["im_size"], cfg_model["im_size"]), mode="bilinear", align_corners=False) for pred in preds]

            losses = []
            for mask in preds:
                if loss_fn == "focal":
                    loss = focal_loss(mask, label.squeeze(1))
                elif loss_fn == "focal_dice_half":
                    dice_weight = 0.5
                    assert mask.shape[1] == 3, "got {}".format(mask.shape)
                    pred_softmax = F.softmax(mask, dim=1)
                    loss = focal_loss(mask, label.squeeze(1)) + dice_weight * dice_loss(
                        pred_softmax, label.squeeze(1)
                    )
                elif loss_fn == "focal_dice_full":
                    assert mask.shape[1] == 3, "got {}".format(mask.shape)
                    pred_softmax = F.softmax(mask, dim=1)
                    loss = focal_loss(mask, label.squeeze(1)) + dice_loss(
                        pred_softmax, label.squeeze(1)
                    )
                elif loss_fn == "dice_ce":
                    assert mask.shape[1] == 3, "got {}".format(mask.shape)
                    pred_softmax = F.softmax(mask, dim=1)
                    loss = ce_loss(mask, label.squeeze(1)) + dice_loss(
                        pred_softmax, label.squeeze(1)
                    )
                losses.append(loss)
            loss = sum(losses)

        else:
            mask = model.forward(img)

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

        jaccard = JaccardIndex(task="multiclass", num_classes=cfg_model["num_classes"]).to(
            gpu
        )
        jac = jaccard(torch.argmax(mask, dim=1), label.squeeze(1))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if use_wandb:
        wandb.log({"epoch": epoch, "train_loss": loss})
        wandb.log({"epoch": epoch, "train_jac": jac})

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    print("Time with GPU: ", end_time - start_time)