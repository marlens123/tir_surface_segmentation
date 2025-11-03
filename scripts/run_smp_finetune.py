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
import segmentation_models_pytorch as smp

from models.AutoSAM.loss_functions.dice_loss import SoftDiceLoss

from torch.utils.data import DataLoader
from .utils.data import Dataset
from .utils.train_helpers import compute_class_weights, set_seed, compute_class_frequencies
from .utils.losses import FocalLoss
from models.smp.build_rs_models import create_model_rs

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from .utils.train_helpers import compute_pixel_distance_to_edge

import wandb

wandb.login()

parser = argparse.ArgumentParser(description="PyTorch Unet Training")

# prefix
parser.add_argument(
    "--pref", default="default", type=str, help="used for wandb logging"
)

# hyperparameters
parser.add_argument(
    "--path_to_config", 
    default="configs/unetplusplus/aid.json", 
    type=str, 
    help="Path to config file that stores hyperparameter setting."
)

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
parser.add_argument("--gpu", 
                    default=0, 
                    help="GPU id to use."
                    )

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
parser.add_argument(
    "--arch", 
    type=str, 
    default="UnetPlusPlus", 
    choices=["Unet", "UnetPlusPlus", "MAnet", "Linknet", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "FPN", "PAN"]
)
parser.add_argument(
    "--patch_sampling", 
    default=False, 
    action='store_true', 
    help="whether to use patch sampling"
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

parser.add_argument(
    '--label_uncertainty',
    default=False,
    action='store_true',
    help='whether to use label uncertainty maps'
)

parser.add_argument(
    '--count_pixels',
    default=False,
    action='store_true',
    help='whether to count number of pixels per class in each epoch'
)

parser.add_argument(
    '--decision_rule',
    default=False,
    action='store_true',
    help='whether to use decision rule (divide by class weights during inference)'
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

    if cfg_model["pretrain"] == "none":
        cfg_model["pretrain"] = None

    # create model
    if cfg_model["pretrain"] == "imagenet" or cfg_model["pretrain"] == None:
        model = smp.create_model(
            arch=args.arch,
            encoder_name=cfg_model["backbone"],
            encoder_weights=cfg_model["pretrain"],
            in_channels=3,
            classes=cfg_model["num_classes"],
        )
        print("Using smp pretrain weights")
    else:
        model = create_model_rs(
            arch=args.arch,
            encoder_name=cfg_model["backbone"],
            pretrain=cfg_model["pretrain"],
            in_channels=3,
            classes=cfg_model["num_classes"],
        )
        print("Using custom pretrain weights")
    #print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

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

    # prior probabilities of each class in the training set
    class_frequencies_np = np.array(compute_class_frequencies(args.path_to_y_train))

    # Data loading code
    if args.patch_sampling:
        from .utils.data import PatchSamplingDataset
        train_dataset = PatchSamplingDataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]))
    else:
        train_dataset = Dataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]))
    
    # for validation, do not use patch sampling to get realistic evaluation
    test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), im_size=480)

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

    melt_pond_pixels_all_epochs = 0
    sea_ice_pixels_all_epochs = 0
    open_water_pixels_all_epochs = 0

    for epoch in range(cfg_training["num_epochs"]):
        print("EPOCH {}:".format(epoch + 1))

        if args.count_pixels:
            print("Counting number of pixels per class...")
            # train for one epoch
            mp, si, oc = train(
                train_loader,
                class_weights,
                model,
                optimizer,
                epoch,
                cfg_model,
                args,
                class_weights_np=class_weights_np,
                loss_fn=args.loss_fn,
                label_uncertainty=args.label_uncertainty,
                count_pixels=args.count_pixels
            )
            melt_pond_pixels_all_epochs += mp
            sea_ice_pixels_all_epochs += si
            open_water_pixels_all_epochs += oc
        
        else:
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
                loss_fn=args.loss_fn,
                label_uncertainty=args.label_uncertainty
            )
        
        _, _, mp_iou, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = validate(test_loader, model, epoch, scheduler, cfg_model, args, class_frequencies_np=class_frequencies_np,
                                                                                   class_weights_np=class_weights_np)

        # save model weights
        if mp_iou > best_mp_iou:
            best_mp_iou = mp_iou
            save_path = os.path.join("runs/", args.pref, "weights_resub/")
            os.makedirs(save_path, exist_ok = True)
            torch.save(
                model.state_dict(), os.path.join(save_path, "weights_{0}_epoch_{1}.pth".format(args.arch, epoch))
            )

        if epoch == 145:
            model_weights_path = os.path.join("models/", "weights/", args.arch)
            os.makedirs(model_weights_path, exist_ok = True)
            torch.save(
                model.state_dict(), os.path.join(model_weights_path, "{0}_epoch_{1}.pth".format(args.pref, epoch))
            )

    print("Total melt pond pixels over all epochs: ", melt_pond_pixels_all_epochs)
    print("Total sea ice pixels over all epochs: ", sea_ice_pixels_all_epochs)
    print("Total open water pixels over all epochs: ", open_water_pixels_all_epochs)
    print("Average melt pond pixels per epoch: ", melt_pond_pixels_all_epochs / cfg_training["num_epochs"])
    print("Average sea ice pixels per epoch: ", sea_ice_pixels_all_epochs / cfg_training["num_epochs"])
    print("Average open water pixels per epoch: ", open_water_pixels_all_epochs / cfg_training["num_epochs"])

    wandb.log({"total_melt_pond_pixels": melt_pond_pixels_all_epochs})
    wandb.log({"total_sea_ice_pixels": sea_ice_pixels_all_epochs})
    wandb.log({"total_open_water_pixels": open_water_pixels_all_epochs})
    wandb.log({"avg_melt_pond_pixels_per_epoch": melt_pond_pixels_all_epochs / cfg_training["num_epochs"]})
    wandb.log({"avg_sea_ice_pixels_per_epoch": sea_ice_pixels_all_epochs / cfg_training["num_epochs"]})
    wandb.log({"avg_open_water_pixels_per_epoch": open_water_pixels_all_epochs / cfg_training["num_epochs"]})


def train(
    train_loader,
    class_weights,
    model,
    optimizer,
    epoch,
    cfg_model,
    args=None,
    class_weights_np=None,
    loss_fn="dice_ce",
    label_uncertainty=False,
    teta=3.0,
    arch=None,
    scheduler=None,
    count_pixels=False,
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
        batch_dice=True
    )
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # switch to train mode
    model.train()

    melt_pond_pixels_per_epoch = 0
    sea_ice_pixels_per_epoch = 0
    open_water_pixels_per_epoch = 0

    loss_list = []

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

        melt_pond_pixels_per_epoch += torch.sum(label == 0).item()
        sea_ice_pixels_per_epoch += torch.sum(label == 1).item()
        open_water_pixels_per_epoch += torch.sum(label == 2).item()

        # compute distance maps for the batch if using label uncertainty
        if label_uncertainty:
            pixel_distances = compute_pixel_distance_to_edge(label.cpu().numpy(), teta=teta)

        # compute output
        start_time = time.time()
        mask = model.forward(img)
        end_time = time.time()
        print("Time for forward pass smp: ", end_time - start_time)

        if arch == "SAM2-UNet":
            preds = model.forward(img)

            losses = []
            for mask in preds:
                if loss_fn == "focal":
                    loss = focal_loss(mask, label.squeeze(1), pixel_distances=pixel_distances if label_uncertainty else None)
                elif loss_fn == "focal_dice_half":
                    dice_weight = 0.5
                    assert mask.shape[1] == 3, "got {}".format(mask.shape)
                    pred_softmax = F.softmax(mask, dim=1)
                    loss = focal_loss(mask, label.squeeze(1), pixel_distances=pixel_distances if label_uncertainty else None) + dice_weight * dice_loss(
                        pred_softmax, label.squeeze(1)
                    )
                elif loss_fn == "focal_dice_full":
                    assert mask.shape[1] == 3, "got {}".format(mask.shape)
                    pred_softmax = F.softmax(mask, dim=1)
                    loss = focal_loss(mask, label.squeeze(1), pixel_distances=pixel_distances if label_uncertainty else None) + dice_loss(
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
            if loss_fn == "focal":
                loss = focal_loss(mask, label.squeeze(1), pixel_distances=pixel_distances if label_uncertainty else None)
            elif loss_fn == "focal_dice_half":
                dice_weight = 0.5
                assert mask.shape[1] == 3
                pred_softmax = F.softmax(mask, dim=1)
                loss = focal_loss(mask, label.squeeze(1), pixel_distances=pixel_distances if label_uncertainty else None) + dice_weight * dice_loss(
                    pred_softmax, label.squeeze(1)
                )
            elif loss_fn == "focal_dice_full":
                assert mask.shape[1] == 3
                pred_softmax = F.softmax(mask, dim=1)
                loss = focal_loss(mask, label.squeeze(1), pixel_distances=pixel_distances if label_uncertainty else None) + dice_loss(
                    pred_softmax, label.squeeze(1)
                )
            elif loss_fn == "dice_ce":
                assert mask.shape[1] == 3
                pred_softmax = F.softmax(mask, dim=1)
                loss = ce_loss(mask, label.squeeze(1)) + dice_loss(
                    pred_softmax, label.squeeze(1)
                )
            print("Time for loss calculation smp: ", time.time() - end_time)

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

    if epoch >= 10:
        if scheduler is not None:
            scheduler.step(np.mean(loss_list))

    if use_wandb:
        wandb.log({"epoch": epoch, "train_loss": loss})
        wandb.log({"epoch": epoch, "train_jac": jac})

    if count_pixels:
        return melt_pond_pixels_per_epoch, sea_ice_pixels_per_epoch, open_water_pixels_per_epoch


def validate(
        val_loader, 
        model, 
        epoch, 
        scheduler, 
        cfg_model, 
        args=None, 
        arch=None, 
        class_frequencies_np=None,
        class_weights_np=None,
        decision_rule=False
        ):
    loss_list = []
    jac_list_mp = []
    jac_list_si = []
    jac_list_oc = []
    jac_mean = []
    pred_coll = []
    label_coll = []
    prob_coll = []
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=True)
    model.eval()

    tp = [0] * cfg_model["num_classes"]
    fp = [0] * cfg_model["num_classes"]
    fn = [0] * cfg_model["num_classes"]

    if args is None:
        gpu = 0
        use_wandb = False
    else:
        gpu = args.gpu
        use_wandb = args.use_wandb

    class_frequencies = torch.tensor(class_frequencies_np).view(1, 3, 1, 1)
    class_weights = torch.tensor(class_weights_np).view(1, 3, 1, 1)

    # Expand and repeat
    class_frequencies = class_frequencies.expand(1, 3, 480, 480)   # shape (1, 3, 480, 480)
    class_weights = class_weights.expand(1, 3, 480, 480)   # shape (1, 3, 480, 480)

    with torch.no_grad():
        for i, tup in enumerate(val_loader):
            if gpu is not None:
                img = tup[0].float().cuda(gpu, non_blocking=True)
                label = tup[1].long().cuda(gpu, non_blocking=True)
                class_frequencies = class_frequencies.cuda(gpu, non_blocking=True)
                class_weights = class_weights.cuda(gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]
            b, _, h, w = img.shape

            label_coll.append(label.squeeze(1).cpu())

            # compute output
            if arch == "SAM2-UNet":
                mask, _, _ = model.forward(img)
            else:
                mask = model.forward(img)
            mask = mask.view(b, -1, h, w)

            assert mask.shape[1] == 3
            pred_softmax = F.softmax(mask, dim=1)
            prob_coll.append(pred_softmax.cpu())
            loss = dice_loss(
                pred_softmax, label.squeeze(1)
            )  # self.ce_loss(pred, target.squeeze())
            loss_list.append(loss.item())

            if args.decision_rule or decision_rule:
                # divide by class weights
                print("Using decision rule")
                #pred = torch.argmax(mask / class_frequencies, dim=1)
                pred = torch.argmax(mask / class_weights, dim=1)
            else:
                pred = torch.argmax(mask, dim=1)

            pred_coll.append(pred.cpu())

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

            # compute confusion stats per class
            for c in range(cfg_model["num_classes"]):
                tp[c] += torch.sum((pred == c) & (label.squeeze(1) == c)).item()
                fp[c] += torch.sum((pred == c) & (label.squeeze(1) != c)).item()
                fn[c] += torch.sum((pred != c) & (label.squeeze(1) == c)).item()

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
        if scheduler is not None:
            scheduler.step(np.mean(loss_list))

    print(
        "Validating: Epoch: %2d Loss: %.4f"
        % (epoch, np.mean(loss_list))
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

    # precision weighted per class
    precision_weighted = (precision * (tp + fn).float() / (tp + fn).float().sum()).sum().item()
    recall_weighted = (recall * (tp + fn).float() / (tp + fn).float().sum()).sum().item()

    if use_wandb:
        wandb.log({"epoch": epoch, "val_precision_mp": precision[0].item()})
        wandb.log({"epoch": epoch, "val_precision_si": precision[1].item()})
        wandb.log({"epoch": epoch, "val_precision_oc": precision[2].item()})
        wandb.log({"epoch": epoch, "val_precision_macro": precision_macro})
        wandb.log({"epoch": epoch, "val_precision_micro": precision_micro})
        wandb.log({"epoch": epoch, "val_precision_weighted": precision_weighted})
        wandb.log({"epoch": epoch, "val_recall_mp": recall[0].item()})
        wandb.log({"epoch": epoch, "val_recall_si": recall[1].item()})
        wandb.log({"epoch": epoch, "val_recall_oc": recall[2].item()})
        wandb.log({"epoch": epoch, "val_recall_macro": recall_macro})
        wandb.log({"epoch": epoch, "val_recall_micro": recall_micro})
        wandb.log({"epoch": epoch, "val_recall_weighted": recall_weighted})

    ######################
    # ROC AUC
    ######################
    y_scores = np.concatenate(prob_coll, axis=0)
    y_true = np.concatenate(label_coll, axis=0)

    best_thresholds = []
    for c in range(cfg_model["num_classes"]):
        y_true_c = (y_true == c).flatten()
        y_scores_c = y_scores[:, c].flatten()

        thresholds = np.linspace(0, 1, 101)
        ious = []
        for t in thresholds:
            preds_c = (y_scores_c >= t)
            intersection = np.sum(preds_c & y_true_c)
            union = np.sum(preds_c | y_true_c)
            iou = intersection / union if union > 0 else 0
            ious.append(iou)

        best_t = thresholds[np.argmax(ious)]
        best_thresholds.append(best_t)
        print(f"Class {c}: best threshold = {best_t:.2f}")

    roc_auc_scores = []
    roc_curves = []
    pr_curves = []

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
        pr_curves.append(precision_recall_curve(y_true_c, y_scores_c))

    max_probs = y_scores.max(axis=1)
    print("mean max prob:", max_probs.mean())
    print("pct pixels with max_prob >= 0.5:", (max_probs>=0.5).mean())

    return np.mean(loss_list), np.mean(jac_mean), np.mean(jac_list_mp), np.mean(jac_list_oc), np.mean(jac_list_si), label_coll, pred_coll, precision, recall, precision_macro, recall_macro, precision_micro, recall_micro, precision_weighted, recall_weighted, roc_auc_scores, auc_pr_scores, roc_curves, pr_curves, max_probs


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