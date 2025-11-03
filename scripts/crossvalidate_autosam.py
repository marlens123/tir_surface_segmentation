import wandb
import os
from .run_autosam_finetune import train as autosam_train
from .run_autosam_finetune import validate as autosam_validate
from sklearn.model_selection import KFold
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
import argparse
from .utils.data import Dataset
from .utils.preprocess_helpers import get_preprocessing
from .utils.train_helpers import compute_class_weights, set_seed
from models.AutoSAM.models.build_autosam_seg_model import sam_seg_model_registry
import csv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--final_sweep", default=False, action="store_true"
)
parser.add_argument(
    "--seed", type=int, default=84
)
parser.add_argument(
    "--config", type=str, default="configs/autosam/sa-1b.json"
)
parser.add_argument(
    "--wandb_entity", type=str, default="sea-ice"
)
parser.add_argument(
    "--loss_fn", type=str, default=None
)
parser.add_argument(
    "--arch", type=str, default="AutoSAM"
)

args = parser.parse_args() 
pretrain = args.config.split("/")[-1].split(".")[0]

autosam_sweep_configuration = {
    "name": f"sweep_autosam_{pretrain}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [1e-4, 5e-4, 1e-3, 1e-2]},
        "batch_size": {"values": [1, 2, 4]},
        "optimizer": {"values": ["Adam", "SGD"]},
        "weight_decay": {"values": [0, 1e-5, 1e-3]},
        "loss_fn": {"values": ["dice_ce", "focal", "focal_dice_full", "focal_dice_half"]},
    },
}

final_autosam_sweep_configuration = {
    "name": f"sweep_autosam_{pretrain}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.0005]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.00001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_autosam_sweep_configuration_no = {
    "name": f"sweep_autosam_{pretrain}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def train_autosam(num, sweep_id, sweep_run_name, config, train_loader, test_loader, class_weights, hyper_config, args):
    run_name = f'{sweep_run_name}-{num}'
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )

    cfg_model = hyper_config['model']
    cfg_training = hyper_config['training']

    class_weights_np = class_weights
    class_weights = torch.from_numpy(class_weights).float().cuda(0)

    # create model
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

    torch.cuda.set_device(0)
    model = model.cuda(0)

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

    if config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
    elif config['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay'],
        )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

    cudnn.benchmark = True

    for epoch in range(150):
        autosam_train(
            train_loader,
            class_weights,
            model,
            optimizer,
            epoch,
            cfg_model,
            class_weights_np=class_weights_np,
            loss_fn=args.loss_fn if args.loss_fn is not None else config['loss_fn'],
        )
    _, val_miou, val_mp_iou, val_oc_iou, val_si_iou, y_true, y_pred, precision, recall, precision_macro, recall_macro, precision_micro, recall_micro, precision_weighted, recall_weighted, roc_auc, auc_pr, roc_curves = autosam_validate(test_loader, model, epoch, scheduler, cfg_model)

    precision_mp = precision[0]
    precision_si = precision[1]
    precision_oc = precision[2]
    recall_mp = recall[0]
    recall_si = recall[1]
    recall_oc = recall[2]  
    roc_auc_mp = roc_auc[0]
    roc_auc_si = roc_auc[1]
    roc_auc_oc = roc_auc[2]
    auc_pr_mp = auc_pr[0]
    auc_pr_si = auc_pr[1]
    auc_pr_oc = auc_pr[2]  

    run.log(dict(val_mean_iou=val_miou, val_melt_pond_iou=val_mp_iou, val_ocean_iou=val_oc_iou, val_sea_ice_iou=val_si_iou, precision_mp=precision_mp, precision_si=precision_si, precision_oc=precision_oc, recall_mp=recall_mp, recall_si=recall_si, recall_oc=recall_oc, precision_macro=precision_macro, recall_macro=recall_macro, precision_micro=precision_micro, recall_micro=recall_micro, precision_weighted=precision_weighted, recall_weighted=recall_weighted, roc_auc_mp=roc_auc_mp, roc_auc_si=roc_auc_si, roc_auc_oc=roc_auc_oc, auc_pr_mp=auc_pr_mp, auc_pr_si=auc_pr_si, auc_pr_oc=auc_pr_oc))
    run.finish()
    return val_miou, val_mp_iou, val_oc_iou, val_si_iou, precision, recall, precision_macro, recall_macro, roc_auc, auc_pr, roc_curves


def cross_validate_autosam():
    args=parser.parse_args()

    num_folds = 3

    X_path = "data/training/all_images.npy"
    y_path = "data/training/all_masks.npy"

    with open(args.config) as f:
        hyper_config = json.load(f)

    cfg_training = hyper_config['training']
    cfg_model = hyper_config['model']
    
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=14)

    X, y = np.load(X_path), np.load(y_path)

    class_weights = compute_class_weights(y_path)

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = f'{project_url}/groups/{sweep_id}'
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    cfg_model['im_size'] = sweep_run.config.im_size

    metrics_miou = []
    metrics_mp_iou = []
    metrics_oc_iou = []
    metrics_si_iou = []
    metrics_mp_precision = []
    metrics_si_precision = []
    metrics_oc_precision = []
    metrics_mp_recall = []
    metrics_si_recall = []
    metrics_oc_recall = []
    metrics_precision_macro = []
    metrics_recall_macro = []
    metrics_mp_roc_auc = []
    metrics_si_roc_auc = []
    metrics_oc_roc_auc = []
    metrics_mp_auc_pr = []
    metrics_si_auc_pr = []
    metrics_oc_auc_pr = []
    roc_curve_params_mp = []
    roc_curve_params_si = []
    roc_curve_params_oc = []

    for num, (train, test) in enumerate(kfold.split(X, y)):
        train_dataset = Dataset(cfg_model, cfg_training, mode="train", args=args, images=X[train], masks=y[train], preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]))
        test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, images=X[test], masks=y[test], preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]))

        train_loader = DataLoader(
            train_dataset,
            batch_size=sweep_run.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        reset_wandb_env()
        val_miou, val_mp_iou, val_oc_iou, val_si_iou, precision, recall, precision_macro, recall_macro, roc_auc, auc_pr, roc_curves = train_autosam(
            sweep_id=sweep_id,
            num=num,
            sweep_run_name=sweep_run_name,
            config=dict(sweep_run.config),
            train_loader=train_loader,
            test_loader=test_loader,
            class_weights=class_weights,
            hyper_config=hyper_config,
            args=args
        )
        metrics_miou.append(val_miou)
        metrics_mp_iou.append(val_mp_iou)
        metrics_oc_iou.append(val_oc_iou)
        metrics_si_iou.append(val_si_iou)
        metrics_mp_precision.append(precision[0].item())
        metrics_si_precision.append(precision[1].item())
        metrics_oc_precision.append(precision[2].item())
        metrics_mp_recall.append(recall[0].item())
        metrics_si_recall.append(recall[1].item())
        metrics_oc_recall.append(recall[2].item())
        metrics_precision_macro.append(precision_macro)
        metrics_recall_macro.append(recall_macro)
        metrics_mp_roc_auc.append(roc_auc[0])
        metrics_si_roc_auc.append(roc_auc[1])
        metrics_oc_roc_auc.append(roc_auc[2])
        metrics_mp_auc_pr.append(auc_pr[0])
        metrics_si_auc_pr.append(auc_pr[1])
        metrics_oc_auc_pr.append(auc_pr[2])
        roc_curve_params_mp.append(roc_curves[0])
        roc_curve_params_si.append(roc_curves[1])
        roc_curve_params_oc.append(roc_curves[2])

    # create results directory if it does not exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # add results to a final csv file
    if not os.path.exists("results/final_metrics"):
        os.makedirs("results/final_metrics")

    # only create the final_metrics_.txt file once
    if not os.path.exists(f"results/final_metrics/final_metrics_.txt"):
        with open(f"results/final_metrics/final_metrics_.txt", "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                "Model"
                "val_melt_pond_iou",
                "val_mean_iou",
                "val_ocean_iou",
                "val_sea_ice_iou",
                "val_melt_pond_precision",
                "val_sea_ice_precision",
                "val_ocean_precision",
                "val_melt_pond_recall",
                "val_sea_ice_recall",
                "val_ocean_recall",
                "val_precision_macro",
                "val_recall_macro",
                "val_melt_pond_roc_auc",
                "val_sea_ice_roc_auc",
                "val_ocean_roc_auc",
                "val_melt_pond_auc_pr",
                "val_sea_ice_auc_pr",
                "val_ocean_auc_pr",
            ]
        )
    # else only append to the existing file
    with open(f"results/final_metrics/final_metrics_.txt", "a") as f:
        writer = csv.writer(f)
        # add the average across folds to the txt file
        writer.writerow(
            [
                f"{args.arch}_{args.seed}_{cfg_model['pretrain']}_{args.loss_fn}",
                sum(metrics_mp_iou) / len(metrics_mp_iou),
                sum(metrics_miou) / len(metrics_miou),
                sum(metrics_oc_iou) / len(metrics_oc_iou),
                sum(metrics_si_iou) / len(metrics_si_iou),
                sum(metrics_mp_precision) / len(metrics_mp_precision),
                sum(metrics_si_precision) / len(metrics_si_precision),
                sum(metrics_oc_precision) / len(metrics_oc_precision),
                sum(metrics_mp_recall) / len(metrics_mp_recall),
                sum(metrics_si_recall) / len(metrics_si_recall),
                sum(metrics_oc_recall) / len(metrics_oc_recall),
                sum(metrics_precision_macro) / len(metrics_precision_macro),
                sum(metrics_recall_macro) / len(metrics_recall_macro),
                sum(metrics_mp_roc_auc) / len(metrics_mp_roc_auc),
                sum(metrics_si_roc_auc) / len(metrics_si_roc_auc),
                sum(metrics_oc_roc_auc) / len(metrics_oc_roc_auc),
                sum(metrics_mp_auc_pr) / len(metrics_mp_auc_pr),
                sum(metrics_si_auc_pr) / len(metrics_si_auc_pr),
                sum(metrics_oc_auc_pr) / len(metrics_oc_auc_pr),
            ]
        )

    for fold_params in range(len(roc_curve_params_mp)):
        # store roc curve parameters per fold as npy files
        np.save(f"results/roc_curves/roc_curve_params_mp_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}.npy", np.array(roc_curve_params_mp[fold_params], dtype=object))
        #np.save(f"results/roc_curves/roc_curve_params_si_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}.npy", np.array(roc_curve_params_si[fold_params], dtype=object))
        #np.save(f"results/roc_curves/roc_curve_params_oc_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}.npy", np.array(roc_curve_params_oc[fold_params], dtype=object))


    # resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    sweep_run.log(
        {
            "val_melt_pond_iou": sum(metrics_mp_iou) / len(metrics_mp_iou),
            "val_mean_iou": sum(metrics_miou) / len(metrics_miou),
            "val_ocean_iou": sum(metrics_oc_iou) / len(metrics_oc_iou),
            "val_sea_ice_iou": sum(metrics_si_iou) / len(metrics_si_iou),
            "precision_mp": sum(metrics_mp_precision) / len(metrics_mp_precision),
            "precision_si": sum(metrics_si_precision) / len(metrics_si_precision),
            "precision_oc": sum(metrics_oc_precision) / len(metrics_oc_precision),
            "recall_mp": sum(metrics_mp_recall) / len(metrics_mp_recall),
            "recall_si": sum(metrics_si_recall) / len(metrics_si_recall),
            "recall_oc": sum(metrics_oc_recall) / len(metrics_oc_recall),
            "precision_macro": sum(metrics_precision_macro) / len(metrics_precision_macro),
            "recall_macro": sum(metrics_recall_macro) / len(metrics_recall_macro),
            "val_melt_pond_roc_auc": sum(metrics_mp_roc_auc) / len(metrics_mp_roc_auc),
            "val_sea_ice_roc_auc": sum(metrics_si_roc_auc) / len(metrics_si_roc_auc),
            "val_ocean_roc_auc": sum(metrics_oc_roc_auc) / len(metrics_oc_roc_auc),
            "val_melt_pond_auc_pr": sum(metrics_mp_auc_pr) / len(metrics_mp_auc_pr),
            "val_sea_ice_auc_pr": sum(metrics_si_auc_pr) / len(metrics_si_auc_pr),
            "val_ocean_auc_pr": sum(metrics_oc_auc_pr) / len(metrics_oc_auc_pr),
        }
    )

    sweep_run.finish()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


def main(): 
    args = parser.parse_args() 
 
    if args.final_sweep:
        if "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_autosam_sweep_configuration_no
        else:
            sweep_config = final_autosam_sweep_configuration
        counts = 1
    else:
        sweep_config = autosam_sweep_configuration
        counts = 100

    set_seed(args.seed)

    wandb.login()
    sweep_id, count = wandb.sweep(sweep=sweep_config, project="eds", entity=args.wandb_entity), counts
    wandb.agent(sweep_id, function=cross_validate_autosam, count=count)

    wandb.finish()


if __name__ == "__main__":
    main()
