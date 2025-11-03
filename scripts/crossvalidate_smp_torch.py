import wandb
import os
from sklearn.model_selection import KFold
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import json
import numpy as np
from .utils.data import Dataset
from .utils.train_helpers import compute_class_weights, set_seed, compute_class_frequencies
from .run_smp_finetune import train as unet_torch_train
from .run_smp_finetune import validate as unet_torch_validate
import segmentation_models_pytorch as smp
import argparse
from .utils.preprocess_helpers import get_preprocessing
from sklearn.metrics import confusion_matrix
from models.smp.build_rs_models import create_model_rs
from models.swin_transformer.model import ImageNetWeights
from .utils.data import PatchSamplingDataset
import csv
import torch.nn.functional as F
from torchmetrics import JaccardIndex

parser = argparse.ArgumentParser(description="PyTorch Unet Training")
parser.add_argument(
    "--arch", type=str, default="UnetPlusPlus"
)
parser.add_argument(
    "--final_sweep", default=False, action="store_true"
)
parser.add_argument(
    "--seed", type=int, default=84
)
parser.add_argument(
    "--config", type=str, default="configs/unetplusplus/aid.json"
)
parser.add_argument(
    "--wandb_entity", type=str, default="sea-ice"
)
parser.add_argument(
    "--num_sweep_runs", type=int, default=50
)
parser.add_argument(
    "--search_sweep_config", type=str, default= "all", choices=["all", "lr", "loss", "balance", "teta"]
)
parser.add_argument(
    '--use_wandb', 
    default=False, 
    action='store_true', 
    help='hackey here but we need this to prevent double-logging when called from other scripts'
)
parser.add_argument("--gpu", 
                    default=0, 
                    help="GPU id to use."
                    )

parser.add_argument(
    "--loss_fn", type=str, default=None
)

parser.add_argument(
    '--label_uncertainty',
    default=False,
    action='store_true',
    help='whether to use label uncertainty maps'
)

parser.add_argument(
    "--patch_sampling", 
    default=False, 
    action='store_true', 
    help="whether to use patch sampling"
)

parser.add_argument(
    '--count_pixels',
    default=False,
    action='store_true',
    help='whether to count number of pixels per class in each epoch'
)

parser.add_argument(
    '--ps128',
    default=False,
    action='store_true',
    help='whether to use 128x128 patches sampled from full images during training'
)

parser.add_argument(
    '--decision_rule',
    default=False,
    action='store_true',
    help='whether to use decision rule (divide by class weights during inference)'
)

parser.add_argument(
    '--extract_features',
    default=False,
    action='store_true',
    help='whether to extract features from the final trained models'
)

args = parser.parse_args() 
pretrain = args.config.split("/")[-1].split(".")[0]

# seed everything
set_seed(args.seed)

final_unet_sweep_configuration_imnet = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [1e-5]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_unet_sweep_configuration_aid = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_unet_sweep_configuration_rsd = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_unet_sweep_configuration_no = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_unetplusplus_sweep_configuration_imnet = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
        "loss_fn": {"values": ["dice_ce"]},        
    },
}

final_unetplusplus_sweep_configuration_aid = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_unetplusplus_sweep_configuration_rsd = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.00001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_unetplusplus_sweep_configuration_no = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.0005]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_psp_sweep_configuration_imnet = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
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

final_psp_sweep_configuration_aid = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_psp_sweep_configuration_rsd = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_psp_sweep_configuration_no = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_deeplab_sweep_configuration_imnet = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
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

final_deeplab_sweep_configuration_aid = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_deeplabplus_sweep_configuration_imnet = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
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

final_deeplabplus_sweep_configuration_aid = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.00001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_deeplabplus_sweep_configuration_rsd = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.00001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_deeplabplus_sweep_configuration_no = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.01]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["SGD"]},
        "weight_decay": {"values": [0.001]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_swinb_sweep_configuration_imnet = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [1e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [1e-5]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_swinb_sweep_configuration_satlas = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [1e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [1e-5]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

final_sam2_sweep_configuration = {
    "name": f"final_sweep_{args.arch}_{pretrain}_{args.seed}_{args.loss_fn}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [1e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [1e-5]},
        "loss_fn": {"values": ["dice_ce"]},
    },
}

smp_torch_sweep_configuration = {
    "name": f"sweep_smp_torch_{args.arch}_{pretrain}",
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

lr_sweep_configuration = {
    "name": f"sweep_lr_{args.arch}_{pretrain}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 5e-4, 1e-3, 1e-2]},
        "batch_size": {"values": [4]},
        "im_size": {"values": [480]},
        "weight_decay": {"values": [0]},
        "optimizer": {"values": ["Adam"]},
        "loss_fn": {"values": ["focal_dice_full"]},
    },
}

loss_sweep_configuration = {
    "name": f"sweep_loss_{args.arch}_{pretrain}",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [1e-4]},
        "batch_size": {"values": [4]},
        "weight_decay": {"values": [0]},
        "optimizer": {"values": ["Adam"]},
        "loss_fn": {"values": ["dice_ce", "focal", "focal_dice_full", "focal_dice_half"]},
    },
}

balance_sweep_configuration = {
    "name": f"sweep_balance_{args.arch}_{pretrain}",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "weight_decay": {"values": [0]},
        "optimizer": {"values": ["Adam"]},
        "loss_fn": {"values": ["focal"]},
        "balance_method": {"values": ["none", "patch_sampling", "pixel_distance", "ps128"]},
    },
}

teta_sweep_configuration = {
    "name": f"sweep_teta_{args.arch}_{pretrain}",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "weight_decay": {"values": [0]},
        "optimizer": {"values": ["Adam"]},
        "loss_fn": {"values": ["focal"]},
        "balance_method": {"values": ["pixel_distance"]},
        "teta": {"values": [1.0, 3.0, 9.0, 15.0]},
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


def train_smp_torch(num, sweep_id, sweep_run_name, config, train_loader, test_loader, class_weights, class_frequencies_np, train_loader_features=None):
    args2 = parser.parse_args()
    run_name = f'{sweep_run_name}-{num}'
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )

    with open(args2.config) as f:
        hyper_config = json.load(f)

    cfg_model = hyper_config['model']
    cfg_training = hyper_config['training']

    if cfg_model["pretrain"] == "none":
        cfg_model["pretrain"] = None

    class_weights_np = class_weights
    class_weights = torch.from_numpy(class_weights).float().cuda(0)

    if args2.arch == "SAM2-UNet":
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
        elif cfg_model["pretrain"] == None:
            model = SAM2UNet(model_cfg="sam2_hiera_b+.yaml", checkpoint_path=None)
        else:
            raise ValueError("Invalid pretrain option for SAM2-UNet")

    elif args2.arch == "SwinTransformer":
        if cfg_model["pretrain"] == "imagenet":
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
        elif cfg_model["pretrain"] == "none" or cfg_model["pretrain"] == None:
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
        elif cfg_model["pretrain"] == "satlas":
            from satlaspretrain_models.utils import Head
            from models.satlaspretrain_models.model import Weights as SatlasWeights

            # load model weights from satlas
            weights_manager = SatlasWeights()
            model = weights_manager.get_pretrained_model(
                model_identifier="Aerial_SwinB_SI",
                fpn=True,
                head=Head.SEGMENT,
                num_categories=cfg_model["num_classes"],
                device="cpu",
            )
    else:
        # create smp model
        if cfg_model["pretrain"] == "imagenet" or cfg_model["pretrain"] == None:
            model = smp.create_model(
                arch=args2.arch,
                encoder_name=cfg_model["backbone"],
                encoder_weights=cfg_model["pretrain"],
                in_channels=3,
                classes=cfg_model["num_classes"],
            )
            print("using smp model")
        else:
            model = create_model_rs(
                arch=args2.arch,
                encoder_name=cfg_model["backbone"],
                pretrain=cfg_model["pretrain"],
                in_channels=3,
                classes=cfg_model["num_classes"],
                get_features=True if args2.extract_features else False,
            )
            print("using custom model")

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
    scheduler = None

    cudnn.benchmark = True

    for epoch in range(150):
        unet_torch_train(
            train_loader,
            class_weights,
            model,
            optimizer,
            epoch,
            cfg_model,
            class_weights_np=class_weights_np,
            loss_fn=args2.loss_fn if args2.loss_fn is not None else config['loss_fn'],
            label_uncertainty=args.label_uncertainty,
            teta=config['teta'] if 'teta' in config else 3.0,
            scheduler=scheduler,
            arch=args2.arch,
        )
    if args2.extract_features:
        feature_dir = f'features/{args2.arch}_{pretrain}_fold_{num}_train/'
        features = np.zeros((len(train_loader_features), 512, 15, 15)).astype(np.uint8)
        mp_ious = np.zeros((len(train_loader_features),))
        os.makedirs(feature_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(train_loader_features):
                image = sample[0].cuda(0)
                label = sample[1].cuda(0)
                outputs = model.encoder(image)
                prediction, features_test = model.forward(image, return_features=True)
                pred_softmax = F.softmax(prediction, dim=1)
                features_test = features_test[-1].cpu().numpy()
                features[i] = features_test
                jaccard = JaccardIndex(
                    task="multiclass", num_classes=cfg_model["num_classes"], average=None
                ).cuda(0)
                mp_ious[i] = jaccard(pred_softmax, label.squeeze(1))[0].cpu().numpy()
        np.save(os.path.join(feature_dir, f'mp_ious.npy'), mp_ious)
        np.save(os.path.join(feature_dir, f'features.npy'), features)
        print(f"Extracted features saved to {feature_dir}")
        run.finish()
        return
    
    _, val_miou, val_mp_iou, val_oc_iou, val_si_iou, y_true, y_pred, precision, recall, precision_macro, recall_macro, precision_micro, recall_micro, precision_weighted, recall_weighted, roc_auc, auc_pr, roc_curves, pr_curves, max_probs = unet_torch_validate(test_loader, model, epoch, scheduler, cfg_model, args=args2, arch=args2.arch, class_frequencies_np=class_frequencies_np, class_weights_np=class_weights_np, decision_rule=args2.decision_rule)
    cm = confusion_matrix(np.array(y_true).flatten(), np.array(y_pred).flatten())

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
    return val_miou, val_mp_iou, val_oc_iou, val_si_iou, cm, precision, recall, precision_macro, recall_macro, roc_auc, auc_pr, roc_curves, pr_curves, max_probs


def cross_validate_smp_torch():
    args = parser.parse_args()
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
    # prior probabilities of each class in the training set
    class_frequencies_np = np.array(compute_class_frequencies(y_path))

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
    all_max_probs = []
    pr_curves_mp = []
    pr_curves_si = []
    pr_curves_oc = []

    confusion_matrices = []

    for num, (train, test) in enumerate(kfold.split(X, y)):
        if ('balance_method' in sweep_run.config and sweep_run.config['balance_method'] == 'patch_sampling') or args.patch_sampling:
            print("using patch sampling")
            train_dataset = PatchSamplingDataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[train], masks=y[train])
            test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[test], masks=y[test])
        elif ('balance_method' in sweep_run.config and sweep_run.config['balance_method'] == 'ps128') or args.ps128:
            print("using 128x128 patches sampled from full images during training")
            train_dataset = Dataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[train], masks=y[train], im_size=128)
            test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[test], masks=y[test])
        else:
            train_dataset = Dataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[train], masks=y[train])
            test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[test], masks=y[test])

        train_loader = DataLoader(
            train_dataset,
            batch_size=sweep_run.config.batch_size,
            shuffle=True,
            num_workers=4,
        )

        train_loader_features = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4
        )
        reset_wandb_env()

        if args.extract_features:
            print("Extracting features only, no metric computation.")
            train_smp_torch(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
                train_loader=train_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                class_frequencies_np=class_frequencies_np,
                train_loader_features=train_loader_features,
            )
            continue

        val_miou, val_mp_iou, val_oc_iou, val_si_iou, cm, precision, recall, precision_macro, recall_macro, roc_auc, auc_pr, roc_curves, pr_curves, max_probs = train_smp_torch(
            sweep_id=sweep_id,
            num=num,
            sweep_run_name=sweep_run_name,
            config=dict(sweep_run.config),
            train_loader=train_loader,
            test_loader=test_loader,
            class_weights=class_weights,
            class_frequencies_np=class_frequencies_np,
        )
        metrics_miou.append(val_miou)
        metrics_mp_iou.append(val_mp_iou)
        metrics_oc_iou.append(val_oc_iou)
        metrics_si_iou.append(val_si_iou)
        metrics_mp_precision.append(precision[0])
        metrics_si_precision.append(precision[1])
        metrics_oc_precision.append(precision[2])
        metrics_mp_recall.append(recall[0])
        metrics_si_recall.append(recall[1])
        metrics_oc_recall.append(recall[2])
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
        pr_curves_mp.append(pr_curves[0])
        pr_curves_si.append(pr_curves[1])
        pr_curves_oc.append(pr_curves[2])
        all_max_probs.append(max_probs)

        confusion_matrices.append(cm)

    if args.extract_features:
        print("Extracting features only, no metric computation.")
        sweep_run = wandb.init(id=sweep_run_id, resume="must")
        sweep_run.finish()
        return

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
    
    # store max probabilities and melt scores per fold as npy files
    for fold in range(len(max_probs)):
        np.save(f"results/max_probs/max_probs_{args.arch}_seed_{args.seed}_fold_{fold}_{cfg_model['pretrain']}_{args.loss_fn}_{sweep_run_name}.npy", np.array(max_probs[fold], dtype=object))
        
    for fold_params in range(len(roc_curve_params_mp)):
        # store roc curve parameters per fold as npy files
        np.save(f"results/roc_curves/roc_curve_params_mp_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}_{sweep_run_name}.npy", np.array(roc_curve_params_mp[fold_params], dtype=object))
        np.save(f"results/roc_curves/roc_curve_params_si_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}_{sweep_run_name}.npy", np.array(roc_curve_params_si[fold_params], dtype=object))
        np.save(f"results/roc_curves/roc_curve_params_oc_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}_{sweep_run_name}.npy", np.array(roc_curve_params_oc[fold_params], dtype=object))
        np.save(f"results/pr_curves/pr_curve_params_mp_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}_{sweep_run_name}.npy", np.array(pr_curves_mp[fold_params], dtype=object))
        np.save(f"results/pr_curves/pr_curve_params_si_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}_{sweep_run_name}.npy", np.array(pr_curves_si[fold_params], dtype=object))
        np.save(f"results/pr_curves/pr_curve_params_oc_{args.arch}_seed_{args.seed}_fold_{fold_params}_{cfg_model['pretrain']}_{args.loss_fn}_{sweep_run_name}.npy", np.array(pr_curves_oc[fold_params], dtype=object))


    if args.final_sweep:
        # average confusion matrices
        confusion_matrix_avg = np.mean(np.stack(confusion_matrices, axis=0), axis=0)
        np.save(f"confusion_matrices/confusion_matrix_{sweep_run_name}.npy", confusion_matrix_avg)

    # create a csv with all the final metrics
    with open(f"results/sweep_results_{sweep_run_name}.csv", mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        for i in range(num_folds):
            writer.writerow(
                [
                    metrics_mp_iou[i],
                    metrics_miou[i],
                    metrics_oc_iou[i],
                    metrics_si_iou[i],
                    metrics_mp_precision[i],
                    metrics_si_precision[i],
                    metrics_oc_precision[i],
                    metrics_mp_recall[i],
                    metrics_si_recall[i],
                    metrics_oc_recall[i],
                    metrics_precision_macro[i],
                    metrics_recall_macro[i],
                    metrics_mp_roc_auc[i],
                    metrics_si_roc_auc[i],
                    metrics_oc_roc_auc[i],
                    metrics_mp_auc_pr[i],
                    metrics_si_auc_pr[i],
                    metrics_oc_auc_pr[i],
                ]
            )
        writer.writerow(
            [
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

    # resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    # log metric to sweep run
    sweep_run.log(
        {
            "val_melt_pond_iou": sum(metrics_mp_iou) / len(metrics_mp_iou),
            "val_mean_iou": sum(metrics_miou) / len(metrics_miou),
            "val_ocean_iou": sum(metrics_oc_iou) / len(metrics_oc_iou),
            "val_sea_ice_iou": sum(metrics_si_iou) / len(metrics_si_iou),
            "val_melt_pond_precision": sum(metrics_mp_precision) / len(metrics_mp_precision),
            "val_sea_ice_precision": sum(metrics_si_precision) / len(metrics_si_precision),
            "val_ocean_precision": sum(metrics_oc_precision) / len(metrics_oc_precision),
            "val_melt_pond_recall": sum(metrics_mp_recall) / len(metrics_mp_recall),
            "val_sea_ice_recall": sum(metrics_si_recall) / len(metrics_si_recall),
            "val_ocean_recall": sum(metrics_oc_recall) / len(metrics_oc_recall),
            "val_precision_macro": sum(metrics_precision_macro) / len(metrics_precision_macro),
            "val_recall_macro": sum(metrics_recall_macro) / len(metrics_recall_macro),
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
    wandb.login()
    args=parser.parse_args()

    set_seed(args.seed)

    count = 1

    if args.final_sweep and args.arch == "Unet":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_unet_sweep_configuration_aid
        elif "rsd" in args.config:
            print("Using RSD configuration")
            sweep_config = final_unet_sweep_configuration_rsd
        elif "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_unet_sweep_configuration_no
        else:
            sweep_config = final_unet_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "UnetPlusPlus":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_unetplusplus_sweep_configuration_aid
        elif "rsd" in args.config:
            print("Using RSD configuration")
            sweep_config = final_unetplusplus_sweep_configuration_rsd
        elif "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_unetplusplus_sweep_configuration_no
        else:
            sweep_config = final_unetplusplus_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "PSPNet":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_psp_sweep_configuration_aid
        elif "rsd" in args.config:
            print("Using RSD configuration")
            sweep_config = final_psp_sweep_configuration_rsd
        elif "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_psp_sweep_configuration_no
        else:
            sweep_config = final_psp_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "DeepLabV3":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_deeplab_sweep_configuration_aid
        else:
            sweep_config = final_deeplab_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "DeepLabV3Plus":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_deeplabplus_sweep_configuration_aid
        elif "rsd" in args.config:
            print("Using RSD configuration")
            sweep_config = final_deeplabplus_sweep_configuration_rsd
        elif "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_deeplabplus_sweep_configuration_no
        else:
            sweep_config = final_deeplabplus_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "SwinTransformer":
        if "satlas" in args.config:
            print("Using Satlas configuration")
            sweep_config = final_swinb_sweep_configuration_satlas
        else:
            sweep_config = final_swinb_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "SAM2-UNet":
        sweep_config = final_sam2_sweep_configuration
    else:
        if args.search_sweep_config == "all":
            sweep_config = smp_torch_sweep_configuration
        elif args.search_sweep_config == "lr":
            sweep_config = lr_sweep_configuration
        elif args.search_sweep_config == "loss":
            sweep_config = loss_sweep_configuration
        elif args.search_sweep_config == "balance":
            sweep_config = balance_sweep_configuration
        elif args.search_sweep_config == "teta":
            sweep_config = teta_sweep_configuration
        else:
            sweep_config = smp_torch_sweep_configuration
        count = args.num_sweep_runs
        print(f"hyperparameter sweep with count {count}")
    sweep_id = wandb.sweep(sweep=sweep_config, project="eds_final", entity=args.wandb_entity)
    wandb.agent(sweep_id, function=cross_validate_smp_torch, count=count)

    wandb.finish()


if __name__ == "__main__":
    main()
