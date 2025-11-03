"""
Copyright (c) 2020, Zhengyang Feng
All rights reserved.

(BSD 3-Clause License)
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
“AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
OF THE POSSIBILITY OF SUCH DAMAGE.

MODIFIED by marlen123 to fit our data loading needs.
"""

# Recommend training a supervised baseline first,
# then conduct self-training from it to avoid mini-batch size issues

import os
import time
import copy
import torch
import argparse
import random
import json
import numpy as np
from tqdm import tqdm
from .utils.losses import DynamicMutualLoss
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler
from models.smp.build_rs_models import create_model_rs
from .utils.preprocess_helpers import get_preprocessing
from .utils.dmt.data import SemiSupervisedDataset
import warnings
from .utils.train_helpers import set_seed
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau

def generate_pseudo_labels(net, device, loader, num_classes, input_size, cbst_thresholds=None,
                           is_mixed_precision=False, label_ratio=0.2, label_stage_id="default"):
    # Generate pseudo labels and save to disk (negligible time compared with training)
    # Not very sure if there are any cache inconsistency issues (technically this should be fine)
    net.eval()

    # 1 forward pass (hard labels)
    if cbst_thresholds is None:  # Default
        cbst_thresholds = torch.tensor([0.99 for _ in range(num_classes)])
    cbst_thresholds = cbst_thresholds.to(device)
    net.eval()
    labeled_counts = 0
    ignored_counts = 0
    with torch.no_grad():
        for images, file_name_lists in tqdm(loader):
            images = images.to(device)
            with autocast(is_mixed_precision):
                outputs = net(images)

            # Generate pseudo labels (d1 x d2 x 2)
            for i in range(0, len(file_name_lists)):
                prediction = outputs[i]
                prediction = prediction.softmax(dim=0)  # ! softmax
                temp = prediction.max(dim=0)
                pseudo_label = temp.indices
                values = temp.values
                for j in range(num_classes):
                    pseudo_label[((pseudo_label == j) * (values < cbst_thresholds[j]))] = 255

                # N x d1 x d2 x 2 (pseudo labels | original confidences)
                pseudo_label = pseudo_label.unsqueeze(-1).float()
                pseudo_label = torch.cat([pseudo_label, values.unsqueeze(-1)], dim=-1)

                # Counting & Saving
                labeled_counts += (pseudo_label[:, :, 0] != 255).sum().item()
                ignored_counts += (pseudo_label[:, :, 0] == 255).sum().item()
                pseudo_label = pseudo_label.cpu().numpy()
                if ".png" in file_name_lists[i]:
                    filename = file_name_lists[i].split(".png")[0]
                elif ".npy" in file_name_lists[i]:
                    filename = file_name_lists[i].split(".npy")[0]
                else:
                    filename = file_name_lists[i]
                np.save(filename, pseudo_label)

                # additionally save over different stages

    # Return overall labeled ratio
    return labeled_counts / (labeled_counts + ignored_counts)

# Reimplemented (all converted to tensor ops) based on yzou2/CRST
def generate_class_balanced_pseudo_labels(net, device, loader, label_ratio, num_classes, input_size,
                                          down_sample_rate=16, buffer_size=100, is_mixed_precision=False,
                                          label_stage_id="default"):
    # Max memory usage surge ratio has an upper limit of 2x (caused by array concatenation).
    # Keep a fixed GPU buffer size to achieve a good enough speed-memory trade-off,
    # since casting to cpu is very slow.
    # Note that tensor.expand() does not allocate new memory,
    # and that Python's list consumes at least 3 times the memory that a typical array would've required,
    # though it is 3 times faster in concatenations, it is rather slow in sorting,
    # thus the overall time consumption is similar.
    # buffer_size: GPU buffer size, MB.
    # down_sample_rate: Pixel sample ratio, i.e. pick one pixel every #down_sample_rate pixels.
    net.eval()
    buffer_size = buffer_size * 1024 * 1024 / 12  # MB -> how many pixels

    # 1 forward pass (sample predicted probabilities,
    # sorting here is unnecessary since there is relatively negligible time-consumption to consider)
    pseudo_label = torch.tensor([], dtype=torch.int64, device=device)
    pseudo_probability = torch.tensor([], dtype=torch.float32, device=device)
    probabilities = [np.array([], dtype=np.float32) for _ in range(num_classes)]
    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(device)
            with autocast(is_mixed_precision):
                outputs = net(images)

            # Generate pseudo labels (d1 x d2) and reassemble
            for i in range(0, len(images)):
                prediction = outputs[i]
                temp = prediction.softmax(dim=0)  # ! softmax
                temp = temp.max(dim=0)
                pseudo_label = torch.cat([pseudo_label, temp.indices.flatten()[:: down_sample_rate]])
                pseudo_probability = torch.cat([pseudo_probability, temp.values.flatten()[:: down_sample_rate]])

            # Count and reallocate
            if pseudo_probability.shape[0] > buffer_size:
                for j in range(num_classes):
                    probabilities[j] = np.concatenate((probabilities[j],
                                                       pseudo_probability[pseudo_label == j].cpu().numpy()))
                pseudo_label = torch.tensor([], dtype=torch.int64, device=device)
                pseudo_probability = torch.tensor([], dtype=torch.float32, device=device)

        # Final count
        for j in range(num_classes):
            probabilities[j] = np.concatenate((probabilities[j],
                                               pseudo_probability[pseudo_label == j].cpu().numpy()))

    # Sort (n * log(n) << n * label_ratio, so just sort is good) and find kc
    print('Sorting...')
    kc = []
    for j in range(num_classes):
        if len(probabilities[j]) == 0:
            with open('exceptions.txt', 'a') as f:
                f.write(str(time.asctime()) + '--' + str(j) + '\n')

    for j in tqdm(range(num_classes)):
        probabilities[j].sort()
        if label_ratio >= 1:
            kc.append(probabilities[j][0])
        else:
            if len(probabilities[j]) * label_ratio < 1:
                kc.append(0.00001)
            else:
                kc.append(probabilities[j][-int(len(probabilities[j]) * label_ratio) - 1])
    del probabilities  # Better be safe than...

    print(kc)
    return generate_pseudo_labels(net=net, device=device, loader=loader, cbst_thresholds=torch.tensor(kc),
                                  input_size=input_size, num_classes=num_classes, is_mixed_precision=is_mixed_precision, 
                                  label_ratio=label_ratio, label_stage_id=label_stage_id)

# Save model checkpoints(supports amp)
def save_checkpoint(net, optimizer, lr_scheduler, is_mixed_precision, filename='temp.pt'):
    save_path = f'dmt_checkpoints_{args.pref}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, filename)
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        # 'amp': amp.state_dict() if is_mixed_precision else None
    }
    torch.save(checkpoint, filename)


# Load model checkpoints(supports amp)
def load_checkpoint(net, optimizer, lr_scheduler, is_mixed_precision, filename):
    checkpoint = torch.load(filename)
    try:
        net.load_state_dict(checkpoint['model'])
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        net.load_state_dict(checkpoint)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    # if is_mixed_precision and checkpoint['amp'] is not None:
    #     amp.load_state_dict(checkpoint['amp'])

# Copied and simplified from torch/vision/references/segmentation to compute mean IoU
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # For pseudo labels(which has 255), just don't let your network predict 255
            k = (a >= 0) & (a < n) & (b != 255)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu


def init(batch_size_labeled, batch_size_pseudo, state, split, sets_id):
    # Return data_loaders/data_loader
    # depending on whether the state is
    # 0: Pseudo labeling
    # 1: Semi-supervised training
    # 2: Fully-supervised training
    # 3: Just testing

    # For unlabeled set divisions
    split_u = split.replace('-r', '')
    split_u = split_u.replace('-l', '')

    test_dataset = SemiSupervisedDataset(
        cfg_model, 
        cfg_training, 
        preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), 
        im_size=480, 
        label_state=0, 
        image_set='test',
        image_dir="data/dmt/images/",)

    val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=batch_size_labeled + batch_size_pseudo,
                                             num_workers=0, 
                                             shuffle=False)


    # Testing
    if state == 3:
        return val_loader
    else:
        # Fully-supervised training
        if state == 2:
            labeled_set = SemiSupervisedDataset(
                            cfg_model, 
                            cfg_training,  
                            preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), 
                            im_size=480, 
                            label_state=0,
                            image_set=(str(split) + '_labeled_' + str(sets_id)),
                            image_dir="data/dmt/images/",
                            )
            labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set, 
                                                         batch_size=batch_size_labeled,
                                                         num_workers=0, 
                                                         shuffle=True)
            labeled_set2 = SemiSupervisedDataset(
                            cfg_model, 
                            cfg_training,  
                            preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), 
                            im_size=480, 
                            label_state=0,
                            image_set=(str(split) + '_labeled_' + str(sets_id)),
                            image_dir="data/dmt/images/",
                            )
            return labeled_loader, val_loader

        # Semi-supervised training
        elif state == 1:
            pseudo_labeled_set = SemiSupervisedDataset(
                                    cfg_model, 
                                    cfg_training, 
                                    preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), 
                                    im_size=480, 
                                    label_state=1, 
                                    image_set=(str(split_u) + '_unlabeled_' + str(sets_id)),
                                    mask_type=".npy")
            labeled_set = SemiSupervisedDataset(cfg_model, 
                                  cfg_training, 
                                  preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), 
                                  im_size=480, 
                                  label_state=0,
                                  image_set=(str(split) + '_labeled_' + str(sets_id)))
            pseudo_labeled_loader = torch.utils.data.DataLoader(dataset=pseudo_labeled_set,
                                                                batch_size=batch_size_pseudo,
                                                                num_workers=0, 
                                                                shuffle=True)
            labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set,
                                                         batch_size=batch_size_labeled,
                                                         num_workers=0, 
                                                         shuffle=True)
            return labeled_loader, pseudo_labeled_loader, val_loader

        else:
            # Labeling
            unlabeled_set = SemiSupervisedDataset(cfg_model, 
                                    cfg_training, 
                                    preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), 
                                    im_size=480, 
                                    label_state=2, 
                                    mask_type=".npy",
                                    image_set=(str(split_u) + '_unlabeled_' + str(sets_id)))
            unlabeled_loader = torch.utils.data.DataLoader(dataset=unlabeled_set, 
                                                           batch_size=batch_size_labeled,
                                                           num_workers=0,
                                                           shuffle=False)
            return unlabeled_loader


def train(loader_c, loader_sup, validation_loader, device, criterion, net, optimizer, lr_scheduler,
          num_epochs, is_mixed_precision, with_sup, num_classes, input_sizes,
          val_num_steps=1000, loss_freq=1, best_mp_iou=0):
    #######
    # c for carry (pseudo labeled), sup for support (labeled with ground truth) -_-
    # Don't ask me why
    #######
    # Poly training schedule
    # Epoch length measured by "carry" (c) loader
    # Batch ratio is determined by loaders' own batch size
    # Validate and find the best snapshot per val_num_steps
    loss_num_steps = int(len(loader_c) / loss_freq)
    net.train()
    epoch = 0
    if with_sup:
        iter_sup = iter(loader_sup)

    if is_mixed_precision:
        scaler = GradScaler()

    # Training
    running_stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'loss': 0.0}
    while epoch < num_epochs:
        conf_mat = ConfusionMatrix(num_classes)
        time_now = time.time()
        for i, data in enumerate(loader_c, 0):
            # Combine loaders (maybe just alternate training will work)
            if with_sup:
                inputs_c, labels_c = data
                inputs_sup, labels_sup = next(iter_sup, (0, 0))
                if type(inputs_sup) == type(labels_sup) == int:
                    iter_sup = iter(loader_sup)
                    inputs_sup, labels_sup = next(iter_sup, (0, 0))

                if labels_sup.ndim == 4:
                    labels_sup = labels_sup.squeeze(1)

                # Formatting (prob: label + max confidence, label: just label)
                float_labels_sup = labels_sup.clone().float().unsqueeze(1)
                probs_sup = torch.cat([float_labels_sup, torch.ones_like(float_labels_sup)], dim=1)
                probs_c = labels_c.clone()
                labels_c = labels_c[:, 0, :, :].long()

                # Concatenating
                inputs = torch.cat([inputs_c, inputs_sup])
                labels = torch.cat([labels_c, labels_sup])
                probs = torch.cat([probs_c, probs_sup])

                probs = probs.to(device)
            else:
                inputs, labels = data

            # Normal training
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast(is_mixed_precision):
                outputs = net(inputs)
                import pdb; pdb.set_trace()
                conf_mat.update(labels.flatten(), outputs.argmax(1).flatten())

                if with_sup:
                    loss, stats = criterion(outputs, probs, inputs_c.shape[0])
                else:
                    loss, stats = criterion(outputs, labels)

                    ########### TODO: REMOVE THIS ############
                    #from models.AutoSAM.loss_functions.dice_loss import SoftDiceLoss
                    #class_weights_np = np.array([2.57657597, 0.51193112, 1.51860234], dtype=np.float32)
                    #class_weights = torch.tensor(class_weights_np).float().to(device)

                    #dice_loss = SoftDiceLoss(
                    #    batch_dice=True, do_bg=False, rebalance_weights=class_weights_np
                    #)
                    #ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
                    #loss = ce_loss(outputs, labels.squeeze(1).long()) + dice_loss(F.softmax(outputs, dim=1), labels.squeeze(1).long())
                    ##########################################

            if is_mixed_precision:
                accelerator.backward(scaler.scale(loss))
                scaler.step(optimizer)
                scaler.update()
            else:
                accelerator.backward(loss)
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            # Logging
            for key in stats.keys():
                running_stats[key] += stats[key]
            current_step_num = int(epoch * len(loader_c) + i + 1)
            if current_step_num % loss_num_steps == (loss_num_steps - 1):
                for key in running_stats.keys():
                    print('[%d, %d] ' % (epoch + 1, i + 1) + key + ' : %.4f' % (running_stats[key] / loss_num_steps))
                    running_stats[key] = 0.0

            # Validate and find the best snapshot
            if current_step_num % val_num_steps == (val_num_steps - 1) or \
                current_step_num == num_epochs * len(loader_c) - 1:
                # Apex bug https://github.com/NVIDIA/apex/issues/706, fixed in PyTorch1.6, kept here for BC
                metrics = test_one_set(loader=validation_loader, device=device, net=net,
                                                              num_classes=num_classes,
                                                              output_size=input_sizes[2],
                                                              is_mixed_precision=is_mixed_precision)
                mp_iou = metrics[1]

                net.train()

                # Record best model (Straight to disk)
                if mp_iou > best_mp_iou:
                    best_mp_iou = mp_iou
                    print('New best model (mean IoU: %.2f, mp IoU: %.2f), saving...' % (metrics[0] * 100, mp_iou * 100))
                    save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                    is_mixed_precision=is_mixed_precision)
                    
        # Evaluate training accuracies(same metric as validation, but must be on-the-fly to save time)
        acc_global, acc, iu = conf_mat.compute()
        print((
            'global correct: {:.2f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}').format(
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100))

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))

    return metrics


# Copied and modified from torch/vision/references/segmentation
def test_one_set(loader, device, net, num_classes, output_size, is_mixed_precision):
    # Evaluate on 1 data_loader (cudnn impact < 0.003%)
    net.eval()

    # our metrics (to report)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    pred_coll = []
    prob_coll = []
    label_coll = []
    jac_list_mp = []
    jac_list_si = []
    jac_list_oc = []
    jac_mean = []

    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            with autocast(is_mixed_precision):
                mask = net(image)

            b, _, h, w = image.shape

            mask = mask.view(b, -1, h, w)

            assert mask.shape[1] == 3
            pred_softmax = F.softmax(mask, dim=1)
            prob_coll.append(pred_softmax.cpu())
            label_coll.append(target.cpu().squeeze(1))

            pred = torch.argmax(mask, dim=1)
            pred_coll.append(pred.cpu())

            jaccard = JaccardIndex(
                task="multiclass", num_classes=cfg_model["num_classes"], average=None
            ).to(device)
            jaccard_mean = JaccardIndex(
                task="multiclass", num_classes=cfg_model["num_classes"]
            ).to(device)

            jac = jaccard(pred_softmax, target.squeeze(1))
            jac_m = jaccard_mean(pred_softmax, target.squeeze(1))

            jac_list_mp.append(jac[0].item())
            jac_list_si.append(jac[1].item())
            jac_list_oc.append(jac[2].item())
            jac_mean.append(jac_m.item())

            # compute confusion stats per class
            for c in range(cfg_model["num_classes"]):
                tp[c] += torch.sum((pred == c) & (target.squeeze(1) == c)).item()
                fp[c] += torch.sum((pred == c) & (target.squeeze(1) != c)).item()
                fn[c] += torch.sum((pred != c) & (target.squeeze(1) == c)).item()

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

    precision_mp = precision[0]
    precision_si = precision[1]
    precision_oc = precision[2]
    recall_mp = recall[0]
    recall_si = recall[1]
    recall_oc = recall[2]
    roc_auc_mp = roc_auc_scores[0]
    roc_auc_si = roc_auc_scores[1]
    roc_auc_oc = roc_auc_scores[2]

    print("Melt pond IoU: {:.4f}, Sea ice IoU: {:.4f}, Open water IoU: {:.4f}, Mean IoU: {:.4f}".format(
        np.mean(jac_list_mp), np.mean(jac_list_si), np.mean(jac_list_oc), np.mean(jac_mean)
    ))

    return np.mean(jac_mean), np.mean(jac_list_mp), np.mean(jac_list_oc), np.mean(jac_list_si), precision_mp, precision_si, precision_oc, recall_mp, recall_si, recall_oc, precision_macro, recall_macro, roc_auc_mp, roc_auc_si, roc_auc_oc


def after_loading():
    global lr_scheduler

    # The "poly" policy, variable names are confusing(May need reimplementation)
    if not args.labeling:
        if args.state == 2:
            len_loader = (len(labeled_loader) * args.epochs)
        else:
            len_loader = (len(pseudo_labeled_loader) * args.epochs)
        lr_scheduler = None
        #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / len_loader) ** 0.9)
        #lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=20)

    # Resume training?
    if args.continue_from is not None:
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.6.0 && torchvision 0.7.0')
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--gamma1', type=float, default=0,
                        help='Gamma for entropy minimization in agreement (default: 0)')
    parser.add_argument('--gamma2', type=float, default=0,
                        help='Gamma for learning in disagreement (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--val-num-steps', type=int, default=500,
                        help='How many steps between validations (default: 500)')
    parser.add_argument('--label_ratio', type=float, default=0.2,
                        help='Initial labeling ratio (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate (default: 0.002)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay for Optimizer (default: 0)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for the fully-supervised initialization (default: 30)')
    parser.add_argument('--batch-size-labeled', type=int, default=1,
                        help='Batch size for labeled data (default: 1)')
    parser.add_argument('--batch-size-pseudo', type=int, default=7,
                        help='Batch size for pseudo labeled data (default: 7)')
    parser.add_argument('--do-not-save', action='store_false', default=True,
                        help='Save model (default: True)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--labeling', action='store_true', default=False,
                        help='Just pseudo labeling (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Self-training begins from a previous checkpoint/Test on this')
    parser.add_argument('--train-set', type=str, default='1',
                        help='e.g. 1:7(8), 1:3(4), 1:1(2), 1:0(1) labeled/unlabeled split (default: 1)')
    parser.add_argument('--sets_id', type=int, default=0,
                        help='Different random splits(0/1/2) (default: 0)')
    parser.add_argument('--state', type=int, default=1,
                        help="Final test(3)/Fully-supervised training(2)/Semi-supervised training(1)")
    parser.add_argument('--path_to_config', type=str, default="configs/unetplusplus/aid.json")
    parser.add_argument('--aid', action='store_true', default=False,
                        help='Use AID pre-trained backbone (default: False)')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus',
                        help='Architecture (default: UnetPlusPlus)')
    parser.add_argument('--pref', type=str, default='28103',
                        help='Prefix for checkpoint saving (default: 28103)')
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.path_to_config) as f:
        config = json.load(f)

    cfg_model = config['model']
    cfg_training = config['training']

    accelerator = Accelerator(split_batches=True)
    device = accelerator.device

    num_classes = 3
    input_sizes = [(480, 480), (480, 480), (480, 480)] # adjusted from DMT: training input min, training input max, testing input

    if args.aid:
        print("Running with aid checkpoint")
        net = create_model_rs(
            arch=args.arch,
            encoder_name=cfg_model["backbone"],
            pretrain="aid",
            in_channels=3,
            classes=cfg_model["num_classes"],
        )
    else:
        print("Running with imagenet checkpoint")
        net = smp.create_model(
            arch=args.arch,
            encoder_name=cfg_model["backbone"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=cfg_model["num_classes"],
        )
    print(device)
    net.to(device)

    label_stage_id = f"{args.aid}_{args.label_ratio}"

    # Define optimizer
    # Use different learning rates if you want, we do not observe improvement from different learning rates
    params_to_optimize = [
        {"params": [p for p in net.parameters() if p.requires_grad]},
    ]

    #optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    # Just to be safe (a little bit more memory, by all means, save it to disk if you want)
    if args.state == 1:
        st_optimizer_init = copy.deepcopy(optimizer.state_dict())

    # Testing
    if args.state == 3:
        net, optimizer = accelerator.prepare(net, optimizer)
        test_loader = init(batch_size_labeled=args.batch_size_labeled, batch_size_pseudo=args.batch_size_pseudo,
                           state=3, split=args.train_set, sets_id=args.sets_id)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        test_one_set(loader=test_loader, device=device, net=net, num_classes=num_classes,
                     output_size=input_sizes[2], is_mixed_precision=args.mixed_precision)
    else:
        criterion = DynamicMutualLoss(gamma1=args.gamma1, gamma2=args.gamma2, ignore_index=255)

        # Only fully-supervised training
        if args.state == 2:
            labeled_loader, val_loader = init(batch_size_labeled=args.batch_size_labeled,
                                              batch_size_pseudo=args.batch_size_pseudo, sets_id=args.sets_id,
                                              state=2, split=args.train_set)
            after_loading()
            net, optimizer, labeled_loader = accelerator.prepare(net, optimizer, labeled_loader)
            metrics = train(loader_c=labeled_loader, loader_sup=None, validation_loader=val_loader,
                      device=device, criterion=criterion, net=net, optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      num_epochs=args.epochs, num_classes=num_classes,
                      is_mixed_precision=args.mixed_precision, with_sup=False,
                      val_num_steps=args.val_num_steps, input_sizes=input_sizes)

        # Self-training
        elif args.state == 1:
            if args.labeling:
                unlabeled_loader = init(
                    batch_size_labeled=args.batch_size_labeled, 
                    batch_size_pseudo=args.batch_size_pseudo,
                    state=0, 
                    split=args.train_set, 
                    sets_id=args.sets_id)
                after_loading()
                net, optimizer = accelerator.prepare(net, optimizer)
                time_now = time.time()
                ratio = generate_class_balanced_pseudo_labels(net=net, device=device, loader=unlabeled_loader,
                                                              input_size=input_sizes[2],
                                                              label_ratio=args.label_ratio, num_classes=num_classes,
                                                              is_mixed_precision=args.mixed_precision, label_stage_id=label_stage_id)
                print(ratio)
                print('Pseudo labeling time: %.2fs' % (time.time() - time_now))
            else:
                labeled_loader, pseudo_labeled_loader, val_loader = init(
                    batch_size_labeled=args.batch_size_labeled, 
                    batch_size_pseudo=args.batch_size_pseudo,
                    state=1, 
                    split=args.train_set, 
                    sets_id=args.sets_id)
                after_loading()
                net, optimizer, labeled_loader, pseudo_labeled_loader = accelerator.prepare(net, optimizer,
                                                                                            labeled_loader,
                                                                                            pseudo_labeled_loader)
                # NOTE: removed lr_scheduler=lr_scheduler
                metrics = train(loader_c=pseudo_labeled_loader, loader_sup=labeled_loader,
                          validation_loader=val_loader, lr_scheduler=lr_scheduler,
                          device=device, criterion=criterion, net=net, optimizer=optimizer,
                          num_epochs=args.epochs, num_classes=num_classes,
                          is_mixed_precision=args.mixed_precision, with_sup=True,
                          val_num_steps=args.val_num_steps, input_sizes=input_sizes)

        else:
            # Support unsupervised learning here if that's what you want
            # But we do not think that works, yet...
            raise ValueError

        if not args.labeling:
            # --do-not-save => args.do_not_save = False
            if args.do_not_save:  # Rename the checkpoint
                os.rename(f'dmt_checkpoints_{args.pref}/temp.pt', f'dmt_checkpoints_{args.pref}/{args.exp_name}.pt')
            else:  # Since the checkpoint is already saved, it should be deleted
                os.remove(f'dmt_checkpoints_{args.pref}/temp.pt')

            with open(f'dmt_checkpoints_{args.pref}/log.txt', 'a') as f:
                f.write(args.exp_name + ': ' + ', '.join(str(metric) for metric in metrics) + '\n')
