import torch
import torch.nn.functional as F

class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, *input):
        raise NotImplementedError


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(reduction)
        self.register_buffer('weight', weight)

    def forward(self, *input):
        raise NotImplementedError

# Credits to: https://github.com/voldemortX/DST-CBC/blob/master/segmentation/utils/losses.py 
# Dynamic loss for DMT (Dynamic Mutual-Training): Weights depend on confidence
class DynamicMutualLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma1=0, gamma2=0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        # < split_index are pseudo labeled samples
        # Cross-entropy loss for all samples
        # Targets could be soft (for DMT) or hard (for fully-supervised training)
        if targets.dim() == 4:
            real_targets = targets[:, 0, :, :].long()
        else:
            real_targets = targets
        total_loss = self.criterion_ce(inputs, real_targets)  # Pixel-level weight is always 1
        stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0,
                 'loss': total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()}

        if split_index is None or (self.gamma1 == 0 and self.gamma2 == 0):  # No dynamic loss
            total_loss = total_loss.sum() / (real_targets != self.ignore_index).sum()
        else:  # Dynamic loss
            outputs = inputs.softmax(dim=1).clone().detach()
            decision_current = outputs.argmax(1)  # N
            decision_pseudo = real_targets.clone().detach()  # N
            confidence_current = outputs.max(1).values  # N

            temp = real_targets.unsqueeze(1).clone().detach()
            temp[temp == self.ignore_index] = 0
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)

            confidence_pseudo = targets[:, 1, :, :].clone().detach()  # N
            dynamic_weights = torch.ones_like(decision_current).float()

            # Prepare indices
            disagreement = decision_current != decision_pseudo
            current_win = confidence_current > confidence_pseudo
            stats['disagree'] = (disagreement * (real_targets != self.ignore_index))[:split_index].sum().int().item()
            stats['current_win'] = ((disagreement * current_win) * (real_targets != self.ignore_index))[:split_index] \
                .sum().int().item()

            # Agree
            indices = ~disagreement
            # dynamic_weights[indices] = (confidence_current[indices] * confidence_pseudo[indices]) ** self.gamma1
            # dynamic_weights[indices] = (probabilities_current[indices] ** self.gamma2) * (confidence_pseudo[indices] ** self.gamma1)
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma1

            # Disagree (current model wins, do not learn!)
            indices = disagreement * current_win
            dynamic_weights[indices] = 0

            # Disagree
            indices = disagreement * ~current_win
            # dynamic_weights[indices] = ((1 - confidence_current[indices]) * confidence_pseudo[indices]) ** self.gamma2
            # dynamic_weights[indices] = (probabilities_current[indices] ** self.gamma2) * (confidence_pseudo[indices] ** self.gamma1)
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma2

            # Weight loss
            dynamic_weights[split_index:] = 1
            stats['avg_weights'] = dynamic_weights[real_targets != self.ignore_index].mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()

        return total_loss, stats


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=[0.25, 0.25, 0.25], gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.alpha is not None and isinstance(self.alpha, (list, torch.Tensor)):
            if isinstance(self.alpha, list):
                self.alpha = torch.Tensor(self.alpha)

    def focal_loss_multiclass(self, inputs, targets, num_classes, distance_map=None):
        """ Focal loss for multi-class classification. """

        # this is only true for our case
        assert num_classes == 3

        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
        
        # convert distance map to tensor if not None
        if distance_map is not None and not isinstance(distance_map, torch.Tensor):
            distance_map = torch.tensor(distance_map, dtype=torch.float32).to(inputs.device)

        # Convert logits to probabilities with softmax
        assert inputs.shape[-1] == num_classes
        probs = F.softmax(inputs, dim=-1)
        log_probs = F.log_softmax(inputs, dim=-1)   # numerically stable log-softmax

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute cross-entropy for each class
        #ce_loss_old = -targets_one_hot * torch.log(probs)
        ce_loss = -targets_one_hot * log_probs

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=-1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            # we need to handle that alpha has shape (C,) and ce_loss has shape (B, H, W, C)
            alpha_t = self.alpha[targets]
            if distance_map is not None:
                print("Using label uncertainty with focal loss.")
                # Apply distance map to alpha
                alpha_t = alpha_t * distance_map

            ce_loss = alpha_t.unsqueeze(-1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(-1) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def forward(self, inputs, targets, pixel_distances=None):

        num_classes = inputs.shape[1]

        # we need to make sure to have inputs of shape (B, H, W, C)
        if inputs.shape[-1] != num_classes:
            inputs = inputs.permute(0, 2, 3, 1)

        return self.focal_loss_multiclass(inputs, targets, num_classes=num_classes, distance_map=pixel_distances)