"""
Sparse R-CNN builder (ResNet-50-FPN, 100 learnable proposals, 6 dynamic stages), Image task.

Paper: Sun et al., "Sparse R-CNN: End-to-End Object Detection with Learnable Proposals" (CVPR 2021, arXiv 2011.12450).
Code : official PeizeSun/SparseR-CNN (detectron2 project): projects/SparseRCNN/sparsercnn/{detector.py, head.py,
       loss.py, util/box_ops.py}, config sparsercnn.res50.100pro.3x (NUM_PROPOSALS 100, HIDDEN_DIM 256, NHEADS 8,
       DIM_FEEDFORWARD 2048, NUM_HEADS 6, NUM_CLS 1, NUM_REG 3, DIM_DYNAMIC 64, NUM_DYNAMIC 2, DROPOUT 0.0,
       focal loss 0.25 / 2, CLASS 2.0, L1 5.0, GIOU 2.0, deep supervision, Hungarian matching with the same weights,
       bbox_weights (2, 2, 1, 1), proposal boxes initialised to the whole image). detectron2 ROIPooler -> torchvision
       MultiScaleRoIAlign (same canonical level mapping); backbone = torchvision ResNet-50-FPN as detectors/faster_rcnn.

Raw contract: the 100 final boxes per image with sigmoid class probabilities (objectness 1). The paper uses no NMS
(top-100); the common post-processing still applies its NMS ([Unified]).

[Unified] no ImageNet weights, trainable BN + PreciseBN, common init (conv weights / BN; Linear layers keep the
official xavier / prior init, proposal embeddings keep their init), no mean / std, 608 input, Adam 1e-4 instead of
AdamW 2.5e-5 with gradient clipping, common augmentation and post-processing.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign, generalized_box_iou, sigmoid_focal_loss

from detectors.anchor_free.common import HeadStats, split_targets

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


class DynamicConv(nn.Module):
    def __init__(self, hidden_dim=256, dim_dynamic=64, num_dynamic=2, pooler_resolution=7):
        super().__init__()
        self.hidden_dim, self.dim_dynamic, self.num_dynamic = hidden_dim, dim_dynamic, num_dynamic
        self.num_params = hidden_dim * dim_dynamic
        self.dynamic_layer = nn.Linear(hidden_dim, num_dynamic * self.num_params)
        self.norm1, self.norm2 = nn.LayerNorm(dim_dynamic), nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_dim * pooler_resolution ** 2, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, pro_features, roi_features):
        """pro_features (1, N*B, d), roi_features (49, N*B, d)"""
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)
        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)
        features = self.activation(self.norm1(torch.bmm(features, param1)))
        features = self.activation(self.norm2(torch.bmm(features, param2)))
        return self.activation(self.norm3(self.out_layer(features.flatten(1))))


class RCNNHead(nn.Module):
    def __init__(self, num_classes, d_model=256, dim_feedforward=2048, nhead=8, dropout=0.0, num_cls=1, num_reg=3,
                 bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(d_model)
        self.linear1, self.linear2 = nn.Linear(d_model, dim_feedforward), nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        self.cls_module = nn.ModuleList(sum([[nn.Linear(d_model, d_model, False), nn.LayerNorm(d_model),
                                              nn.ReLU(inplace=True)] for _ in range(num_cls)], []))
        self.reg_module = nn.ModuleList(sum([[nn.Linear(d_model, d_model, False), nn.LayerNorm(d_model),
                                              nn.ReLU(inplace=True)] for _ in range(num_reg)], []))
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, image_size):
        N, nr_boxes = bboxes.shape[:2]
        roi_features = pooler(features, list(bboxes), [image_size] * N)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = self.norm1(pro_features + self.dropout1(pro_features2))
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes,
                                                                                             self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        obj_features = self.norm2(pro_features + self.dropout2(pro_features2))
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = self.norm3(obj_features + self.dropout3(obj_features2))
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature, reg_feature = fc_feature.clone(), fc_feature.clone()
        for m in self.cls_module:
            cls_feature = m(cls_feature)
        for m in self.reg_module:
            reg_feature = m(reg_feature)
        class_logits = self.class_logits(cls_feature)
        pred_bboxes = self.apply_deltas(self.bboxes_delta(reg_feature), bboxes.view(-1, 4))
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        boxes = boxes.to(deltas.dtype)
        widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        ctr_x, ctr_y = boxes[:, 0] + 0.5 * widths, boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = self.bbox_weights
        dx, dy = deltas[:, 0::4] / wx, deltas[:, 1::4] / wy
        dw = torch.clamp(deltas[:, 2::4] / ww, max=_DEFAULT_SCALE_CLAMP)
        dh = torch.clamp(deltas[:, 3::4] / wh, max=_DEFAULT_SCALE_CLAMP)
        pcx, pcy = dx * widths[:, None] + ctr_x[:, None], dy * heights[:, None] + ctr_y[:, None]
        pw, ph = torch.exp(dw) * widths[:, None], torch.exp(dh) * heights[:, None]
        return torch.cat([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], -1)


class SparseRCNN(nn.Module):
    def __init__(self, num_classes, num_proposals=100, hidden_dim=256, num_heads=6, prior_prob=0.01):
        super().__init__()
        self.num_classes, self.num_proposals = num_classes, num_proposals
        self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None, norm_layer=nn.BatchNorm2d,
                                            trainable_layers=5)
        self.pooler = MultiScaleRoIAlign(["0", "1", "2", "3"], 7, sampling_ratio=2)
        self.init_proposal_features = nn.Embedding(num_proposals, hidden_dim)
        self.init_proposal_boxes = nn.Embedding(num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        self.head_series = nn.ModuleList([RCNNHead(num_classes) for _ in range(num_heads)])
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for p in self.head_series.parameters():                         # DynamicHead._reset_parameters
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for h in self.head_series:
            nn.init.constant_(h.class_logits.bias, bias_value)
        self.weight = dict(cls=2.0, l1=5.0, giou=2.0)
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def heads(self, x):
        N, _, H, W = x.shape
        feats = self.backbone(x)
        features = {k: feats[k] for k in ["0", "1", "2", "3"]}
        boxes = box_cxcywh_to_xyxy(self.init_proposal_boxes.weight.clone())[None] * x.new_tensor([W, H, W, H])
        boxes = boxes.repeat(N, 1, 1)
        pro = self.init_proposal_features.weight[None].repeat(1, N, 1).clone()
        outs = []
        for head in self.head_series:
            class_logits, pred_bboxes, pro = head(features, boxes, pro, self.pooler, (H, W))
            outs.append((class_logits, pred_bboxes))
            boxes = pred_bboxes.detach()
        return outs

    @torch.no_grad()
    def match(self, logits, boxes, tgt, img_wh):
        """HungarianMatcher (focal class cost, L1 on normalised xyxy, GIoU)"""
        out_prob = logits.sigmoid()
        idx = []
        for b, t in enumerate(tgt):
            if len(t["labels"]) == 0:
                idx.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
                continue
            p = out_prob[b][:, t["labels"]]
            alpha, gamma = 0.25, 2.0
            neg = (1 - alpha) * (p ** gamma) * (-(1 - p + 1e-8).log())
            pos = alpha * ((1 - p) ** gamma) * (-(p + 1e-8).log())
            cost_class = pos - neg
            cost_bbox = torch.cdist(boxes[b] / img_wh, t["boxes"] / img_wh, p=1)
            cost_giou = -generalized_box_iou(boxes[b], t["boxes"])
            C = (self.weight["l1"] * cost_bbox + self.weight["cls"] * cost_class + self.weight["giou"] * cost_giou)
            i, j = linear_sum_assignment(C.cpu())
            idx.append((torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)))
        return idx

    def loss_stage(self, logits, boxes, tgt, img_wh, num_boxes):
        idx = self.match(logits, boxes, tgt, img_wh)
        target_classes = torch.zeros_like(logits)
        src_b, tgt_b = [], []
        for b, (i, j) in enumerate(idx):
            target_classes[b, i, tgt[b]["labels"][j]] = 1.0
            src_b.append(boxes[b, i])
            tgt_b.append(tgt[b]["boxes"][j])
        loss_ce = sigmoid_focal_loss(logits.flatten(0, 1), target_classes.flatten(0, 1), 0.25, 2.0, "sum") / num_boxes
        src_b, tgt_b = torch.cat(src_b), torch.cat(tgt_b)
        if len(src_b):
            loss_giou = (1 - torch.diag(generalized_box_iou(src_b, tgt_b))).sum() / num_boxes
            loss_l1 = F.l1_loss(src_b / img_wh, tgt_b / img_wh, reduction="none").sum() / num_boxes
        else:
            loss_giou = loss_l1 = boxes.sum() * 0
        return self.weight["cls"] * loss_ce, self.weight["l1"] * loss_l1, self.weight["giou"] * loss_giou, idx

    def forward(self, x, targets=None):
        B, C = x.shape[0], self.num_classes
        if targets is None and self.training:                           # PreciseBN: BN only in the backbone
            self.backbone(x)
            return x.new_zeros(B, 0, 5 + C)
        outs = self.heads(x)
        logits, boxes = outs[-1]
        cxcy = (boxes[..., :2] + boxes[..., 2:]) / 2
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
        raw = torch.cat([cxcy, wh, torch.ones_like(logits[..., :1]), logits.sigmoid()], -1)
        if targets is None:
            return raw
        tgt = split_targets(targets, B, x.shape[2])
        img_wh = x.new_tensor([x.shape[3], x.shape[2], x.shape[3], x.shape[2]])
        num_boxes = max(float(sum(len(t["labels"]) for t in tgt)), 1.0)
        loss_dict = {}
        for s, (lg, bx) in enumerate(outs):                              # deep supervision
            lc, l1, lg_, idx = self.loss_stage(lg, bx, tgt, img_wh, num_boxes)
            loss_dict.update({f"loss_ce_{s}": lc, f"loss_bbox_{s}": l1, f"loss_giou_{s}": lg_})
        loss = sum(loss_dict.values())
        fg = torch.zeros(B, self.num_proposals, dtype=torch.bool, device=x.device)
        fg_lab = torch.full((B, self.num_proposals), -1, dtype=torch.long, device=x.device)
        for b, (i, j) in enumerate(idx):
            fg[b, i] = True
            fg_lab[b, i] = tgt[b]["labels"][j]
        lf = logits.detach().reshape(-1, C)
        self.stats.update({"losses": loss, "loss_ce": loss_dict["loss_ce_5"], "loss_giou": loss_dict["loss_giou_5"],
                           "num_pos": fg.float().sum() / B}, lf.sigmoid().max(-1).values, lf, fg.view(-1),
                          fg_lab.view(-1)[fg.view(-1)])
        return loss, raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("Sparse R-CNN has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    return SparseRCNN(num_classes)
