import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        if ('ENNABLE_MULTIVIEW' in losses_cfg):
            self.add_module(
                'consis_cls_loss_func',
                loss_utils.ConsisSigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
            )
            self.add_module(
                'consis_reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.CONSIS_LOSS_WEIGHTS.get('code_weights', None))
            )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def assign_targets_multiview(self, gt_boxes, mv_num):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets_multiview(
            self.anchors, gt_boxes, mv_num
        )
        return targets_dict

    def assign_common_targets(self, data_dict, targets_dict, mv_num):
        batch_size = data_dict['batch_size']
        # common_bev8x_hash_mask = data_dict['common_bev8x_hash_mask'].view(batch_size, -1)
        batch_mv_cls_preds = data_dict['batch_mv_cls_preds']
        batch_mv_box_preds = data_dict['batch_mv_box_preds']
        cls_labels = targets_dict['mv_box_cls_labels'][0]

        # print('common_bev8x_hash_mask_size:', common_bev8x_hash_mask.shape)  # [b, 35200]
        # print('common_bev8x_hash_mask0_sum:', common_bev8x_hash_mask[0].sum())  # 1448
        # print('common_bev8x_hash_mask1_sum:', common_bev8x_hash_mask[1].sum())  # 1210
        # print('batch_mv_cls_preds0_size:', batch_mv_cls_preds[0].shape)  # [b, 211200, 3]
        # print('batch_mv_box_preds0_size:', batch_mv_box_preds[0].shape)  # [b, 211200, 7]
        # print('cls_labels_size:', cls_labels.shape)  # [b, 211200]
        # print('cls_labels0_sum:', (cls_labels[0] > 0).sum())  #
        # print('cls_labels1_sum:', (cls_labels[1] > 0).sum())  #

        # common_bev8x_mask = []
        # for i in range(batch_size):
        #     mid_common_bev8x_mask = []
        #     for j in range(6):
        #         mid_common_bev8x_mask.append(common_bev8x_hash_mask[i])
        #     common_bev8x_mask.append(torch.cat(mid_common_bev8x_mask, dim=0))
        #     # print('common_bev8x_mask_size:', common_bev8x_mask[i].shape)  # [211200]
        # common_bev8x_mask = torch.stack(common_bev8x_mask, dim=0)
        # # print('common_bev8x_mask_size:', common_bev8x_mask.shape)  # [b, 211200]
        # common_bev8x_flag = []
        # for i in range(batch_size):
        #     mid_common_bev8x_flag = (common_bev8x_mask[i] == 1) & (cls_labels[i] > 0)
        #     print('mid_common_bev8x_flag_sum:', mid_common_bev8x_flag.sum())  #
        #     # mid_common_bev8x_mask2 = common_bev8x_mask[i][cls_labels[i] > 0]
        #     # print('mid_common_bev8x_mask2_size:', mid_common_bev8x_mask2.shape)  #
        #     # for j in range(len(mid_common_bev8x_mask2)):
        #     #     print('mid_common_bev8x_mask2 ', j, ' ', mid_common_bev8x_mask2[j])
        # common_batch_mv_cls_preds = []
        # common_batch_mv_box_preds = []
        # for i in range(mv_num):
        #     batch_cls_preds = batch_mv_cls_preds[i].view(-1, 6, 3)
        #     batch_box_preds = batch_mv_box_preds[i].view(-1, 6, 7)
        #     common_batch_cls_preds = batch_cls_preds[common_bev8x_flag]
        #     common_batch_box_preds = batch_box_preds[common_bev8x_flag]
        #     common_batch_mv_cls_preds.append(common_batch_cls_preds)
        #     common_batch_mv_box_preds.append(common_batch_box_preds)
            # print('batch_cls_preds_size:', batch_cls_preds.shape) # [70400, 6, 3]
            # print('batch_box_preds_size:', batch_box_preds.shape) # [70400, 6, 7]
            # print('common_batch_cls_preds_size:', common_batch_cls_preds.shape)  # [3050, 6, 3]
            # print('common_batch_box_preds_size:', common_batch_box_preds.shape)  # [3050, 6, 7]

        # consis_a_cls_preds = common_batch_mv_cls_preds[1].view(-1, 3)
        # consis_a_box_preds = common_batch_mv_box_preds[1].view(-1, 7)
        # consis_a_cls_labels = common_batch_mv_cls_preds[0].view(-1, 3)
        # consis_a_box_labels = common_batch_mv_box_preds[0].view(-1, 7)
        # if mv_num == 3:
        #     consis_a2_cls_preds = common_batch_mv_cls_preds[2].view(-1, 3)
        #     consis_a2_box_preds = common_batch_mv_box_preds[2].view(-1, 7)
        #     consis_a2_cls_labels = common_batch_mv_cls_preds[0].view(-1, 3)
        #     consis_a2_box_labels = common_batch_mv_box_preds[0].view(-1, 7)

        common_batch_mv_cls_preds = []
        common_batch_mv_box_preds = []
        for i in range(mv_num):
            batch_cls_preds = batch_mv_cls_preds[i]
            batch_box_preds = batch_mv_box_preds[i]
            common_batch_cls_preds = []
            common_batch_box_preds = []
            for j in range(batch_size):
                cls_label_flag = (cls_labels[j] > 0)
                common_cls_preds = batch_cls_preds[j][cls_label_flag]
                common_box_preds = batch_box_preds[j][cls_label_flag]
                common_batch_cls_preds.append(common_cls_preds)
                common_batch_box_preds.append(common_box_preds)
            common_batch_cls_preds = torch.cat(common_batch_cls_preds)
            common_batch_box_preds = torch.cat(common_batch_box_preds)
            # print('common_batch_cls_preds:', common_batch_cls_preds.shape)
            # print('common_batch_box_preds:', common_batch_box_preds.shape)
            common_batch_mv_cls_preds.append(common_batch_cls_preds)
            common_batch_mv_box_preds.append(common_batch_box_preds)

        consis_a_cls_preds = common_batch_mv_cls_preds[1]
        consis_a_box_preds = common_batch_mv_box_preds[1]
        consis_a_cls_labels = common_batch_mv_cls_preds[0]
        consis_a_box_labels = common_batch_mv_box_preds[0]
        if mv_num == 3:
            consis_a2_cls_preds = common_batch_mv_cls_preds[2]
            consis_a2_box_preds = common_batch_mv_box_preds[2]
            consis_a2_cls_labels = common_batch_mv_cls_preds[0]
            consis_a2_box_labels = common_batch_mv_box_preds[0]

        # print('consis_a_cls_preds_size:', consis_a_cls_preds.shape) # [18300, 3]
        # print('consis_a_box_preds_size:', consis_a_box_preds.shape) # [18300, 7]
        # print('consis_a_cls_labels_size:', consis_a_cls_labels.shape) # [18300, 3]
        # print('consis_a_box_labels_size:', consis_a_box_labels.shape) # [18300, 7]
        # print('consis_a2_cls_preds_size:', consis_a2_cls_preds.shape) # [18300, 3]
        # print('consis_a2_box_preds_size:', consis_a2_box_preds.shape) # [18300, 7]
        # print('consis_a2_cls_labels_size:', consis_a2_cls_labels.shape) # [18300, 3]
        # print('consis_a2_box_labels_size:', consis_a2_box_labels.shape) # [18300, 7]

        # targets_dict
        if mv_num == 3:
            targets_dict['consis_cls_preds'] = [consis_a_cls_preds, consis_a2_cls_preds]
            targets_dict['consis_box_preds'] = [consis_a_box_preds, consis_a2_box_preds]
            targets_dict['consis_cls_labels'] = [consis_a_cls_labels, consis_a2_cls_labels]
            targets_dict['consis_box_labels'] = [consis_a_box_labels, consis_a2_box_labels]
        elif mv_num == 2:
            targets_dict['consis_cls_preds'] = [consis_a_cls_preds]
            targets_dict['consis_box_preds'] = [consis_a_box_preds]
            targets_dict['consis_cls_labels'] = [consis_a_cls_labels]
            targets_dict['consis_box_labels'] = [consis_a_box_labels]
        # targets_dict['common_sum'] = common_bev8x_hash_mask.sum()
        targets_dict['common_sum'] = common_batch_mv_cls_preds[0].shape[0]
        # print('common_sum:', targets_dict['common_sum'])
        # print('stop:', data_dict['stop'])
        return targets_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        # print('@@@@@@@@@ cls_preds_size:', cls_preds.shape)  # [2, 200, 176, 18]
        # print('@@@@@@@@@ box_cls_labels_size:', box_cls_labels.shape)  # [2, 211200]

        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        # print('@@@@@@@@@ cls_preds_size:', cls_preds.shape) # [b, 211200, 3]
        # print('@@@@@@@@@ one_hot_targets_size:', one_hot_targets.shape) # [b, 211200, 3]
        # print('@@@@@@@@@ cls_loss_src_size:', cls_loss_src.size()) # [b, 211200, 3]

        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def get_cls_layer_loss_multiview(self, mv_status):
        if mv_status == 1:
            circle_num = 3
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight'] = 0.33
        elif mv_status == 2 or mv_status == 3:
            circle_num = 2
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight'] = 0.5
        mv_cls_preds = self.forward_ret_dict['mv_cls_preds']
        mv_box_cls_labels = self.forward_ret_dict['mv_box_cls_labels']

        mv_cls_loss = []
        for i in range(circle_num):
            cls_preds = mv_cls_preds[i]
            box_cls_labels = mv_box_cls_labels[i]
            # print('cls_preds_size:', cls_preds.shape)  # [2, 200, 176, 18]
            # print('box_cls_labels_size:', box_cls_labels.shape)  # [2, 211200]
            # print('cls_view_id:', i)
            batch_size = int(cls_preds.shape[0])
            cared = box_cls_labels >= 0  # [N, num_anchors]
            positives = box_cls_labels > 0
            negatives = box_cls_labels == 0
            negative_cls_weights = negatives * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives.float()
            if self.num_class == 1:
                # class agnostic
                box_cls_labels[positives] = 1
            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
            cls_targets = cls_targets.unsqueeze(dim=-1)
            cls_targets = cls_targets.squeeze(dim=-1)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
            )
            one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
            one_hot_targets = one_hot_targets[..., 1:]
            cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
            # print('cls_preds_size:', cls_preds.shape) # [b, 211200, 3]
            # print('one_hot_targets_size:', one_hot_targets.shape) # [b, 211200, 3]
            # print('cls_loss_src_size:', cls_loss_src.size()) # [b, 211200, 3]
            cls_loss = cls_loss_src.sum() / batch_size
            cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            mv_cls_loss.append(cls_loss)

        if mv_status == 1:
            mv_cls_loss_sum = mv_cls_loss[0] + mv_cls_loss[1] + mv_cls_loss[2]
            tb_dict={
                'rpn_loss_cls': mv_cls_loss[0].item(),
                'rpn_a1_loss_cls': mv_cls_loss[1].item(),
                'rpn_a2_loss_cls': mv_cls_loss[2].item(),
            }
        elif mv_status == 2 or mv_status == 3:
            mv_cls_loss_sum = mv_cls_loss[0] + mv_cls_loss[1]
            tb_dict={
                'rpn_loss_cls': mv_cls_loss[0].item(),
                'rpn_a_loss_cls': mv_cls_loss[1].item(),
            }
        return mv_cls_loss_sum, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        # print('box_preds_size:', box_preds.shape) # [2, 200, 176, 42]
        # print('box_reg_targets_size:', box_reg_targets.shape) # [2, 211200, 7]

        # 大于0为前景
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_box_reg_layer_loss_multiview(self, mv_status):
        if mv_status == 1:
            circle_num = 3
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight'] = 0.66
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight'] = 0.07
        elif mv_status == 2 or mv_status == 3:
            circle_num = 2
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight'] = 1
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight'] = 0.1
        mv_box_preds = self.forward_ret_dict['mv_box_preds']
        mv_box_reg_targets = self.forward_ret_dict['mv_box_reg_targets']
        mv_box_cls_labels = self.forward_ret_dict['mv_box_cls_labels']
        mv_box_dir_cls_preds = self.forward_ret_dict.get('mv_dir_cls_preds', None)
        mv_loc_loss = []
        mv_dir_loss = []
        for i in range(circle_num):
            box_preds = mv_box_preds[i]
            box_reg_targets = mv_box_reg_targets[i]
            box_cls_labels = mv_box_cls_labels[i]
            batch_size = int(box_preds.shape[0])
            # print('box_preds0_size:', box_preds.shape)  # [2, 200, 176, 42]
            # print('box_reg_targets0_size:', box_reg_targets.shape)  # [2, 211200, 7]
            positives = box_cls_labels > 0
            reg_weights = positives.float()
            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            if isinstance(self.anchors, list):
                if self.use_multihead:
                    anchors = torch.cat(
                        [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                         self.anchors], dim=0)
                else:
                    anchors = torch.cat(self.anchors, dim=-3)
            else:
                anchors = self.anchors
            anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
            box_preds = box_preds.view(batch_size, -1,
                                       box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                       box_preds.shape[-1])
            # sin(a - b) = sinacosb-cosasinb
            # print('box_preds_size:', box_preds.shape) # [b, 211200, 7]
            # print('box_reg_targets_size:', box_reg_targets.shape)  # [b, 211200, 7]
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
            loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
            # print('box_preds_sin_size:', box_preds_sin.shape) # [b, 211200, 7]
            # print('reg_targets_sin_size:', reg_targets_sin.shape)  # [b, 211200, 7]
            # print('loc_loss_src_size:', loc_loss_src.shape)  # [b, 211200, 7]
            loc_loss = loc_loss_src.sum() / batch_size
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            mv_loc_loss.append(loc_loss)
            if mv_box_dir_cls_preds is not None:
                box_dir_cls_preds = mv_box_dir_cls_preds[i]
                # print('dir_cls_preds_size:', box_dir_cls_preds.shape)  # [2, 200, 176, 12]
                dir_targets = self.get_direction_target(
                    anchors, box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS
                )
                dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                weights = positives.type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
                # print('dir_logits_size:', dir_logits.shape)  # [b, 211200, 2]
                # print('dir_targets_size:', dir_targets.shape)  # [b, 211200, 2]
                # print('dir_loss_size:', dir_loss.shape)  # [b, 211200]
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
                mv_dir_loss.append(dir_loss)

        if mv_status == 1:
            mv_box_loss_sum = mv_loc_loss[0] + mv_loc_loss[1] + mv_loc_loss[2]
            tb_dict={
                'rpn_loss_loc': mv_loc_loss[0].item(),
                'rpn_a1_loss_loc': mv_loc_loss[1].item(),
                'rpn_a2_loss_loc': mv_loc_loss[2].item(),
            }
            if mv_box_dir_cls_preds is not None:
                mv_box_loss_sum += mv_dir_loss[0] + mv_dir_loss[1] + mv_dir_loss[2]
                tb_dict.update({
                    'rpn_loss_dir': mv_dir_loss[0].item(),
                    'rpn_a1_loss_dir': mv_dir_loss[1].item(),
                    'rpn_a2_loss_dir': mv_dir_loss[2].item(),
                })
        elif mv_status == 2 or mv_status == 3:
            mv_box_loss_sum = mv_loc_loss[0] + mv_loc_loss[1]
            tb_dict={
                'rpn_loss_loc': mv_loc_loss[0].item(),
                'rpn_a_loss_loc': mv_loc_loss[1].item(),
            }
            if mv_box_dir_cls_preds is not None:
                mv_box_loss_sum += mv_dir_loss[0] + mv_dir_loss[1]
                tb_dict.update({
                    'rpn_loss_dir': mv_dir_loss[0].item(),
                    'rpn_a_loss_dir': mv_dir_loss[1].item(),
                })
        return mv_box_loss_sum, tb_dict

    def get_consis_loss_multiview(self, mv_status, tb_dict=None):
        if mv_status == 1:
            circle_num = 2
            self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['consis_cls_weight'] = 0.5
            self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['consis_box_weight'] = 0.5
        elif mv_status == 2 or mv_status == 3:
            circle_num = 1
            self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['consis_cls_weight'] = 1
            self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['consis_box_weight'] = 1
        weight_molecule = self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['weight_molecule']
        common_sum = self.forward_ret_dict['common_sum']
        if common_sum <= weight_molecule:
            dynamic_weight = 1
        else:
            dynamic_weight = weight_molecule / common_sum
        # print('weight_molecule:', self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['weight_molecule']) # 2000
        # print('common_sum:', self.forward_ret_dict['common_sum']) # 2569
        # print('dynamic_weight:', dynamic_weight) # 0.7785
        # print('stop:', self.forward_ret_dict['stop'])
        point_cls_labels = []
        point_cls_preds = []
        point_box_labels = []
        point_box_preds = []
        for i in range(circle_num):
            point_cls_labels.append(self.forward_ret_dict['consis_cls_labels'][i].view(-1, 3)) # [n, 3]
            point_cls_preds.append(self.forward_ret_dict['consis_cls_preds'][i].view(-1, 3)) # [n, 3]
            point_box_labels.append(self.forward_ret_dict['consis_box_labels'][i].view(-1, 7))  # [n, 7]
            point_box_preds.append(self.forward_ret_dict['consis_box_preds'][i].view(-1, 7))  # [n, 7]
        consis_loss_cls = []
        consis_loss_box = []
        for i in range(circle_num):
            cls_weights = torch.sum((point_cls_preds[i] != 10)/ self.num_class, dim=1).float()
            mid_consis_num = cls_weights.sum(dim=0).float()
            cls_weights /= torch.clamp(mid_consis_num,
                           min=1.0)
            cls_loss_src = self.consis_cls_loss_func(point_cls_preds[i], point_cls_labels[i], weights=cls_weights)
                                                                                # 求Sigmoid Focal Classification Loss
            mid_consis_loss_cls = cls_loss_src.sum()
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS
            # mid_consis_loss_cls = mid_consis_loss_cls * loss_weights_dict['consis_cls_weight'] * dynamic_weight
            mid_consis_loss_cls = mid_consis_loss_cls * loss_weights_dict['consis_cls_weight']
            consis_loss_cls.append(mid_consis_loss_cls)
            reg_weights = cls_weights
            box_loss_src = self.consis_reg_loss_func(point_box_preds[i][None, ...], point_box_labels[i][None, ...],
                                                     weights=reg_weights[None, ...]) # 求 WeightedSmoothL1Loss
            mid_consis_loss_box = box_loss_src.sum()
            # mid_consis_loss_box = mid_consis_loss_box * loss_weights_dict['consis_box_weight'] * dynamic_weight
            mid_consis_loss_box = mid_consis_loss_box * loss_weights_dict['consis_box_weight']
            consis_loss_box.append(mid_consis_loss_box)

        if tb_dict is None:
            tb_dict = {}
        if mv_status == 1:
            consis_loss_cls_sum = consis_loss_cls[0] + consis_loss_cls[1]
            consis_loss_box_sum = consis_loss_box[0] + consis_loss_box[1]
            # consis_loss = (consis_loss_cls_sum + consis_loss_box_sum) * finall_weight
            consis_loss = consis_loss_cls_sum + consis_loss_box_sum
            # print('consis_loss:', consis_loss)
            tb_dict.update({
                'consis_a_loss_cls': consis_loss_cls[0].item(),
                'consis_a_loss_box': consis_loss_box[0].item(),
                'consis_a2_loss_cls': consis_loss_cls[1].item(),
                'consis_a2_loss_box': consis_loss_box[1].item(),
                'consis_loss': consis_loss.item()
            })
        elif mv_status == 2 or mv_status == 3:
            consis_loss_cls_sum = consis_loss_cls[0]
            consis_loss_box_sum = consis_loss_box[0]
            # consis_loss = (consis_loss_cls_sum + consis_loss_box_sum) * finall_weight
            consis_loss = consis_loss_cls_sum + consis_loss_box_sum
            tb_dict.update({
                'consis_a_loss_cls': consis_loss_cls[0].item(),
                'consis_a_loss_box': consis_loss_box[0].item(),
                'consis_loss': consis_loss.item()
            })
        # print('tb_dict:', tb_dict)
        return consis_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        # print('done2')
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        # print('anchors_size:', anchors.shape) # [1, 200, 176, 3, 2, 7]
        # print('cls_preds_size:', cls_preds.shape) # [2, 200, 176, 18]
        # print('box_preds_size:', box_preds.shape) # [2, 200, 176, 42]
        # print('dir_cls_preds_size:', dir_cls_preds.shape) # [2, 200, 176, 12]
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        # print('num_anchors:', num_anchors) # 212000
        # print('batch_anchors_size:', batch_anchors.shape) # [2, 211200, 7]
        # print('batch_cls_preds_size:', batch_cls_preds.shape) # [2, 211200, 3]
        # print('batch_box_preds_size:', batch_box_preds.shape) # [2, 211200, 7]
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        # print('batch_box_preds_size:', batch_box_preds.shape) # [2, 211200, 7]

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
