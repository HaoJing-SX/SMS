import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils


class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

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
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss': # use
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None))
        else:
            self.reg_loss_func = F.smooth_l1_loss

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        # print('points_size:', points.shape)
        # print('gt_boxes_size:', gt_boxes.shape)

        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            if set_ignore_flag:
                # print('points_single1:', points_single.unsqueeze(dim=0))
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            # print('point_cls_labels_single:', point_cls_labels_single.size()) # [16384]
            # print('num_class:', self.num_class) # 3

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            # False
            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels # none
        }
        return targets_dict

    def assign_common_targets(self, batch_dict, targets_dict, mv_status):
        # print('batch_dict:', batch_dict)
        # print('batch_dict_gt_boxes:', batch_dict['gt_boxes'])
        # print('batch_dict_common_points:', batch_dict['common_points'])
        # print('batch_dict_common_point_indexs:', batch_dict['common_point_indexs'])
        # print('batch_dict_common_point_a1_indexs:', batch_dict['common_point_a1_indexs'])
        # print('batch_dict_common_point_a2_indexs:', batch_dict['common_point_a2_indexs'])
        # print('batch_cls_preds:', batch_dict['batch_cls_preds'])
        # print('batch_box_preds:', batch_dict['batch_box_preds'])

        # point_cls_labels = points.new_zeros(points.shape[0]).long()
        # point_box_labels = gt_boxes.new_zeros((points.shape[0], 8))
        # point_part_labels = gt_boxes.new_zeros((points.shape[0], 3))

        batch_size = batch_dict['batch_size']
        common_idxs = batch_dict['common_points'][:, 0]
        batch_idxs = batch_dict['point_coords'][:, 0]
        gt_boxes = batch_dict['gt_boxes']

        if True:
            all_num_buffer = []
            fg_num_buffer = []
            mid_gt_boxes = batch_dict['gt_boxes']
            mid_fg_num = 0
            hd_ori_batch_idxs = batch_dict['hd_ori_points'][:, 0]
            for k in range(batch_size):
                mid_bs_mask = (hd_ori_batch_idxs == k)
                mid_points = batch_dict['hd_ori_points'][mid_bs_mask][:, 1:4]
                mid_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    mid_points.unsqueeze(dim=0), mid_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                mid_box_fg_flag = (mid_box_idxs_of_pts >= 0)
                mid_fg_num += mid_box_fg_flag.sum().float()
            all_num_buffer.append(batch_dict['hd_ori_points'].shape[0])
            fg_num_buffer.append(mid_fg_num)

            mid_fg_num = 0
            sd_ori_batch_idxs = batch_dict['sd_ori_points'][:, 0]
            for k in range(batch_size):
                mid_bs_mask = (sd_ori_batch_idxs == k)
                mid_points = batch_dict['sd_ori_points'][mid_bs_mask][:, 1:4]
                mid_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    mid_points.unsqueeze(dim=0), mid_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                mid_box_fg_flag = (mid_box_idxs_of_pts >= 0)
                mid_fg_num += mid_box_fg_flag.sum().float()
            all_num_buffer.append(batch_dict['sd_ori_points'].shape[0])
            fg_num_buffer.append(mid_fg_num)

            mid_fg_num = 0
            hd_batch_idxs = batch_dict['hd_points'][:, 0]
            for k in range(batch_size):
                mid_bs_mask = (hd_batch_idxs == k)
                mid_points = batch_dict['hd_points'][mid_bs_mask][:, 1:4]
                mid_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    mid_points.unsqueeze(dim=0), mid_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                mid_box_fg_flag = (mid_box_idxs_of_pts >= 0)
                mid_fg_num += mid_box_fg_flag.sum().float()
            all_num_buffer.append(batch_dict['hd_points'].shape[0])
            fg_num_buffer.append(mid_fg_num)

            mid_fg_num = 0
            sd_batch_idxs = batch_dict['sd_points'][:, 0]
            for k in range(batch_size):
                mid_bs_mask = (sd_batch_idxs == k)
                mid_points = batch_dict['sd_points'][mid_bs_mask][:, 1:4]
                mid_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    mid_points.unsqueeze(dim=0), mid_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                mid_box_fg_flag = (mid_box_idxs_of_pts >= 0)
                mid_fg_num += mid_box_fg_flag.sum().float()
            all_num_buffer.append(batch_dict['sd_points'].shape[0])
            fg_num_buffer.append(mid_fg_num)
            if mv_status == 1 or mv_status == 2:

                mid_fg_num = 0
                hd_a1_batch_idxs = batch_dict['hd_points_a1'][:, 0]
                for k in range(batch_size):
                    mid_bs_mask = (hd_a1_batch_idxs == k)
                    mid_points = batch_dict['hd_points_a1'][mid_bs_mask][:, 1:4]
                    mid_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        mid_points.unsqueeze(dim=0), mid_gt_boxes[k:k + 1, :, 0:7].contiguous()
                    ).long().squeeze(dim=0)
                    mid_box_fg_flag = (mid_box_idxs_of_pts >= 0)
                    mid_fg_num += mid_box_fg_flag.sum().float()
                all_num_buffer.append(batch_dict['hd_points_a1'].shape[0])
                fg_num_buffer.append(mid_fg_num)

                mid_fg_num = 0
                sd_a1_batch_idxs = batch_dict['sd_points_a1'][:, 0]
                for k in range(batch_size):
                    mid_bs_mask = (sd_a1_batch_idxs == k)
                    mid_points = batch_dict['sd_points_a1'][mid_bs_mask][:, 1:4]
                    mid_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        mid_points.unsqueeze(dim=0), mid_gt_boxes[k:k + 1, :, 0:7].contiguous()
                    ).long().squeeze(dim=0)
                    mid_box_fg_flag = (mid_box_idxs_of_pts >= 0)
                    mid_fg_num += mid_box_fg_flag.sum().float()
                all_num_buffer.append(batch_dict['sd_points_a1'].shape[0])
                fg_num_buffer.append(mid_fg_num)
            if mv_status == 1 or mv_status == 3:

                mid_fg_num = 0
                hd_a2_batch_idxs = batch_dict['hd_points_a2'][:, 0]
                for k in range(batch_size):
                    mid_bs_mask = (hd_a2_batch_idxs == k)
                    mid_points = batch_dict['hd_points_a2'][mid_bs_mask][:, 1:4]
                    mid_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        mid_points.unsqueeze(dim=0), mid_gt_boxes[k:k + 1, :, 0:7].contiguous()
                    ).long().squeeze(dim=0)
                    mid_box_fg_flag = (mid_box_idxs_of_pts >= 0)
                    mid_fg_num += mid_box_fg_flag.sum().float()
                all_num_buffer.append(batch_dict['hd_points_a2'].shape[0])
                fg_num_buffer.append(mid_fg_num)

                mid_fg_num = 0
                sd_a2_batch_idxs = batch_dict['sd_points_a2'][:, 0]
                for k in range(batch_size):
                    mid_bs_mask = (sd_a2_batch_idxs == k)
                    mid_points = batch_dict['sd_points_a2'][mid_bs_mask][:, 1:4]
                    mid_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        mid_points.unsqueeze(dim=0), mid_gt_boxes[k:k + 1, :, 0:7].contiguous()
                    ).long().squeeze(dim=0)
                    mid_box_fg_flag = (mid_box_idxs_of_pts >= 0)
                    mid_fg_num += mid_box_fg_flag.sum().float()
                all_num_buffer.append(batch_dict['sd_points_a2'].shape[0])
                fg_num_buffer.append(mid_fg_num)
            # print('all_num_buffer:', all_num_buffer)
            # print('fg_num_buffer:', fg_num_buffer)

        consis_a_cls_preds = []
        consis_a_box_preds = []
        consis_a_cls_labels = []
        consis_a_box_labels = []
        if mv_status == 1:
            consis_a2_cls_preds = []
            consis_a2_box_preds = []
            consis_a2_cls_labels = []
            consis_a2_box_labels = []

        for k in range(batch_size):
            mid_common_gtpoint_indexs = []
            common_mask = (common_idxs == k)
            points_single = batch_dict['common_points'][common_mask][:, 1:4]
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            # batch mask
            bs_mask = (batch_idxs == k)

            if mv_status == 1:  # p pa1 pa2
                mid_common_gtpoint_indexs.append(batch_dict['common_point_indexs'][common_mask][box_fg_flag].long())
                mid_common_gtpoint_indexs.append(batch_dict['common_point_a1_indexs'][common_mask][box_fg_flag].long())
                mid_common_gtpoint_indexs.append(batch_dict['common_point_a2_indexs'][common_mask][box_fg_flag].long())
                # print('common_idxs_size:', common_idxs.size()) # 14891
                # print('points_single_size:', points_single.size()) # [6878, 3]  [8013, 3]
                # print('box_fg_flag_size:', torch.sum(box_fg_flag)) # 3248  2063
                # print('mid_common_gtpoint_indexs0_size:', mid_common_gtpoint_indexs[0].shape) # 3248  2063
                # print('mid_common_gtpoint_indexs1_size:', mid_common_gtpoint_indexs[1].shape) # 3248  2063
                # print('mid_common_gtpoint_indexs2_size:', mid_common_gtpoint_indexs[2].shape) # 3248  2063

                now_cls_preds = []
                now_box_preds = []
                for i in range(3):
                    mid_cls_preds = batch_dict['batch_cls_preds'][i][bs_mask][mid_common_gtpoint_indexs[i]]
                    mid_box_preds = batch_dict['batch_box_preds'][i][bs_mask][mid_common_gtpoint_indexs[i]]
                    now_cls_preds.append(mid_cls_preds)
                    now_box_preds.append(mid_box_preds)

                consis_a_cls_preds.append((now_cls_preds[1]).view(-1, 3))
                consis_a_box_preds.append((now_box_preds[1]).view(-1, 7))
                consis_a_cls_labels.append((now_cls_preds[0]).view(-1, 3))
                consis_a_box_labels.append((now_box_preds[0]).view(-1, 7))
                consis_a2_cls_preds.append((now_cls_preds[2]).view(-1, 3))
                consis_a2_box_preds.append((now_box_preds[2]).view(-1, 7))
                consis_a2_cls_labels.append((now_cls_preds[0]).view(-1, 3))
                consis_a2_box_labels.append((now_box_preds[0]).view(-1, 7))
            elif mv_status == 2:  # p pa1
                mid_common_gtpoint_indexs.append(batch_dict['common_point_indexs'][common_mask][box_fg_flag].long())
                mid_common_gtpoint_indexs.append(batch_dict['common_point_a1_indexs'][common_mask][box_fg_flag].long())
                # print('common_idxs_size:', common_idxs.size())
                # print('points_single_size:', points_single.size())
                # print('box_fg_flag_size:', torch.sum(box_fg_flag))
                # print('mid_common_gtpoint_indexs0_size:', mid_common_gtpoint_indexs[0])
                # print('mid_common_gtpoint_indexs1_size:', mid_common_gtpoint_indexs[1])

                now_cls_preds = []
                now_box_preds = []
                for i in range(2):
                    mid_cls_preds = batch_dict['batch_cls_preds'][i][bs_mask][mid_common_gtpoint_indexs[i]]
                    mid_box_preds = batch_dict['batch_box_preds'][i][bs_mask][mid_common_gtpoint_indexs[i]]
                    now_cls_preds.append(mid_cls_preds)
                    now_box_preds.append(mid_box_preds)

                consis_a_cls_preds.append((now_cls_preds[1]).view(-1, 3))
                consis_a_box_preds.append((now_box_preds[1]).view(-1, 7))
                consis_a_cls_labels.append((now_cls_preds[0]).view(-1, 3))
                consis_a_box_labels.append((now_box_preds[0]).view(-1, 7))
            elif mv_status == 3:  # p pa2
                mid_common_gtpoint_indexs.append(batch_dict['common_point_indexs'][common_mask][box_fg_flag].long())
                mid_common_gtpoint_indexs.append(batch_dict['common_point_a2_indexs'][common_mask][box_fg_flag].long())
                # print('common_idxs_size:', common_idxs.size())
                # print('points_single_size:', points_single.size())
                # print('box_fg_flag_size:', torch.sum(box_fg_flag))
                # print('mid_common_gtpoint_indexs0_size:', mid_common_gtpoint_indexs[0])
                # print('mid_common_gtpoint_indexs1_size:', mid_common_gtpoint_indexs[1])

                now_cls_preds = []
                now_box_preds = []
                for i in range(2):
                    mid_cls_preds = batch_dict['batch_cls_preds'][i][bs_mask][mid_common_gtpoint_indexs[i]]
                    mid_box_preds = batch_dict['batch_box_preds'][i][bs_mask][mid_common_gtpoint_indexs[i]]
                    now_cls_preds.append(mid_cls_preds)
                    now_box_preds.append(mid_box_preds)

                consis_a_cls_preds.append((now_cls_preds[1]).view(-1, 3))
                consis_a_box_preds.append((now_box_preds[1]).view(-1, 7))
                consis_a_cls_labels.append((now_cls_preds[0]).view(-1, 3))
                consis_a_box_labels.append((now_box_preds[0]).view(-1, 7))

        # batch cat
        consis_a_cls_preds = torch.cat(consis_a_cls_preds, dim=0)
        consis_a_box_preds = torch.cat(consis_a_box_preds, dim=0)
        consis_a_cls_labels = torch.cat(consis_a_cls_labels, dim=0)
        consis_a_box_labels = torch.cat(consis_a_box_labels, dim=0)
        if mv_status == 1:
            consis_a2_cls_preds = torch.cat(consis_a2_cls_preds, dim=0)
            consis_a2_box_preds = torch.cat(consis_a2_box_preds, dim=0)
            consis_a2_cls_labels = torch.cat(consis_a2_cls_labels, dim=0)
            consis_a2_box_labels = torch.cat(consis_a2_box_labels, dim=0)

        # print('a_cls_preds:', consis_a_cls_preds.size()) # [5311, 3]
        # print('a_box_preds:', consis_a_box_preds.size()) # [5311, 7]
        # print('a_cls_labels:', consis_a_cls_labels.size()) # [5311, 3]
        # print('a_box_labels:', consis_a_box_labels.size()) # [5311, 7]
        # print('a2_cls_preds:', consis_a2_cls_preds.size()) # [5311, 3]
        # print('a2_box_preds:', consis_a2_box_preds.size()) # [5311, 7]
        # print('a2_cls_labels:', consis_a2_cls_labels.size()) # [5311, 3]
        # print('a2_box_labels:', consis_a2_box_labels.size()) # [5311, 7]

        # targets_dict
        if mv_status == 1:
            targets_dict['consis_cls_preds'] = [consis_a_cls_preds, consis_a2_cls_preds]
            targets_dict['consis_box_preds'] = [consis_a_box_preds, consis_a2_box_preds]
            targets_dict['consis_cls_labels'] = [consis_a_cls_labels, consis_a2_cls_labels]
            targets_dict['consis_box_labels'] = [consis_a_box_labels, consis_a2_box_labels]
        elif mv_status == 2 or mv_status == 3:
            targets_dict['consis_cls_preds'] = [consis_a_cls_preds]
            targets_dict['consis_box_preds'] = [consis_a_box_preds]
            targets_dict['consis_cls_labels'] = [consis_a_cls_labels]
            targets_dict['consis_box_labels'] = [consis_a_box_labels]

        return targets_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1) # [b * 16384]
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class) # [b * 16384, 3]

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        # print('point_cls_labels:', point_cls_labels)
        # print('positives:', positives)
        # print('negative_cls_weights:', negative_cls_weights)
        # print('cls_weights:', cls_weights)
        # print('pos_normalizer:', pos_normalizer)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        # print('cls_weights:', cls_weights)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        # print('one_hot_targets_size:', one_hot_targets.size()) # [b * 16384, 4]
        one_hot_targets = one_hot_targets[..., 1:]
        # print('one_hot_targets:', one_hot_targets)
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights) # 求Sigmoid Focal Classification Loss
        # print('cls_loss_src:', cls_loss_src)
        # print('cls_loss_src_size:', cls_loss_src.size()) # [b * 16384, 3]
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_cls_layer_loss_multiview(self, mv_status, tb_dict=None):
        if mv_status == 1:
            circle_num = 3
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_cls_weight'] = 0.33
        elif mv_status == 2 or mv_status == 3:
            circle_num = 2
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_cls_weight'] = 0.5
        # print('point_cls_weight:', self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_cls_weight'])
        point_cls_labels = []
        point_cls_preds = []
        for i in range(circle_num):
            point_cls_labels.append(self.forward_ret_dict['point_cls_labels'][i].view(-1)) # [b * 16384]
            point_cls_preds.append(self.forward_ret_dict['point_cls_preds'][i].view(-1, self.num_class)) # [b * 16384, 3]
        point_loss_cls = []
        pos_normalizer = []
        for i in range(circle_num):
            positives = (point_cls_labels[i] > 0)
            negative_cls_weights = (point_cls_labels[i] == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            mid_pos_normalizer = positives.sum(dim=0).float()
            # print('point_cls_labels:', point_cls_labels)
            # print('positives:', positives)
            # print('negative_cls_weights:', negative_cls_weights)
            # print('cls_weights:', cls_weights)
            # print('cls_weights_size:', cls_weights.size())
            # print('mid_pos_normalizer:', mid_pos_normalizer)
            cls_weights /= torch.clamp(mid_pos_normalizer, min=1.0)
            # print('cls_weights:', cls_weights)
            one_hot_targets = point_cls_preds[i].new_zeros(*list(point_cls_labels[i].shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels[i] * (point_cls_labels[i] >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            # print('one_hot_targets_size:', one_hot_targets.size()) # [b * 16384, 4]
            one_hot_targets = one_hot_targets[..., 1:]
            cls_loss_src = self.cls_loss_func(point_cls_preds[i], one_hot_targets, weights=cls_weights) # 求Sigmoid Focal Classification Loss
            # print('point_cls_preds[i]:', point_cls_preds[i])
            # print('one_hot_targets:', one_hot_targets)
            # print('cls_loss_src:', cls_loss_src)
            # print('point_cls_preds[i]_size:', point_cls_preds[i].shape) # [b * 16384, 3]
            # print('one_hot_targets_size:', one_hot_targets.shape) # [b * 16384, 3]
            # print('cls_loss_src_size:', cls_loss_src.size()) # [b * 16384, 3]
            mid_point_loss_cls = cls_loss_src.sum()
            # loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            # mid_point_loss_cls = mid_point_loss_cls * loss_weights_dict['point_cls_weight']
            mid_point_loss_cls = mid_point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_cls_weight']
            point_loss_cls.append(mid_point_loss_cls)
            pos_normalizer.append(mid_pos_normalizer)

        if tb_dict is None:
            tb_dict = {}
        if mv_status == 1:
            point_loss_cls_sum = point_loss_cls[0] + point_loss_cls[1] + point_loss_cls[2]
            tb_dict.update({
                'point_loss_cls': point_loss_cls[0].item(),
                'point_pos_num': pos_normalizer[0].item(),
                'point_a1_loss_cls': point_loss_cls[1].item(),
                'point_a1_pos_num': pos_normalizer[1].item(),
                'point_a2_loss_cls': point_loss_cls[2].item(),
                'point_a2_pos_num': pos_normalizer[2].item()
            })
        elif mv_status == 2 or mv_status == 3:
            point_loss_cls_sum = point_loss_cls[0] + point_loss_cls[1]
            tb_dict.update({
                'point_loss_cls': point_loss_cls[0].item(),
                'point_pos_num': pos_normalizer[0].item(),
                'point_a_loss_cls': point_loss_cls[1].item(),
                'point_a_pos_num': pos_normalizer[1].item(),
            })
        return point_loss_cls_sum, tb_dict

    def get_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_box_labels'] # [b * 16384, 8]
        point_box_preds = self.forward_ret_dict['point_box_preds'] # [b * 16384, 8]

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        ) # 求 WeightedSmoothL1Loss
        # print('point_loss_box_src_size:', point_loss_box_src.size())  # [1, b * 16384, 8]
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    def get_box_layer_loss_multiview(self, mv_status, tb_dict=None):
        if mv_status == 1:
            circle_num = 3
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_box_weight'] = 0.33
        elif mv_status == 2 or mv_status == 3:
            circle_num = 2
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_box_weight'] = 0.5
        pos_mask = []
        point_box_labels = []
        point_box_preds = []
        for i in range(circle_num):
            pos_mask.append(self.forward_ret_dict['point_cls_labels'][i] > 0)
            point_box_labels.append(self.forward_ret_dict['point_box_labels'][i]) # [b * 16384, 8]
            point_box_preds.append(self.forward_ret_dict['point_box_preds'][i]) # [b * 16384, 8]
        point_loss_box = []
        for i in range(circle_num):
            reg_weights = pos_mask[i].float()
            pos_normalizer = pos_mask[i].sum().float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            # print('reg_weights:', reg_weights)
            # print('reg_weights_size:', reg_weights.size()) # [b * 16384]
            # print('point_box_preds_size:', point_box_preds[i].size()) # [b * 16384, 8]
            # print('point_box_labels_size:', point_box_labels[i].size()) # [b * 16384, 8]
            point_loss_box_src = self.reg_loss_func(
                point_box_preds[i][None, ...], point_box_labels[i][None, ...], weights=reg_weights[None, ...]
            ) # 求 WeightedSmoothL1Loss
            # print('point_loss_box_src_size:', point_loss_box_src.size())  # [1, b * 16384, 8]
            mid_point_loss_box = point_loss_box_src.sum()

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            mid_point_loss_box = mid_point_loss_box * loss_weights_dict['point_box_weight']
            point_loss_box.append(mid_point_loss_box)

        if mv_status == 1:
            point_loss_box_sum = point_loss_box[0] + point_loss_box[1] + point_loss_box[2]
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({'point_loss_box': point_loss_box[0].item(),
                            'point_a1_loss_box': point_loss_box[1].item(),
                            'point_a2_loss_box': point_loss_box[2].item()})
        elif mv_status == 2 or mv_status == 3:
            point_loss_box_sum = point_loss_box[0] + point_loss_box[1]
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({'point_loss_box': point_loss_box[0].item(),
                            'point_a_loss_box': point_loss_box[1].item()})
        return point_loss_box_sum, tb_dict

    def get_consis_loss_multiview(self, mv_status, tb_dict=None):
        if mv_status == 1:
            circle_num = 2
            self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['consis_cls_weight'] = 0.5
            self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['consis_box_weight'] = 0.5
        elif mv_status == 2 or mv_status == 3:
            circle_num = 1
            self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['consis_cls_weight'] = 1
            self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS['consis_box_weight'] = 1
        point_cls_labels = []
        point_cls_preds = []
        point_box_labels = []
        point_box_preds = []
        for i in range(circle_num):
            point_cls_labels.append(self.forward_ret_dict['consis_cls_labels'][i].view(-1, 3)) # [n, 3]
            point_cls_preds.append(self.forward_ret_dict['consis_cls_preds'][i].view(-1, 3)) # [n, 3]
            point_box_labels.append(self.forward_ret_dict['consis_box_labels'][i].view(-1, 7))  # [n, 7]
            point_box_preds.append(self.forward_ret_dict['consis_box_preds'][i].view(-1, 7))  # [n, 7]
        consis_num = []
        consis_loss_cls = []
        consis_loss_box = []
        for i in range(circle_num):
            # 全部是前景
            cls_weights = torch.sum((point_cls_preds[i] != 10)/ self.num_class, dim=1).float()
            mid_consis_num = cls_weights.sum(dim=0).float()
            cls_weights /= torch.clamp(mid_consis_num,
                                       min=1.0)

            cls_loss_src = self.consis_cls_loss_func(point_cls_preds[i], point_cls_labels[i], weights=cls_weights)

            mid_consis_loss_cls = cls_loss_src.sum()
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.CONSIS_LOSS_WEIGHTS
            mid_consis_loss_cls = mid_consis_loss_cls * loss_weights_dict['consis_cls_weight']
            consis_loss_cls.append(mid_consis_loss_cls)
            consis_num.append(mid_consis_num)

            reg_weights = cls_weights
            box_loss_src = self.consis_reg_loss_func(point_box_preds[i][None, ...], point_box_labels[i][None, ...],
                                                     weights=reg_weights[None, ...]) # 求 WeightedSmoothL1Loss
            mid_consis_loss_box = box_loss_src.sum()
            mid_consis_loss_box = mid_consis_loss_box * loss_weights_dict['consis_box_weight']
            consis_loss_box.append(mid_consis_loss_box)

        # # finall consis loss weight
        # now_epoch = self.forward_ret_dict['epoch_num'] # 1~80
        # full_weight_epoch = self.model_cfg.LOSS_CONFIG.FULL_WEIGHT_RPOCH
        # if now_epoch < full_weight_epoch:
        #     x = torch.tensor(now_epoch / full_weight_epoch)
        # else:
        #     x = torch.tensor(1.0)
        # finall_weight = 1 / torch.exp(((1 - x) ** 2) * 5)
        # print('finall_weight:', finall_weight)

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
                'consis_a_num': consis_num[0].item(),
                'consis_a2_loss_cls': consis_loss_cls[1].item(),
                'consis_a2_loss_box': consis_loss_box[1].item(),
                'consis_a2_num': consis_num[1].item(),
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
                'consis_a_num': consis_num[0].item(),
                'consis_loss': consis_loss.item()
            })
        # print('done1')
        # print('tb_dict:', tb_dict)
        return consis_loss, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)
        # print('points:', points)

        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
