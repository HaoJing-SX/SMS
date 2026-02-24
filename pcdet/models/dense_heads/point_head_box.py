import torch
import torch.nn as nn
import numpy as np

from ...utils import box_coder_utils, box_utils, common_utils
from .point_head_template import PointHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils
from ..model_utils.model_nms_utils import class_agnostic_nms, mv_class_agnostic_nms

class PointHeadBox(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)

        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC, # [256, 256]
            input_channels=input_channels, # 128
            output_channels=num_class # 3
        )
        # print('input_channels:', input_channels)
        # print('num_class:', num_class)

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC, # [256, 256]
            input_channels=input_channels, # 128
            output_channels=self.box_coder.code_size # 8
        )
        # print('input_channels:', input_channels)
        # print('code_size:', self.box_coder.code_size)

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords'] # [b * 16384, 4]
        gt_boxes = input_dict['gt_boxes'] # [b, n1, 8]
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)
        point_box_preds = self.box_layers(point_features)
        # print('point_features_size:', point_features.size()) # [b * 16384, 128]

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)
        # print('point_cls_preds_max_size:', point_cls_preds_max.size()) # [b * 16384]
        # print('point_cls_preds_max:', point_cls_preds_max)

        # print('batch_dict:', batch_dict)
        # print('gt_boxes_size:', batch_dict['gt_boxes'].size())  # [b, N, 8]--[x, y, z, dx, dy, dz, heading, class]
        # print('points_size:', batch_dict['points'].size())  # [b * 16384, 5]--[b, x, y, z, r]
        # print('point_features_size:', batch_dict['point_features'].size())  # [b * 16384, 128]
        # print('point_coords_size:', batch_dict['point_coords'].size())  # [b * 16384, 4]--[b, x, y, z]
        # print('point_cls_scores_size:', batch_dict['point_cls_scores'].size())  # [b * 16384]
        # print('point_cls_preds_size:', point_cls_preds.size()) # [b * 16384, 3]
        # print('point_box_preds_size:', point_box_preds.size()) # [b * 16384, 8]

        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels'] # [b * 16384]
            ret_dict['point_box_labels'] = targets_dict['point_box_labels'] # [b * 16384, 8]
            # print('targets_dict_cls_size:', targets_dict['point_cls_labels'].size())
            # print('targets_dict_box_size:', targets_dict['point_box_labels'].size())
            # print('targets_dict_cls:', targets_dict['point_cls_labels'])
            # print('targets_dict_box:', targets_dict['point_box_labels'])

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict


class PointHeadMultiView(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)

        # self.predict_boxes_when_training = predict_boxes_when_training
        self.predict_boxes_when_training = True
        # self.cls_layers = nn.ModuleList()
        # for k in range(3):
        #     self.cls_layers.append(self.make_fc_layers(
        #         fc_cfg=self.model_cfg.CLS_FC, # [256, 256]
        #         input_channels=input_channels, # 128
        #         output_channels=num_class # 3
        #     ))
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC, # [256, 256]
            input_channels=input_channels, # 128
            output_channels=num_class # 3
        )
        # print('input_channels:', input_channels)
        # print('num_class:', num_class)

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        # self.box_layers = nn.ModuleList()
        # for k in range(3):
        #     self.box_layers.append(self.make_fc_layers(
        #         fc_cfg=self.model_cfg.REG_FC, # [256, 256]
        #         input_channels=input_channels, # 128
        #         output_channels=self.box_coder.code_size # 8
        #     ))
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC, # [256, 256]
            input_channels=input_channels, # 128
            output_channels=self.box_coder.code_size # 8
        )
        # print('input_channels:', input_channels)
        # print('code_size:', self.box_coder.code_size)

    def assign_targets(self, input_dict, mv_status):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        if mv_status == 1:
            point_coords = [input_dict['point_coords'], input_dict['point_a1_coords'],
                            input_dict['point_a2_coords']] # [b * 16384, 4]
            gt_boxes = [input_dict['gt_boxes'], input_dict['gt_boxes_a1'], input_dict['gt_boxes_a2']] # [b, n1, 8]
        elif mv_status == 2:
            point_coords = [input_dict['point_coords'], input_dict['point_a1_coords']] # [b * 16384, 4]
            gt_boxes = [input_dict['gt_boxes'], input_dict['gt_boxes_a1']] # [b, n1, 8]
        elif mv_status == 3:
            point_coords = [input_dict['point_coords'], input_dict['point_a2_coords']] # [b * 16384, 4]
            gt_boxes = [input_dict['gt_boxes'], input_dict['gt_boxes_a2']] # [b, n1, 8]
        assert gt_boxes[0].shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes[0].shape)
        assert point_coords[0].shape.__len__() in [2], 'points.shape=%s' % str(point_coords[0].shape)

        batch_size = gt_boxes[0].shape[0]
        targets_dict = {'point_cls_labels': [], 'point_box_labels': []}
        for i in range(len(point_coords)):
            extend_gt_boxes = box_utils.enlarge_box3d(
                gt_boxes[i].view(-1, gt_boxes[i].shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes[i].shape[-1])

            mid_targets_dict = self.assign_stack_targets(
                points=point_coords[i], gt_boxes=gt_boxes[i], extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_part_labels=False, ret_box_labels=True
            )
            targets_dict['point_cls_labels'].append(mid_targets_dict['point_cls_labels'])
            targets_dict['point_box_labels'].append(mid_targets_dict['point_box_labels'])
        return targets_dict

    def get_loss(self, tb_dict=None):
        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss_multiview(mv_status)
        point_loss_box, tb_dict_2 = self.get_box_layer_loss_multiview(mv_status)

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def get_consis_loss(self, tb_dict=None):
        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1
        tb_dict = {} if tb_dict is None else tb_dict
        consis_loss, tb_dict1 = self.get_consis_loss_multiview(mv_status)
        tb_dict.update(tb_dict1)
        return consis_loss, tb_dict

    # iou-based consis code+
    def mv_targets(self, batch_dict, targets_dict, mv_status):
        print('cls_preds0_size:', batch_dict['batch_cls_preds'][0].size())
        print('box_preds0_size:', batch_dict['batch_box_preds'][0].size())
    #     # # print('batch_dict:', batch_dict)
    #     # # print('targets_dict:', targets_dict)
    #     # print('nms_config:', self.model_cfg['NMS_CONFIG'])
    #     # print('SCORE_THRESH:', self.model_cfg['SCORE_THRESH'])
    #     batch_size = batch_dict['batch_size']
    #     batch_index = batch_dict['point_coords'][:, 0]
    #     consis_a_cls_preds = []
    #     consis_a_box_preds = []
    #     consis_a_cls_labels = []
    #     consis_a_box_labels = []
    #     consis_a2_cls_preds = []
    #     consis_a2_box_preds = []
    #     consis_a2_cls_labels = []
    #     consis_a2_box_labels = []
    #     for k in range(batch_size):
    #     # for k in range(1):
    #         bs_mask = (batch_index == k)
    #         now_cls_preds = []
    #         now_box_preds = []
    #         now_cls_scores = []
    #         now_cls_target = []
    #         if mv_status == 1: # p pa1 pa2
    #             for i in range(3):
    #                 mid_cls_preds = batch_dict['batch_cls_preds'][i][bs_mask]
    #                 mid_box_preds = batch_dict['batch_box_preds'][i][bs_mask]
    #                 mid_cls_target = targets_dict['point_cls_labels'][i][bs_mask]
    #                 mid_cls_scores, _ = torch.max(mid_cls_preds, dim=1)
    #                 # print('mid_score_size:', mid_cls_scores.size())
    #                 # a, _ = torch.max(mid_cls_scores, dim=0)
    #                 # b, _ = torch.min(mid_cls_scores, dim=0)
    #                 # print('max:', a, 'min:', b)
    #                 if i == 0: # core p
    #                     # selected, selected_scores = class_agnostic_nms(box_scores=mid_cls_scores,
    #                     #         box_preds=mid_box_preds, nms_config=self.nms_cfg['CORE'],
    #                     #         score_thresh=self.model_cfg.SCORE_THRESH)
    #                     selected, selected_scores = class_agnostic_nms(box_scores=mid_cls_scores,
    #                             box_preds=mid_box_preds, nms_config=self.nms_cfg['CORE'])
    #                 else:
    #                     selected, selected_scores = class_agnostic_nms(box_scores=mid_cls_scores,
    #                             box_preds=mid_box_preds, nms_config=self.nms_cfg['BRANCH'])
    #                 now_cls_preds.append(mid_cls_preds[selected])
    #                 now_box_preds.append(mid_box_preds[selected])
    #                 now_cls_target.append(mid_cls_target[selected])
    #                 now_cls_scores.append(selected_scores)
    #             # print('now_box_preds1_size:', now_box_preds[1].size())
    #             # print('now_box_preds0_size:', now_box_preds[0].size())
    #             # print('selected_scores:', selected_scores)
    #             # print('selected_scores_size:', selected_scores.size())
    #
    #             # iou match: pa1 p
    #             pa_iou = iou3d_nms_utils.boxes_iou3d_gpu(now_box_preds[1][:, 0:7], now_box_preds[0][:, 0:7])
    #             # print('pa_iou:', pa_iou)
    #             # print('pa_iou_size:', pa_iou.shape)
    #             pa_iou_max, pa_iou_max_index = torch.max(pa_iou, dim=1)
    #             # print('pa_iou_max_size:', pa_iou_max.shape)
    #             # print('pa_iou_max_index_size:', pa_iou_max_index.shape)
    #             pa_iou_flag = (pa_iou_max > self.model_cfg.IOU_THRESH) & (now_cls_target[1] > 0)
    #             # print('pa_iou_flag_sum:', k, torch.sum(pa_iou_flag))
    #             if torch.sum(pa_iou_flag) == 0:
    #                 _, pa_iou_max_top_index = torch.sort(pa_iou_max, descending=True)
    #                 consis_a_cls_preds.append((now_cls_preds[1][pa_iou_max_top_index[0]]).view(-1, 3))
    #                 consis_a_box_preds.append((now_box_preds[1][pa_iou_max_top_index[0]]).view(-1, 7))
    #                 consis_a_cls_labels.append((now_cls_preds[0][pa_iou_max_index[pa_iou_max_top_index[0]]]).view(-1, 3))
    #                 consis_a_box_labels.append((now_box_preds[0][pa_iou_max_index[pa_iou_max_top_index[0]]]).view(-1, 7))
    #             else:
    #                 consis_a_cls_preds.append((now_cls_preds[1][pa_iou_flag]).view(-1, 3))
    #                 consis_a_box_preds.append((now_box_preds[1][pa_iou_flag]).view(-1, 7))
    #                 consis_a_cls_labels.append((now_cls_preds[0][pa_iou_max_index[pa_iou_flag]]).view(-1, 3))
    #                 consis_a_box_labels.append((now_box_preds[0][pa_iou_max_index[pa_iou_flag]]).view(-1, 7))
    #             # iou match: pa2 p
    #             pa2_iou = iou3d_nms_utils.boxes_iou3d_gpu(now_box_preds[2][:, 0:7], now_box_preds[0][:, 0:7])
    #             pa2_iou_max, pa2_iou_max_index = torch.max(pa2_iou, dim=1)
    #             pa2_iou_flag = (pa2_iou_max > self.model_cfg.IOU_THRESH) & (now_cls_target[2] > 0)
    #             # print('pa2_iou_flag_sum:', k, torch.sum(pa2_iou_flag))
    #             if torch.sum(pa2_iou_flag) == 0:
    #                 _, pa2_iou_max_top_index = torch.sort(pa2_iou_max, descending=True)
    #                 consis_a2_cls_preds.append((now_cls_preds[2][pa2_iou_max_top_index[0]]).view(-1, 3))
    #                 consis_a2_box_preds.append((now_box_preds[2][pa2_iou_max_top_index[0]]).view(-1, 7))
    #                 consis_a2_cls_labels.append((now_cls_preds[0][pa2_iou_max_index[pa2_iou_max_top_index[0]]]).view(-1, 3))
    #                 consis_a2_box_labels.append((now_box_preds[0][pa2_iou_max_index[pa2_iou_max_top_index[0]]]).view(-1, 7))
    #             else:
    #                 consis_a2_cls_preds.append((now_cls_preds[2][pa2_iou_flag]).view(-1, 3))
    #                 consis_a2_box_preds.append((now_box_preds[2][pa2_iou_flag]).view(-1, 7))
    #                 consis_a2_cls_labels.append((now_cls_preds[0][pa2_iou_max_index[pa2_iou_flag]]).view(-1, 3))
    #                 consis_a2_box_labels.append((now_box_preds[0][pa2_iou_max_index[pa2_iou_flag]]).view(-1, 7))
    #         elif mv_status == 2 or mv_status == 3:  # pa1 p or pa2 p
    #             for i in range(2):
    #                 mid_cls_preds = batch_dict['batch_cls_preds'][i][bs_mask]
    #                 mid_box_preds = batch_dict['batch_box_preds'][i][bs_mask]
    #                 mid_cls_target = targets_dict['point_cls_labels'][i][bs_mask]
    #                 mid_cls_scores, _ = torch.max(mid_cls_preds, dim=1)
    #                 # print('mid_score_size:', mid_cls_scores.size())
    #                 if i == 0: # core p
    #                     selected, selected_scores = class_agnostic_nms(box_scores=mid_cls_scores,
    #                             box_preds=mid_box_preds, nms_config=self.nms_cfg['CORE'],
    #                             score_thresh=self.model_cfg.SCORE_THRESH)
    #                 else:
    #                     selected, selected_scores = class_agnostic_nms(box_scores=mid_cls_scores,
    #                             box_preds=mid_box_preds, nms_config=self.nms_cfg['BRANCH'],
    #                             score_thresh=self.model_cfg.SCORE_THRESH)
    #                 now_cls_preds.append(mid_cls_preds[selected])
    #                 now_box_preds.append(mid_box_preds[selected])
    #                 now_cls_target.append(mid_cls_target[selected])
    #                 now_cls_scores.append(selected_scores)
    #             # iou match: pa1 p or pa2 p
    #             pa_iou = iou3d_nms_utils.boxes_iou3d_gpu(now_box_preds[1][:, 0:7], now_box_preds[0][:, 0:7])
    #             pa_iou_max, pa_iou_max_index = torch.max(pa_iou, dim=1)
    #             pa_iou_flag = (pa_iou_max > self.model_cfg.IOU_THRESH) & (now_cls_target[1] > 0)
    #             if torch.sum(pa_iou_flag) == 0:
    #                 _, pa_iou_max_top_index = torch.sort(pa_iou_max, descending=True)
    #                 consis_a_cls_preds.append((now_cls_preds[1][pa_iou_max_top_index[0]]).view(-1, 3))
    #                 consis_a_box_preds.append((now_box_preds[1][pa_iou_max_top_index[0]]).view(-1, 7))
    #                 consis_a_cls_labels.append((now_cls_preds[0][pa_iou_max_index[pa_iou_max_top_index[0]]]).view(-1, 3))
    #                 consis_a_box_labels.append((now_box_preds[0][pa_iou_max_index[pa_iou_max_top_index[0]]]).view(-1, 7))
    #             else:
    #                 consis_a_cls_preds.append((now_cls_preds[1][pa_iou_flag]).view(-1, 3))
    #                 consis_a_box_preds.append((now_box_preds[1][pa_iou_flag]).view(-1, 7))
    #                 consis_a_cls_labels.append((now_cls_preds[0][pa_iou_max_index[pa_iou_flag]]).view(-1, 3))
    #                 consis_a_box_labels.append((now_box_preds[0][pa_iou_max_index[pa_iou_flag]]).view(-1, 7))
    #     # print('a2_cls_preds:', consis_a2_cls_preds)
    #     # print('a2_box_preds:', consis_a2_box_preds)
    #     # print('a2_cls_labels:', consis_a2_cls_labels)
    #     # print('a2_box_labels:', consis_a2_box_labels)
    #     # print('a2_cls_preds:', consis_a2_cls_preds)
    #     # print('a2_box_preds:', consis_a2_box_preds)
    #     # print('a2_cls_labels:', consis_a2_cls_labels)
    #     # print('a2_box_labels:', consis_a2_box_labels)
    #     # batch cat
    #     consis_a_cls_preds = torch.cat(consis_a_cls_preds, dim=0)
    #     consis_a_box_preds = torch.cat(consis_a_box_preds, dim=0)
    #     consis_a_cls_labels = torch.cat(consis_a_cls_labels, dim=0)
    #     consis_a_box_labels = torch.cat(consis_a_box_labels, dim=0)
    #     if mv_status == 1:
    #         consis_a2_cls_preds = torch.cat(consis_a2_cls_preds, dim=0)
    #         consis_a2_box_preds = torch.cat(consis_a2_box_preds, dim=0)
    #         consis_a2_cls_labels = torch.cat(consis_a2_cls_labels, dim=0)
    #         consis_a2_box_labels = torch.cat(consis_a2_box_labels, dim=0)
    #     # print('a_cls_preds:', consis_a_cls_preds)
    #     # print('a_box_preds:', consis_a_box_preds)
    #     # print('a_cls_labels:', consis_a_cls_labels)
    #     # print('a_box_labels:', consis_a_box_labels)
    #     # print('a2_cls_preds:', consis_a2_cls_preds)
    #     # print('a2_box_preds:', consis_a2_box_preds)
    #     # print('a2_cls_labels:', consis_a2_cls_labels)
    #     # print('a2_box_labels:', consis_a2_box_labels)
    #
    #     # targets_dict
    #     if mv_status == 1:
    #         targets_dict['consis_cls_preds'] = [consis_a_cls_preds, consis_a2_cls_preds]
    #         targets_dict['consis_box_preds'] = [consis_a_box_preds, consis_a2_box_preds]
    #         targets_dict['consis_cls_labels'] = [consis_a_cls_labels, consis_a2_cls_labels]
    #         targets_dict['consis_box_labels'] = [consis_a_box_labels, consis_a2_box_labels]
    #     elif mv_status == 2 or mv_status == 3:
    #         targets_dict['consis_cls_preds'] = consis_a_cls_preds
    #         targets_dict['consis_box_preds'] = consis_a_box_preds
    #         targets_dict['consis_cls_labels'] = consis_a_cls_labels
    #         targets_dict['consis_box_labels'] = consis_a_box_labels
    #
        return targets_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        # print('batch_dict:', batch_dict)

        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1

        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = []
            if mv_status == 1:
                point_features = [batch_dict['point_features'], batch_dict['point_a1_features'],
                                  batch_dict['point_a2_features']]
                point_coords = [batch_dict['point_coords'][:, 1:4], batch_dict['point_a1_coords'][:, 1:4],
                                batch_dict['point_a2_coords'][:, 1:4]]
            elif mv_status == 2:
                point_features = [batch_dict['point_features'], batch_dict['point_a1_features']]
                point_coords = [batch_dict['point_coords'][:, 1:4], batch_dict['point_a1_coords'][:, 1:4]]
            elif mv_status == 3:
                point_features = [batch_dict['point_features'], batch_dict['point_a2_features']]
                point_coords = [batch_dict['point_coords'][:, 1:4], batch_dict['point_a2_coords'][:, 1:4]]
        point_cls_preds = []
        point_box_preds = []
        point_cls_preds_max = []
        pred_classes = []
        for i in range(len(point_features)):
            cls_preds = self.cls_layers(point_features[i])
            box_preds = self.box_layers(point_features[i])
            mid_cls_preds_max, _ = cls_preds.max(dim=-1)

            point_cls_preds.append(cls_preds)
            point_box_preds.append(box_preds)
            point_cls_preds_max.append(torch.sigmoid(mid_cls_preds_max))
            # pred_classes.append(mid_pred_classes)

        ret_dict = {'point_cls_preds': point_cls_preds.copy(),
                    'point_box_preds': point_box_preds.copy()}

        if self.training:
            targets_dict = self.assign_targets(batch_dict, mv_status)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels'] # [b * 16384] len=3 or 2
            ret_dict['point_box_labels'] = targets_dict['point_box_labels'] # [b * 16384, 8] len=3 or 2

            # print('len_targets_dict_cls:', len(targets_dict['point_cls_labels'])) # len=3 or 2
            # print('targets_dict_cls0_size:', targets_dict['point_cls_labels'][0].shape) # [b * 16384] len=3 or 2
            # print('targets_dict_box0_size:', targets_dict['point_box_labels'][0].shape) # [b * 16384, 8] len=3 or 2
            # print('targets_dict_cls:', targets_dict['point_cls_labels'])
            # print('targets_dict_box:', targets_dict['point_box_labels'])

        if not self.training or self.predict_boxes_when_training:
            mid_point_box_preds = point_box_preds.copy()
            for i in range(len(point_box_preds)):
                point_cls_preds[i], mid_point_box_preds[i] = self.generate_predicted_boxes(
                    points=point_coords[i],
                    point_cls_preds=point_cls_preds[i], point_box_preds=point_box_preds[i]
                )
            # print('mid_point_box_preds_size:', mid_point_box_preds[0].size())

            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = mid_point_box_preds
            batch_dict['point_cls_scores'] = point_cls_preds_max
            batch_dict['cls_preds_normalized'] = False
            if mv_status == 1:
                batch_dict['batch_index'] = [batch_dict['point_coords'][:, 0], batch_dict['point_a1_coords'][:, 0],
                                            batch_dict['point_a2_coords'][:, 0]]
            elif mv_status == 2:
                batch_dict['batch_index'] = [batch_dict['point_coords'][:, 0], batch_dict['point_a1_coords'][:, 0]]
            elif mv_status == 3:
                batch_dict['batch_index'] = [batch_dict['point_coords'][:, 0], batch_dict['point_a2_coords'][:, 0]]

            # print('len_batch_cls_preds:', len(batch_dict['batch_cls_preds'])) # len=3 or 2
            # print('batch_a1_coords_size:', batch_dict['point_a1_coords'].size()) # [b * 16384, 4]
            # print('batch_a2_coords_size:', batch_dict['point_a2_coords'].size()) # [b * 16384, 4]
            # print('batch_a1_features_size:', batch_dict['point_a1_features'].size()) # [b * 16384, 128]
            # print('batch_a2_features_size:', batch_dict['point_a2_features'].size()) # [b * 16384, 128]
            # print('batch_cls_preds1_size:', batch_dict['batch_cls_preds'][1].size()) # [b * 16384, 3]
            # print('batch_box_preds1_size:', batch_dict['batch_box_preds'][1].size()) # [b * 16384, 7]
            # print('point_cls_scores1_size:', batch_dict['point_cls_scores'][1].size()) # [b * 16384]
            # print('batch_cls_preds2_size:', batch_dict['batch_cls_preds'][2].size()) # [b * 16384, 3]
            # print('batch_box_preds2_size:', batch_dict['batch_box_preds'][2].size()) # [b * 16384, 7]
            # print('point_cls_scores2_size:', batch_dict['point_cls_scores'][2].size()) # [b * 16384]

        # prepare for consis loss
        if self.training:
            targets_dict = self.assign_common_targets(batch_dict, targets_dict, mv_status)
            ret_dict['consis_cls_preds'] = targets_dict['consis_cls_preds']
            ret_dict['consis_box_preds'] = targets_dict['consis_box_preds']
            ret_dict['consis_cls_labels'] = targets_dict['consis_cls_labels']
            ret_dict['consis_box_labels'] = targets_dict['consis_box_labels']
            ret_dict['epoch_num'] = batch_dict['epoch_num']

        self.forward_ret_dict = ret_dict

        # print('ret_consis_cls_preds:', ret_dict['consis_cls_preds'])
        # print('ret_consis_box_preds:', ret_dict['consis_box_preds'])
        # print('ret_consis_cls_labels:', ret_dict['consis_cls_labels'])
        # print('ret_consis_box_labels:', ret_dict['consis_box_labels'])
        # print('ret_consis_cls_preds:', ret_dict['consis_cls_preds'][0].size()) # [n, 3]
        # print('ret_consis_box_preds:', ret_dict['consis_box_preds'][0].size()) # [n, 7]
        # print('ret_consis_cls_labels:', ret_dict['consis_cls_labels'][0].size()) # [n, 3]
        # print('ret_consis_box_labels:', ret_dict['consis_box_labels'][0].size()) # [n, 7]
        # print('ret_consis_cls_preds2:', ret_dict['consis_cls_preds'][1].size())
        # print('ret_consis_box_preds2:', ret_dict['consis_box_preds'][1].size())
        # print('ret_consis_cls_labels2:', ret_dict['consis_cls_labels'][1].size())
        # print('ret_consis_box_labels2:', ret_dict['consis_box_labels'][1].size())

        return batch_dict