import torch

from ...utils import box_utils
from .point_head_template import PointHeadTemplate


class PointHeadSimple(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

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
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
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
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }

        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict


class PointHeadSimpleMultiview(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

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
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def assign_targets_multiview(self, input_dict, mv_status):
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
            point_coords = [input_dict['mv_point_coords'], input_dict['mv_point_coords_a1'],
                            input_dict['mv_point_coords_a2']] # [b * 2048, 4]
            gt_boxes = [input_dict['gt_boxes'], input_dict['gt_boxes_a1'], input_dict['gt_boxes_a2']] # [b, n1, 8]
        elif mv_status == 2:
            point_coords = [input_dict['mv_point_coords'], input_dict['mv_point_coords_a1']] # [b * 16384, 4]
            gt_boxes = [input_dict['gt_boxes'], input_dict['gt_boxes_a1']] # [b, n1, 8]
        elif mv_status == 3:
            point_coords = [input_dict['mv_point_coords'], input_dict['mv_point_coords_a2']] # [b * 16384, 4]
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
                ret_part_labels=False, ret_box_labels=False
            )
            targets_dict['point_cls_labels'].append(mid_targets_dict['point_cls_labels'])

        return targets_dict

    def get_loss(self, tb_dict=None):
        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss_multiview(mv_status)
        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        # print('tb_dict:', tb_dict)
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
        # # single-view
        # if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
        #     point_features = batch_dict['point_features_before_fusion']
        # else:
        #     point_features = batch_dict['point_features']
        # # print('point_features_size:', point_features.shape) # [b*2048, 640]
        # # print('stop:', batch_dict['stop'])
        # point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        # # ret_dict = {
        # #     'point_cls_preds': point_cls_preds,
        # # }
        # point_cls_scores = torch.sigmoid(point_cls_preds)
        # batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        # if self.training:
        #     targets_dict = self.assign_targets(batch_dict)
        #     # ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        # # self.forward_ret_dict = ret_dict

        # multi-view
        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1
        if mv_status == 1:
            mv_point_features = batch_dict['mv_point_features_before_fusion']
            mv_point_features_a1 = batch_dict['mv_point_features_before_fusion_a1']
            mv_point_features_a2 = batch_dict['mv_point_features_before_fusion_a2']
            point_cls_preds = self.cls_layers(mv_point_features)
            point_cls_preds_a1 = self.cls_layers(mv_point_features_a1)
            point_cls_preds_a2 = self.cls_layers(mv_point_features_a2)
            mv_point_cls_preds = [point_cls_preds, point_cls_preds_a1, point_cls_preds_a2]
            ret_dict = {'point_cls_preds': mv_point_cls_preds}
            mv_point_cls_scores = torch.sigmoid(mv_point_cls_preds[0])
            mv_point_cls_scores_a1 = torch.sigmoid(mv_point_cls_preds[1])
            mv_point_cls_scores_a2 = torch.sigmoid(mv_point_cls_preds[2])
            batch_dict['mv_point_cls_scores'], _ = mv_point_cls_scores.max(dim=-1)
            batch_dict['mv_point_cls_scores_a1'], _ = mv_point_cls_scores_a1.max(dim=-1)
            batch_dict['mv_point_cls_scores_a2'], _ = mv_point_cls_scores_a2.max(dim=-1)
            # print('mv_point_cls_preds0_size:', mv_point_cls_preds[0].shape) # [b*2048, 1]
            # print('mv_point_cls_scores_size:', mv_point_cls_scores.shape)  # [b*2048, 1]
            # print('mv_point_cls_scores_max_size:', batch_dict['mv_point_cls_scores'].shape)  # [b*2048]
            if self.training:
                targets_dict = self.assign_targets_multiview(batch_dict, mv_status)
                ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
                # print('mv_point_cls_labels_len:', len(ret_dict['mv_point_cls_labels'])) # 3
                # print('mv_point_cls_labels0_size:', ret_dict['mv_point_cls_labels'][0].shape) # [b*2048]
                # print('stop:', batch_dict['stop'])
            self.forward_ret_dict = ret_dict
        return batch_dict