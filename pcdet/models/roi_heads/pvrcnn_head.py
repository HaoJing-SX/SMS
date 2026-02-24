import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        # print('@@@@@@@@@@@@@@ batch_dict:', batch_dict)
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
            # print('@@@@@@@@@@@@@@@ batch_dict:', batch_dict)

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        # print('pooled_features_size:', pooled_features.size()) # [b*128, 6x6x6, 128]

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        # a = {}
        # print("stop:", a[10])
        return batch_dict


class PVRCNNHeadMultiview(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    # @torch.no_grad()
    def data_cat(self, batch_dict, data_dict, a_data_dict):
        # print('batch_dict:', batch_dict)
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['mv_point_coords'][:, 0]
        mv_status = batch_dict['mv_status']
        # # delete
        # if mv_status == 1:
        #     batch_dict.pop('gt_boxes_a1', None)
        #     batch_dict.pop('gt_boxes_a2', None)
        #     batch_dict.pop('voxels', None)
        #     batch_dict.pop('voxel_coords', None)
        #     batch_dict.pop('voxel_num_points', None)
        #     batch_dict.pop('voxels_a1', None)
        #     batch_dict.pop('voxel_a1_coords', None)
        #     batch_dict.pop('voxel_a1_num_points', None)
        #     batch_dict.pop('voxels_a2', None)
        #     batch_dict.pop('voxel_a2_coords', None)
        #     batch_dict.pop('voxel_a2_num_points', None)
        #     batch_dict.pop('ori_points', None)
        #     batch_dict.pop('hd_ori_points', None)
        #     batch_dict.pop('sd_ori_points', None)
        #     batch_dict.pop('hd_points', None)
        #     batch_dict.pop('sd_points', None)
        #     batch_dict.pop('hd_points_a1', None)
        #     batch_dict.pop('sd_points_a1', None)
        #     batch_dict.pop('hd_points_a2', None)
        #     batch_dict.pop('sd_points_a2', None)
        #     batch_dict.pop('voxel_features', None)
        #     batch_dict.pop('voxel_features_a1', None)
        #     batch_dict.pop('voxel_features_a2', None)
        #     batch_dict.pop('spatial_features', None)
        #     batch_dict.pop('spatial_features_a1', None)
        #     batch_dict.pop('spatial_features_a2', None)
        #     batch_dict.pop('spatial_features_2d', None)
        #     batch_dict.pop('spatial_features_2d_a1', None)
        #     batch_dict.pop('spatial_features_2d_a2', None)

        mid_point_coords = []
        mid_point_features = []
        mid_point_cls_scores = []
        for index in range(batch_size):
            if mv_status == 1:
                batch_mask = (batch_idx == index)
                mid_point_coords.append(torch.cat([batch_dict['mv_point_coords'][batch_mask],
                                                   batch_dict['mv_point_coords_a1'][batch_mask],
                                                   batch_dict['mv_point_coords_a2'][batch_mask]]))
                mid_point_features.append(torch.cat([batch_dict['mv_point_features'][batch_mask],
                                                   batch_dict['mv_point_features_a1'][batch_mask],
                                                   batch_dict['mv_point_features_a2'][batch_mask]]))
                mid_point_cls_scores.append(torch.cat([batch_dict['mv_point_cls_scores'][batch_mask],
                                                   batch_dict['mv_point_cls_scores_a1'][batch_mask],
                                                   batch_dict['mv_point_cls_scores_a2'][batch_mask]]))
        # cat
        batch_dict['point_coords'] = (torch.cat(mid_point_coords))
        batch_dict['point_features'] = (torch.cat(mid_point_features))
        batch_dict['point_cls_scores'] = (torch.cat(mid_point_cls_scores))
        batch_dict['has_class_labels'] = data_dict['has_class_labels']
        # print('point_coords_size:', batch_dict['point_coords'].size()) # [b*c*2048, 4]
        # print('point_features_size:', batch_dict['point_features'].size()) # [b*c*2048, 128]
        # print('point_cls_scores_size:', batch_dict['point_cls_scores'].size()) # [b*c*2048]
        if mv_status == 1:
            batch_dict['rois'] = torch.cat([data_dict['rois'], a_data_dict[0]['rois'], a_data_dict[1]['rois']], dim=1)
            batch_dict['roi_scores'] = torch.cat([data_dict['roi_scores'], a_data_dict[0]['roi_scores'],
                                                  a_data_dict[1]['roi_scores']], dim=1)
            batch_dict['roi_labels'] = torch.cat([data_dict['roi_labels'], a_data_dict[0]['roi_labels'],
                                                  a_data_dict[1]['roi_labels']], dim=1)
        # print('rois_size:', batch_dict['rois'].size()) # [b, c*512, 7]
        # print('roi_scores_size:', batch_dict['roi_scores'].size()) # [b, c*512]
        # print('roi_labels_size:', batch_dict['roi_labels'].size()) # [b, c*512]
        return batch_dict

    # @torch.no_grad()
    def data_cat_plusplus(self, batch_dict, is_train, target_dict):
        # print('batch_dict:', batch_dict)
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['mv_point_coords'][:, 0]
        mv_status = batch_dict['mv_status']
        # # delete
        # if mv_status == 1:
        #     batch_dict.pop('gt_boxes_a1', None)
        #     batch_dict.pop('gt_boxes_a2', None)
        #     batch_dict.pop('voxels', None)
        #     batch_dict.pop('voxel_coords', None)
        #     batch_dict.pop('voxel_num_points', None)
        #     batch_dict.pop('voxels_a1', None)
        #     batch_dict.pop('voxel_a1_coords', None)
        #     batch_dict.pop('voxel_a1_num_points', None)
        #     batch_dict.pop('voxels_a2', None)
        #     batch_dict.pop('voxel_a2_coords', None)
        #     batch_dict.pop('voxel_a2_num_points', None)
        #     batch_dict.pop('ori_points', None)
        #     batch_dict.pop('hd_ori_points', None)
        #     batch_dict.pop('sd_ori_points', None)
        #     batch_dict.pop('hd_points', None)
        #     batch_dict.pop('sd_points', None)
        #     batch_dict.pop('hd_points_a1', None)
        #     batch_dict.pop('sd_points_a1', None)
        #     batch_dict.pop('hd_points_a2', None)
        #     batch_dict.pop('sd_points_a2', None)
        #     batch_dict.pop('voxel_features', None)
        #     batch_dict.pop('voxel_features_a1', None)
        #     batch_dict.pop('voxel_features_a2', None)
        #     batch_dict.pop('spatial_features', None)
        #     batch_dict.pop('spatial_features_a1', None)
        #     batch_dict.pop('spatial_features_a2', None)
        #     batch_dict.pop('spatial_features_2d', None)
        #     batch_dict.pop('spatial_features_2d_a1', None)
        #     batch_dict.pop('spatial_features_2d_a2', None)

        mid_point_coords = []
        mid_point_features = []
        mid_point_cls_scores = []
        for index in range(batch_size):
            if mv_status == 1:
                batch_mask = (batch_idx == index)
                mid_point_coords.append(torch.cat([batch_dict['mv_point_coords'][batch_mask],
                                                   batch_dict['mv_point_coords_a1'][batch_mask],
                                                   batch_dict['mv_point_coords_a2'][batch_mask]]))
                mid_point_features.append(torch.cat([batch_dict['mv_point_features'][batch_mask],
                                                   batch_dict['mv_point_features_a1'][batch_mask],
                                                   batch_dict['mv_point_features_a2'][batch_mask]]))
                mid_point_cls_scores.append(torch.cat([batch_dict['mv_point_cls_scores'][batch_mask],
                                                   batch_dict['mv_point_cls_scores_a1'][batch_mask],
                                                   batch_dict['mv_point_cls_scores_a2'][batch_mask]]))
        # cat
        batch_dict['point_coords'] = (torch.cat(mid_point_coords))
        batch_dict['point_features'] = (torch.cat(mid_point_features))
        batch_dict['point_cls_scores'] = (torch.cat(mid_point_cls_scores))
        # print('point_coords_size:', batch_dict['point_coords'].size()) # [b*c*2048, 4] [b*c*4096, 4]
        # print('point_features_size:', batch_dict['point_features'].size()) # [b*c*2048, 128] [b*c*4096, 90]
        # print('point_cls_scores_size:', batch_dict['point_cls_scores'].size()) # [b*c*2048] [b*c*4096]

        if mv_status == 1:
            batch_dict['rois'] = torch.cat([batch_dict['mv_rois'][0], batch_dict['mv_rois'][1], batch_dict['mv_rois'][2]], dim=1)
            batch_dict['roi_scores'] = torch.cat([batch_dict['mv_roi_scores'][0], batch_dict['mv_roi_scores'][1],
                                                  batch_dict['mv_roi_scores'][2]], dim=1)
            batch_dict['roi_labels'] = torch.cat([batch_dict['mv_roi_labels'][0], batch_dict['mv_roi_labels'][1],
                                                  batch_dict['mv_roi_labels'][2]], dim=1)
            # print('rois_size:', batch_dict['rois'].size()) # [b, c*512, 7] [b, c*128, 7]
            # print('roi_scores_size:', batch_dict['roi_scores'].size()) # [b, c*512] [b, c*128]
            # print('roi_labels_size:', batch_dict['roi_labels'].size()) # [b, c*512] [b, c*128]
            if is_train:
                target_dict['rois'] = batch_dict['rois']
                target_dict['roi_scores'] = batch_dict['roi_scores']
                target_dict['roi_labels'] = batch_dict['roi_labels']
                target_dict['gt_of_rois'] = torch.cat((batch_dict['mv_roi_targets_dict'][0]['gt_of_rois'],
                                                       batch_dict['mv_roi_targets_dict'][1]['gt_of_rois'],
                                                       batch_dict['mv_roi_targets_dict'][2]['gt_of_rois']), dim=1)
                target_dict['gt_iou_of_rois'] = torch.cat((batch_dict['mv_roi_targets_dict'][0]['gt_iou_of_rois'],
                                                        batch_dict['mv_roi_targets_dict'][1]['gt_iou_of_rois'],
                                                        batch_dict['mv_roi_targets_dict'][2]['gt_iou_of_rois']), dim=1)
                target_dict['reg_valid_mask'] = torch.cat((batch_dict['mv_roi_targets_dict'][0]['reg_valid_mask'],
                                                        batch_dict['mv_roi_targets_dict'][1]['reg_valid_mask'],
                                                        batch_dict['mv_roi_targets_dict'][2]['reg_valid_mask']), dim=1)
                target_dict['rcnn_cls_labels'] = torch.cat((batch_dict['mv_roi_targets_dict'][0]['rcnn_cls_labels'],
                                                        batch_dict['mv_roi_targets_dict'][1]['rcnn_cls_labels'],
                                                        batch_dict['mv_roi_targets_dict'][2]['rcnn_cls_labels']), dim=1)
                target_dict['gt_of_rois_src'] = torch.cat((batch_dict['mv_roi_targets_dict'][0]['gt_of_rois_src'],
                                                        batch_dict['mv_roi_targets_dict'][1]['gt_of_rois_src'],
                                                        batch_dict['mv_roi_targets_dict'][2]['gt_of_rois_src']), dim=1)
                # print('gt_of_rois_size:', target_dict['gt_of_rois'].shape) # [b, c*128, 8]
                # print('gt_iou_of_rois_size:', target_dict['gt_iou_of_rois'].shape) # [b, c*128]
                # print('reg_valid_mask_size:', target_dict['reg_valid_mask'].shape) # [b, c*128]
                # print('rcnn_cls_labels_size:', target_dict['rcnn_cls_labels'].shape) # [b, c*128]
                # print('gt_of_rois_src_size:', target_dict['gt_of_rois_src'].shape) # [b, c*128, 8]
                batch_dict.pop('mv_roi_targets_dict', None)
        # print('@@@@@@@@@@@@@ batch_dict:', batch_dict)
        # a = {}
        # print("stop:", a[10])
        return batch_dict, target_dict

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        # print('batch_dict_before:', batch_dict)
        # batch_dict = self.proposal_layer(
        #     batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        # )
        # if self.training:
        #     targets_dict = batch_dict.get('roi_targets_dict', None)
        #     if targets_dict is None:
        #         targets_dict = self.assign_targets(batch_dict)
        #         batch_dict['rois'] = targets_dict['rois']
        #         batch_dict['roi_labels'] = targets_dict['roi_labels']

        # multi-view
        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1

        mv_rois = batch_dict.get('mv_rois', None) # for pv_plusplus
        if mv_rois is None:
            data_dict = {'batch_size':batch_dict['batch_size'],
                          'batch_cls_preds':batch_dict['batch_mv_cls_preds'][0],
                          'batch_box_preds':batch_dict['batch_mv_box_preds'][0]}
            if mv_status == 1:
                a1_data_dict = {'batch_size': batch_dict['batch_size'],
                                'batch_cls_preds': batch_dict['batch_mv_cls_preds'][1],
                                'batch_box_preds': batch_dict['batch_mv_box_preds'][1]}
                a2_data_dict = {'batch_size': batch_dict['batch_size'],
                                'batch_cls_preds': batch_dict['batch_mv_cls_preds'][2],
                                'batch_box_preds': batch_dict['batch_mv_box_preds'][2]}
                a_data_dict = [a1_data_dict, a2_data_dict]

            data_dict = self.proposal_layer(
                data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
            # print('rois_size:', data_dict['rois'].size()) # PV [b, 512, 7]
            # print('roi_cls_preds_size:', data_dict['roi_cls_preds'].size()) # PV [b, 512, 3]
            # print('roi_scores_size:', data_dict['roi_scores'].size()) # PV [b, 512]
            # print('roi_labels_size:', data_dict['roi_labels'].size()) # PV [b, 512]
            # a1 and a2 train 512 test 100
            if mv_status == 1:
                for i in range(2):
                    a_data_dict[i] = self.proposal_layer(
                        a_data_dict[i], nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
                    # print('rois_size:', a_data_dict[i]['rois'].size())  # PV [b, 512, 7]
                    # print('roi_cls_preds_size:', a_data_dict[i]['roi_cls_preds'].size())  # PV [b, 512, 3]
                    # print('roi_scores_size:', a_data_dict[i]['roi_scores'].size())  # PV [b, 512]
                    # print('roi_labels_size:', a_data_dict[i]['roi_labels'].size())  # PV [b, 512]
            # size
            single_view_roi_size = data_dict['rois'].shape[1]
            batch_dict['svr_size'] = single_view_roi_size  # train 512 test 100
            batch_dict['mv_status'] = mv_status
            # cat
            batch_dict = self.data_cat(batch_dict, data_dict, a_data_dict)

            if self.training:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                # print('targets_dict_roi:', targets_dict['rois'].size()) # [b, c*128, 7]
                # print('targets_dict_gt_of_rois:', targets_dict['gt_of_rois'].size()) # [b, c*128, 8]
                # print('targets_dict_gt_iou_of_rois:', targets_dict['gt_iou_of_rois'].size()) # [b, c*128]
                # print('targets_dict_roi_scores:', targets_dict['roi_scores'].size()) # [b, c*128]
                # print('targets_dict_roi_labels:', targets_dict['roi_labels'].size()) # [b, c*128]
                # print('targets_dict_reg_valid_mask:', targets_dict['reg_valid_mask'].size()) # [b, c*128]
                # print('targets_dict_rcnn_cls_labels:', targets_dict['rcnn_cls_labels'].size()) # [b, c*128]
                # print('targets_dict_gt_of_rois_src:', targets_dict['gt_of_rois_src'].size()) # [b, c*128, 8]
        else:
            # cat
            batch_dict['mv_status'] = mv_status
            targets_dict = {}
            batch_dict, targets_dict = self.data_cat_plusplus(batch_dict, self.training, targets_dict)
        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        # print('pooled_features_size:', pooled_features.size()) # [b*c*128, 6x6x6, 128]

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
