import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils
from ...ops.iou3d_nms import iou3d_nms_utils
from .roi_head_template import RoIHeadTemplate

class PointRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in, # 512
            output_channels=self.num_class, # 1
            fc_list=self.model_cfg.CLS_FC # [256, 256]
        )
        # print('channel_in:', channel_in)
        # print('num_class:', self.num_class)
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in, # 512
            output_channels=self.box_coder.code_size * self.num_class, # 7
            fc_list=self.model_cfg.REG_FC # [256, 256]
        )
        # print('channel_in:', channel_in)
        # print('num_class:', self.box_coder.code_size * self.num_class)

        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
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

    def roipool3d_gpu(self, batch_dict):
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
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        rois = batch_dict['rois']  # (B, num_rois train128 test100, 7 + C)

        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert batch_cnt.min() == batch_cnt.max()

        point_scores = batch_dict['point_cls_scores'].detach()
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1)
        batch_points = point_coords.view(batch_size, -1, 3)
        # shape （batch * 16384, 130）--> （batch , 16384, 130）
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])

        with torch.no_grad():
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )
            # print('batch_points_size:', batch_points.size()) # [1, 16384, 3]
            # print('batch_point_features_size:', batch_point_features.size()) # [1, 16384, 130]
            # print('rois', rois.size()) # [1, 128, 7]
            # print('pooled_features', pooled_features.size()) # [1, 128, 512, 133]
            # print('pooled_empty_flag_size', pooled_empty_flag.size()) # [1, 128]
            # print('pooled_empty_flag', pooled_empty_flag)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
            # （batch, num_rois, num_sampled_points, 133） --> （batch * num_rois, num_sampled_points, 133）
            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )

            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
            # print('pooled_features_size:', pooled_features.size()) # [128, 512, 133]
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:

        """
        # print('done1')

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois'] # [b, 128, 7]
            batch_dict['roi_labels'] = targets_dict['roi_labels'] # [b, 128]

        pooled_features = self.roipool3d_gpu(batch_dict)
        # print('pooled_features_size:', pooled_features.size()) # [128, 512, 133]

        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        # (b * num_of_roi train128 test100, 128, num_sampled_points 512, 1)
        xyz_features = self.xyz_up_layer(xyz_input)
        # (b * num_of_roi train128 test100, 128, num_sampled_points 512, 1)
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3)
        # (b * num_of_roi train128 test100, 256, num_sampled_points 512, 1)
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        # (b * num_of_roi train128 test100, 128, num_sampled_points 512, 1)
        merged_features = self.merge_down_layer(merged_features)

        # print('xyz_input:', xyz_input.size())
        # print('xyz_features:', xyz_features.size())
        # print('point_features:', point_features.size())
        # print('merged_features:', merged_features.size())

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1] # (b * num_of_roi train128 test100, num_sampled_points 512, 1)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1) # (b * num_of_roi train128 test100, 1)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1) # (b * num_of_roi train128 test100, 7)

        # print('shared_features_size:', shared_features.size())
        # print('rcnn_cls_size:', rcnn_cls.size())
        # print('rcnn_reg_size:', rcnn_reg.size())

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds # [b, 100, 1]
            batch_dict['batch_box_preds'] = batch_box_preds # [b, 100, 7]
            batch_dict['cls_preds_normalized'] = False
            batch_dict.pop('batch_index', None)
            # print('batch_dict:', batch_dict)

        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            # print('rcnn_cls:', rcnn_cls)
            # print('rcnn_reg:', rcnn_reg)

            self.forward_ret_dict = targets_dict

        # print('batch_dict:', batch_dict)
        return batch_dict

class PointRCNNHeadMultiView(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.SA_roi_modules = nn.ModuleList()
        channel_in = input_channels

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        # self.num_prefix_channels = 3 + 1  # xyz + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_roi_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in, # 512
            output_channels=self.num_class, # 1
            fc_list=self.model_cfg.CLS_FC # [256, 256]
        )
        # print('channel_in:', channel_in)
        # print('num_class:', self.num_class)
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in, # 512
            output_channels=self.box_coder.code_size * self.num_class, # 7
            fc_list=self.model_cfg.REG_FC # [256, 256]
        )
        # print('channel_in:', channel_in)
        # print('num_class:', self.box_coder.code_size * self.num_class)

        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
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

    def roipool3d_gpu(self, batch_dict):
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
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        rois = batch_dict['rois']

        # print('batch_index_size:', batch_idx.size()) # [b*c*16384]
        # print('point_coords_size:', point_coords.size()) # [b*c*16384, 4]
        # print('point_features_size:', point_features.size()) # [b*c*16384, 128]
        # print('rois_size:', rois.size()) # train [b, c*128, 7] test [b, c*100, 7]
        # print('roi_view_idx:', roi_view_idx.size()) # train [b, c*128] test [b, c*100]

        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert batch_cnt.min() == batch_cnt.max()

        point_scores = batch_dict['point_cls_scores'].detach()
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1)
        # print('point_features_all:', point_features_all.size())

        if self.model_cfg.FUSE_MV_FEATURE:
            batch_points = point_coords.view(batch_size, -1, 3)
            batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])

            with torch.no_grad():
                pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                    batch_points, batch_point_features, rois
                )
            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
            # （b, c * num_rois, num_sampled_points, 133） --> （b * c * num_rois, num_sampled_points, 133）
            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )

            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
            # print('pooled_features_size:', pooled_features.size()) # [b*c*128, 512, 133]
            # print('batch_dict_000:', batch_dict['abc'])
        else:
            mv_status = batch_dict['mv_status']
            roi_view_idx = batch_dict['roi_view_idx']

            if mv_status == 1:
                batch_points = point_coords.view(batch_size, 3, -1, 3)
                batch_point_features = point_features_all.view(batch_size, 3, -1, point_features_all.shape[-1])
                # print('batch_points_size:', batch_points.size())
                # print('batch_point_features_size:', batch_point_features.size())

                mid_pooled_features = []
                mid_pooled_empty_flag = []
                for bs_idx in range(batch_size):
                    for i in range(3):
                        mid_batch_points = batch_points[bs_idx][i].view(1, -1, 3)
                        mid_batch_point_features = batch_point_features[bs_idx][i].view(1, -1, point_features_all.shape[-1])
                        mid_view_mask = (roi_view_idx[bs_idx] == i)
                        mid_rois = rois[bs_idx][mid_view_mask].view(1, -1, 7)
                        # print('mid_batch_points:', mid_batch_points.size())
                        # print('mid_batch_point_features:', mid_batch_point_features.size())
                        # print('mid_view_mask:', mid_view_mask.sum())
                        # print('mid_rois:', mid_rois.size())

                        if mid_view_mask.sum() > 0:
                            with torch.no_grad():
                                now_pooled_features, now_pooled_empty_flag = self.roipoint_pool3d_layer(
                                    mid_batch_points, mid_batch_point_features, mid_rois
                                )
                                mid_pooled_features.append(now_pooled_features)
                                mid_pooled_empty_flag.append(now_pooled_empty_flag)
                            # print('now_pooled_features:', now_pooled_features.size())
                            # print('now_pooled_empty_flag:', now_pooled_empty_flag.size())
                # print('mid_pooled_features', mid_pooled_features)
                # print('mid_pooled_empty_flag', mid_pooled_empty_flag)

                pooled_features = torch.cat(mid_pooled_features, dim=1)\
                    .view(batch_size, rois.shape[1], self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS, -1)
                pooled_empty_flag = torch.cat(mid_pooled_empty_flag, dim=1).view(batch_size, -1)
                # print('pooled_features', pooled_features.size()) # [b, c*128, 512, 133]
                # print('pooled_empty_flag_size', pooled_empty_flag.size()) # [b, c*128]

                # canonical transformation
                roi_center = rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                # （b, c * num_rois, num_sampled_points, 133） --> （b * c * num_rois, num_sampled_points, 133）
                pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
                pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                    pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
                )

                pooled_features[pooled_empty_flag.view(-1) > 0] = 0
                # print('pooled_features_size:', pooled_features.size()) # [b*c*128, 512, 133]
            else:
                batch_points = point_coords.view(batch_size, 2, -1, 3)
                batch_point_features = point_features_all.view(batch_size, 2, -1, point_features_all.shape[-1])
                # print('batch_points_size:', batch_points.size())
                # print('batch_point_features_size:', batch_point_features.size())

                mid_pooled_features = []
                mid_pooled_empty_flag = []
                for bs_idx in range(batch_size):
                    for i in range(2):
                        mid_batch_points = batch_points[bs_idx][i].view(1, -1, 3)
                        mid_batch_point_features = batch_point_features[bs_idx][i].view(1, -1, point_features_all.shape[-1])
                        mid_view_mask = (roi_view_idx[bs_idx] == i)
                        mid_rois = rois[bs_idx][mid_view_mask].view(1, -1, 7)
                        # print('mid_batch_points:', mid_batch_points.size())
                        # print('mid_batch_point_features:', mid_batch_point_features.size())
                        # print('mid_view_mask:', mid_view_mask.sum())
                        # print('mid_rois:', mid_rois.size())

                        if mid_view_mask.sum() > 0:
                            with torch.no_grad():
                                now_pooled_features, now_pooled_empty_flag = self.roipoint_pool3d_layer(
                                    mid_batch_points, mid_batch_point_features, mid_rois
                                )
                                mid_pooled_features.append(now_pooled_features)
                                mid_pooled_empty_flag.append(now_pooled_empty_flag)
                            # print('now_pooled_features:', now_pooled_features.size())
                            # print('now_pooled_empty_flag:', now_pooled_empty_flag.size())

                # print('mid_pooled_features', mid_pooled_features)
                # print('mid_pooled_empty_flag', mid_pooled_empty_flag)

                pooled_features = torch.cat(mid_pooled_features, dim=1) \
                    .view(batch_size, rois.shape[1], self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS, -1)
                pooled_empty_flag = torch.cat(mid_pooled_empty_flag, dim=1).view(batch_size, -1)
                # print('pooled_features', pooled_features.size()) # [b, c*128, 512, 133]
                # print('pooled_empty_flag_size', pooled_empty_flag.size()) # [b, c*128]

                # canonical transformation
                roi_center = rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                # （b, c * num_rois, num_sampled_points, 133） --> （b * c * num_rois, num_sampled_points, 133）
                pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
                pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                    pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
                )
                pooled_features[pooled_empty_flag.view(-1) > 0] = 0
                # print('pooled_features_size:', pooled_features.size()) # [b*c*128, 512, 133]
        return pooled_features

    @torch.no_grad()
    def data_cat(self, batch_dict, data_dict, a_data_dict):
        batch_size = batch_dict['batch_size']
        batch_index = batch_dict['batch_index']
        mv_status = batch_dict['mv_status']
        mid_index = []
        mid_point_coords = []
        mid_point_features = []
        mid_point_cls_scores = []
        mid_point_cls_preds = []
        mid_point_box_preds = []
        for index in range(batch_size):
            if mv_status == 1:
                batch_mask = (batch_index[0] == index)
                batch_a1_mask = (batch_index[1] == index)
                batch_a2_mask = (batch_index[2] == index)
                mid_index.append(torch.cat([batch_dict['batch_index'][0][batch_mask],
                                                   batch_dict['batch_index'][1][batch_a1_mask],
                                                   batch_dict['batch_index'][2][batch_a2_mask]]))
                mid_point_coords.append(torch.cat([batch_dict['point_coords'][batch_mask],
                                                   batch_dict['point_a1_coords'][batch_a1_mask],
                                                   batch_dict['point_a2_coords'][batch_a2_mask]]))
                mid_point_features.append(torch.cat([batch_dict['point_features'][batch_mask],
                                                   batch_dict['point_a1_features'][batch_a1_mask],
                                                   batch_dict['point_a2_features'][batch_a2_mask]]))
                mid_point_cls_scores.append(torch.cat([batch_dict['point_cls_scores'][0][batch_mask],
                                                   batch_dict['point_cls_scores'][1][batch_a1_mask],
                                                   batch_dict['point_cls_scores'][2][batch_a2_mask]]))
                mid_point_cls_preds.append(torch.cat([batch_dict['batch_cls_preds'][0][batch_mask],
                                                   batch_dict['batch_cls_preds'][1][batch_a1_mask],
                                                   batch_dict['batch_cls_preds'][2][batch_a2_mask]]))
                mid_point_box_preds.append(torch.cat([batch_dict['batch_box_preds'][0][batch_mask],
                                                   batch_dict['batch_box_preds'][1][batch_a1_mask],
                                                   batch_dict['batch_box_preds'][2][batch_a2_mask]]))
            elif mv_status == 2 or mv_status == 3:
                batch_mask = (batch_index[0] == index)
                batch_a_mask = (batch_index[1] == index)
                mid_index.append(torch.cat([batch_dict['batch_index'][0][batch_mask],
                                                   batch_dict['batch_index'][1][batch_a_mask]]))
                mid_point_cls_scores.append(torch.cat([batch_dict['point_cls_scores'][0][batch_mask],
                                                   batch_dict['point_cls_scores'][1][batch_a_mask]]))
                mid_point_cls_preds.append(torch.cat([batch_dict['batch_cls_preds'][0][batch_mask],
                                                   batch_dict['batch_cls_preds'][1][batch_a_mask]]))
                mid_point_box_preds.append(torch.cat([batch_dict['batch_box_preds'][0][batch_mask],
                                                   batch_dict['batch_box_preds'][1][batch_a_mask]]))
                if mv_status == 2:
                    mid_point_coords.append(torch.cat([batch_dict['point_coords'][batch_mask],
                                                       batch_dict['point_a1_coords'][batch_a_mask]]))
                    mid_point_features.append(torch.cat([batch_dict['point_features'][batch_mask],
                                                         batch_dict['point_a1_features'][batch_a_mask]]))
                elif mv_status == 3:
                    mid_point_coords.append(torch.cat([batch_dict['point_coords'][batch_mask],
                                                       batch_dict['point_a2_coords'][batch_a_mask]]))
                    mid_point_features.append(torch.cat([batch_dict['point_features'][batch_mask],
                                                         batch_dict['point_a2_features'][batch_a_mask]]))
        # cat
        batch_dict['batch_index'] = (torch.cat(mid_index))
        batch_dict['point_coords'] = (torch.cat(mid_point_coords))
        batch_dict['point_features'] = (torch.cat(mid_point_features))
        batch_dict['point_cls_scores'] = (torch.cat(mid_point_cls_scores))
        batch_dict['batch_cls_preds'] = (torch.cat(mid_point_cls_preds))
        batch_dict['batch_box_preds'] = (torch.cat(mid_point_box_preds))
        batch_dict['has_class_labels'] = data_dict['has_class_labels']
        if mv_status == 1:
            batch_dict['rois'] = torch.cat([data_dict['rois'], a_data_dict[0]['rois'], a_data_dict[1]['rois']], dim=1)
            batch_dict['roi_scores'] = torch.cat([data_dict['roi_scores'], a_data_dict[0]['roi_scores'],
                                                  a_data_dict[1]['roi_scores']], dim=1)
            batch_dict['roi_labels'] = torch.cat([data_dict['roi_labels'], a_data_dict[0]['roi_labels'],
                                                  a_data_dict[1]['roi_labels']], dim=1)
            # roi_view_idx
            roi_view_idx = batch_dict['rois'].new_zeros((batch_size, batch_dict['rois'].shape[1]), dtype=torch.long)
            # print('roi_view_idx:', roi_view_idx.size()) # train [b, c*512] test [b, c*100]
            for i in range(3):
                roi_view_idx[:, i * batch_dict['svr_size']:(i + 1) * batch_dict['svr_size']] = i
            batch_dict['roi_view_idx'] = roi_view_idx
        elif mv_status == 2 or mv_status == 3:
            batch_dict['rois'] = torch.cat([data_dict['rois'], a_data_dict['rois']], dim=1)
            batch_dict['roi_scores'] = torch.cat([data_dict['roi_scores'], a_data_dict['roi_scores']], dim=1)
            batch_dict['roi_labels'] = torch.cat([data_dict['roi_labels'], a_data_dict['roi_labels']], dim=1)
            # roi_view_idx
            roi_view_idx = batch_dict['rois'].new_zeros((batch_size, batch_dict['rois'].shape[1]), dtype=torch.long)
            # print('roi_view_idx:', roi_view_idx.size()) # train [b, c*512] test [b, c*100]
            for i in range(2):
                roi_view_idx[:, i * batch_dict['svr_size']:(i + 1) * batch_dict['svr_size']] = i
            batch_dict['roi_view_idx'] = roi_view_idx

        # print('batch_index_size:', batch_dict['batch_index'].size()) # [b*c*16384]
        # print('point_coords_size:', batch_dict['point_coords'].size()) # [b*c*16384, 4]
        # print('point_features_size:', batch_dict['point_features'].size()) # [b*c*16384, 128]
        # print('point_cls_scores_size:', batch_dict['point_cls_scores'].size()) # [b*c*16384]
        # print('batch_cls_preds_size:', batch_dict['batch_cls_preds'].size()) # [b*c*16384, 3]
        # print('batch_box_preds_size:', batch_dict['batch_box_preds'].size()) # [b*c*16384, 7]
        # print('rois_size:', batch_dict['rois'].size()) # [b, c*512, 7]
        # print('roi_scores_size:', batch_dict['roi_scores'].size()) # [b, c*512]
        # print('roi_labels_size:', batch_dict['roi_labels'].size()) # [b, c*512]
        # print('roi_view_idx_size:', batch_dict['roi_view_idx'].size()) # [b, c*512]

        # print('0:',batch_roi_ori_view_idx[:, 508:516]) # [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]
        # print('1:', batch_roi_ori_view_idx[:, 1020:1028]) # [[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 2, 2]]

        # batch_dict.pop('points', None)
        batch_dict.pop('points_a1', None)
        batch_dict.pop('points_a2', None)
        batch_dict.pop('gt_boxes_a1', None)
        batch_dict.pop('gt_boxes_a2', None)
        if mv_status == 1:
            batch_dict.pop('point_a1_features', None)
            batch_dict.pop('point_a1_coords', None)
            batch_dict.pop('point_a2_features', None)
            batch_dict.pop('point_a2_coords', None)
        elif mv_status == 2:
            batch_dict.pop('point_a1_features', None)
            batch_dict.pop('point_a1_coords', None)
        elif mv_status == 3:
            batch_dict.pop('point_a2_features', None)
            batch_dict.pop('point_a2_coords', None)
        return batch_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
        Returns:
        """
        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1
        data_dict = {'batch_size':batch_dict['batch_size'], 'batch_index':batch_dict['batch_index'][0],
                      'batch_cls_preds':batch_dict['batch_cls_preds'][0],
                      'batch_box_preds':batch_dict['batch_box_preds'][0]}
        if mv_status == 1:
            a1_data_dict = {'batch_size': batch_dict['batch_size'], 'batch_index': batch_dict['batch_index'][1],
                            'batch_cls_preds': batch_dict['batch_cls_preds'][1],
                            'batch_box_preds': batch_dict['batch_box_preds'][1]}
            a2_data_dict = {'batch_size': batch_dict['batch_size'], 'batch_index': batch_dict['batch_index'][2],
                            'batch_cls_preds': batch_dict['batch_cls_preds'][2],
                            'batch_box_preds': batch_dict['batch_box_preds'][2]}
            a_data_dict = [a1_data_dict, a2_data_dict]
        elif mv_status == 2 or mv_status == 3:
            a_data_dict = {'batch_size': batch_dict['batch_size'], 'batch_index': batch_dict['batch_index'][1],
                            'batch_cls_preds': batch_dict['batch_cls_preds'][1],
                            'batch_box_preds': batch_dict['batch_box_preds'][1]}

        # print('data_dict:', data_dict)
        # print('a_data_dict:', a_data_dict)
        data_dict = self.proposal_layer(
            data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        # print('data_dict:', data_dict)
        # print('data_dict_roi:', data_dict['rois'].size()) # [b, 512, 7]
        # print('data_dict_roi_labels:', data_dict['roi_labels'].size()) # [b, 512]

        # a1 and a2 train 512 test 100
        if mv_status == 1:
            for i in range(2):
                a_data_dict[i] = self.proposal_layer(
                    a_data_dict[i], nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        # print('a_data_dict:', a_data_dict)
        # print('a_data_dict_roi0:', a_data_dict[0]['rois'].size()) # [b, 512, 7]
        # print('a_data_dict_roi_labels0:', a_data_dict[0]['roi_labels'].size()) # [b, 512]
        # print('a_data_dict_roi0:', a_data_dict[1]['rois'].size()) # [b, 512, 7]
        # print('a_data_dict_roi_labels0:', a_data_dict[1]['roi_labels'].size()) # [b, 512]
        # only a1 train 512 test 100
        elif mv_status == 2 or mv_status == 3:
            a_data_dict = self.proposal_layer(
                a_data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        # print('a_data_dict:', a_data_dict)
        # print('a_data_dict_roi:', a_data_dict['rois'].size()) # [b, 512, 7]
        # print('a_data_dict_roi_labels:', a_data_dict['roi_labels'].size()) # [b, 512]

        # print('point_coords:', batch_dict['point_coords'].size()) # [b*16384, 4]
        # print('point_features:', batch_dict['point_features'].size()) # [b*16384, 128]
        # size
        batch_size = batch_dict['batch_size']
        single_view_roi_size = data_dict['rois'].shape[1]
        batch_dict['svr_size'] = single_view_roi_size  # train 512 test 100
        batch_dict['mv_status'] = mv_status

        # cat
        batch_dict = self.data_cat(batch_dict, data_dict, a_data_dict)
        # print('batch_dict:', batch_dict)
        # print('data_dict:', data_dict)
        # print('a_data_dict:', a_data_dict)

        # print('roi_view_idx:', batch_dict['roi_view_idx'].size())  # train [b, c*512] test [b, c*100]
        # ROI与GT的target assignment 128

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_view_idx'] = targets_dict['roi_view_idx']
            # print('targets_dict:', targets_dict)
            # print('targets_dict_roi:', targets_dict['rois'].size()) # [b, c*128, 7]
            # print('targets_dict_gt_of_rois:', targets_dict['gt_of_rois'].size()) # [b, c*128, 8]
            # print('targets_dict_gt_iou_of_rois:', targets_dict['gt_iou_of_rois'].size()) # [b, c*128]
            # print('targets_dict_roi_scores:', targets_dict['roi_scores'].size()) # [b, c*128]
            # print('targets_dict_roi_labels:', targets_dict['roi_labels'].size()) # [b, c*128]
            # print('targets_dict_reg_valid_mask:', targets_dict['reg_valid_mask'].size()) # [b, c*128]
            # print('targets_dict_rcnn_cls_labels:', targets_dict['rcnn_cls_labels'].size()) # [b, c*128]
            # print('targets_dict_gt_of_rois_src:', targets_dict['gt_of_rois_src'].size()) # [b, c*128, 8]
            # print('roi_view_idx:', targets_dict['roi_view_idx'].size()) # train [b, c*128]


        pooled_features = self.roipool3d_gpu(batch_dict)
        # print('pooled_features_size:', pooled_features.size()) # [b*c*128, 512, 133]

        # (b * num_of_roi train128 test100, 5, num_sampled_points 512, 1)   5
        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        # (b * num_of_roi train128 test100, 128, num_sampled_points 512, 1)
        xyz_features = self.xyz_up_layer(xyz_input)
        # (b * num_of_roi train128 test100, 128, num_sampled_points 512, 1)    128
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3)
        # (b * num_of_roi train128 test100, 256, num_sampled_points 512, 1)
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        # (b * num_of_roi train128 test100, 128, num_sampled_points 512, 1)
        merged_features = self.merge_down_layer(merged_features)

        # print('xyz_input:', xyz_input.size()) # [b*c*128, 5, 512, 1]
        # print('xyz_features:', xyz_features.size()) # [b*c*128, 128, 512, 1]
        # print('point_features:', point_features.size()) # [b*c*128, 128, 512, 1]
        # print('merged_features:', merged_features.size()) # [b*c*128, 128, 512, 1]

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_roi_modules)):
            li_xyz, li_features = self.SA_roi_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1] # (b * num_of_roi train128 test100, num_sampled_points 512, 1)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1) # (b * num_of_roi train128 test100, 1)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1) # (b * num_of_roi train128 test100, 7)

        # print('shared_features_size:', shared_features.size()) # [b*c*128, 512, 1]
        # print('rcnn_cls_size:', rcnn_cls.size()) # [b*c*128, 1]
        # print('rcnn_reg_size:', rcnn_reg.size()) # [b*c*128, 7]
        # print('done1')

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds # [b, 100, 1]
            batch_dict['batch_box_preds'] = batch_box_preds # [b, 100, 7]
            batch_dict['cls_preds_normalized'] = False
            batch_dict.pop('batch_index', None)
            # print('batch_cls_preds', batch_dict['batch_cls_preds'])
            # print('batch_box_preds', batch_dict['batch_box_preds'])
            # print('roi_labels', batch_dict['roi_labels'].size())
            # print('batch_dict:', batch_dict)
            # print('done1')

        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            # print('rcnn_cls:', rcnn_cls)
            # print('rcnn_reg:', rcnn_reg)
            targets_dict['epoch_num'] = batch_dict['epoch_num']
            targets_dict['mv_status'] = batch_dict['mv_status']

            self.forward_ret_dict = targets_dict
            # print('targets_dict:', targets_dict)

        # print('batch_dict:', batch_dict)
        # print('data_dict_000:', data_dict['abc'])
        return batch_dict
