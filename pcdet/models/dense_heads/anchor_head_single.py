import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        # print('stop:', data_dict['stop'])

        return data_dict


class AnchorHeadSingleMultiview(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_loss(self):
        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1
        mv_cls_loss, tb_dict = self.get_cls_layer_loss_multiview(mv_status)
        mv_box_loss, tb_dict_box = self.get_box_reg_layer_loss_multiview(mv_status)
        tb_dict.update(tb_dict_box)
        rpn_loss = mv_cls_loss + mv_box_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        # print('rpn_loss_log:', tb_dict)
        return rpn_loss, tb_dict

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

    def forward(self, data_dict):
        # # single-view
        # spatial_features_2d = data_dict['spatial_features_2d']
        # cls_preds = self.conv_cls(spatial_features_2d)
        # box_preds = self.conv_box(spatial_features_2d)
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        # self.forward_ret_dict['cls_preds'] = cls_preds
        # self.forward_ret_dict['box_preds'] = box_preds
        # if self.conv_dir_cls is not None:
        #     dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
        #     dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        #     self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        # else:
        #     dir_cls_preds = None
        # # print('cls_preds:', self.forward_ret_dict['cls_preds'].shape) # [b, 200, 176, 18]
        # # print('box_preds:', self.forward_ret_dict['box_preds'].shape) # [b, 200, 176, 42]
        # # print('dir_cls_preds:', self.forward_ret_dict['dir_cls_preds'].shape) # [b, 200, 176, 12]
        # if self.training:
        #     targets_dict = self.assign_targets(
        #         gt_boxes=data_dict['gt_boxes']
        #     )
        #     self.forward_ret_dict.update(targets_dict)
        # # print('box_cls_labels_size:', self.forward_ret_dict['box_cls_labels'].shape)
        # # print('box_reg_targets_size:', self.forward_ret_dict['box_reg_targets'].shape)
        # # print('reg_weights_size:', self.forward_ret_dict['reg_weights'].shape)
        # # print('forward_ret_dict:', self.forward_ret_dict)
        # if not self.training or self.predict_boxes_when_training:
        #     batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
        #         batch_size=data_dict['batch_size'],
        #         cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        #     )
        #     data_dict['batch_cls_preds'] = batch_cls_preds
        #     data_dict['batch_box_preds'] = batch_box_preds
        #     data_dict['cls_preds_normalized'] = False
        # print('batch_cls_preds:', data_dict['batch_cls_preds'].shape) # [2, 211200, 3]
        # print('batch_box_preds:', data_dict['batch_box_preds'].shape) # [2, 211200, 7]

        # mv
        mv_status = 1 # a1 and a2
        if not ('a1' in self.model_cfg['MV_CLASS']):
            mv_status = 3 # only a2
        if not ('a2' in self.model_cfg['MV_CLASS']):
            mv_status = 2  # only a1
        mv_spatial_features_2d = []
        if mv_status == 1:
            mv_spatial_features_2d = [data_dict['spatial_features_2d'], data_dict['spatial_features_2d_a1'],
                                      data_dict['spatial_features_2d_a2']]
        elif mv_status == 2:
            mv_spatial_features_2d = [data_dict['spatial_features_2d'], data_dict['spatial_features_2d_a1']]
        elif mv_status == 3:
            mv_spatial_features_2d = [data_dict['spatial_features_2d'], data_dict['spatial_features_2d_a2']]
        mv_cls_preds = []
        mv_box_preds = []
        mv_dir_cls_preds = []
        for i in range(len(mv_spatial_features_2d)):
            cls_preds = self.conv_cls(mv_spatial_features_2d[i])
            box_preds = self.conv_box(mv_spatial_features_2d[i])
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            mv_cls_preds.append(cls_preds)
            mv_box_preds.append(box_preds)
            if self.conv_dir_cls is not None:
                dir_cls_preds = self.conv_dir_cls(mv_spatial_features_2d[i])
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                mv_dir_cls_preds.append(dir_cls_preds)
            else:
                mv_dir_cls_preds = None
        self.forward_ret_dict['mv_cls_preds'] = mv_cls_preds.copy()
        self.forward_ret_dict['mv_box_preds'] = mv_box_preds.copy()
        if self.conv_dir_cls is not None:
            self.forward_ret_dict['mv_dir_cls_preds'] = mv_dir_cls_preds.copy()
        # print('mv_cls_preds0_size:', mv_cls_preds[0].shape) # [2, 200, 176, 18]
        # print('mv_box_preds0_size:', mv_box_preds[0].shape) # [2, 200, 176, 42]
        # print('mv_dir_cls_preds0_size:', mv_dir_cls_preds[0].shape) # [2, 200, 176, 12]
        if self.training:
            mid_mv_num = 3
            if mv_status == 2 and mv_status == 3:
                mid_mv_num = 2
            targets_dict = self.assign_targets_multiview(
                gt_boxes=data_dict['gt_boxes'], mv_num=mid_mv_num
            )
            self.forward_ret_dict.update(targets_dict)
        # print('mv_cls_preds_len:', len(self.forward_ret_dict['mv_cls_preds'])) # 3
        # print('mv_box_preds_len:', len(self.forward_ret_dict['mv_box_preds'])) # 3
        # print('mv_dir_cls_preds_len:', len(self.forward_ret_dict['mv_dir_cls_preds'])) # 3
        # print('mv_cls_preds0_size:', self.forward_ret_dict['mv_cls_preds'][0].shape) # [2, 200, 176, 18]
        # print('mv_box_preds0_size:', self.forward_ret_dict['mv_box_preds'][0].shape) # [2, 200, 176, 42]
        # print('mv_dir_cls_preds0_size:', self.forward_ret_dict['mv_dir_cls_preds'][0].shape) # [2, 200, 176, 12]

        # print('mv_box_cls_labels0_size:', self.forward_ret_dict['mv_box_cls_labels'][0].shape) # [2, 211200]
        # print('mv_box_reg_targets_size:', self.forward_ret_dict['mv_box_reg_targets'][0].shape) # [2, 211200, 7]
        # print('mv_reg_weights_size:', self.forward_ret_dict['mv_reg_weights'][0].shape) # [2, 211200]
        # print('mv_box_cls_labels_len:', len(self.forward_ret_dict['mv_box_cls_labels'])) # 3
        # print('mv_box_reg_targets_len:', len(self.forward_ret_dict['mv_box_reg_targets'])) # 3
        # print('mv_reg_weights_len:', len(self.forward_ret_dict['mv_reg_weights'])) # 3

        # print('mv_box_cls_labels0_01sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][0] == -1).sum()) #
        # print('mv_box_cls_labels0_0sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][0] == 0).sum())  #
        # print('mv_box_cls_labels0_1sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][0] == 1).sum())  #
        # print('mv_box_cls_labels0_2sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][0] == 2).sum())  #
        # print('mv_box_cls_labels0_3sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][0] == 3).sum())  #
        # print('mv_box_cls_labels1_01sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][1] == -1).sum()) #
        # print('mv_box_cls_labels1_0sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][1] == 0).sum())  #
        # print('mv_box_cls_labels1_1sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][1] == 1).sum())  #
        # print('mv_box_cls_labels1_2sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][1] == 2).sum())  #
        # print('mv_box_cls_labels1_3sum:', (self.forward_ret_dict['mv_box_cls_labels'][0][1] == 3).sum())  #
        # print('stop:', data_dict['stop'])

        if not self.training or self.predict_boxes_when_training:
            batch_mv_cls_preds = []
            batch_mv_box_preds = []
            for i in range(len(mv_cls_preds)):
                mid_batch_cls_preds, mid_batch_box_preds = self.generate_predicted_boxes(
                    batch_size=data_dict['batch_size'],
                    cls_preds=mv_cls_preds[i], box_preds=mv_box_preds[i], dir_cls_preds=mv_dir_cls_preds[i]
                )
                batch_mv_cls_preds.append(mid_batch_cls_preds)
                batch_mv_box_preds.append(mid_batch_box_preds)
            data_dict['batch_mv_cls_preds'] = batch_mv_cls_preds
            data_dict['batch_mv_box_preds'] = batch_mv_box_preds
            data_dict['cls_preds_normalized'] = False
        # print('batch_mv_cls_preds_size:', data_dict['batch_mv_cls_preds'][0].shape) # [2, 211200, 3]
        # print('batch_mv_box_preds_size:', data_dict['batch_mv_box_preds'][0].shape) # [2, 211200, 7]
        # print('batch_mv_cls_preds_len:', len(data_dict['batch_mv_cls_preds'])) # 3
        # print('batch_mv_box_preds_len:', len(data_dict['batch_mv_box_preds'])) # 3
        # print('stop:', data_dict['stop'])

        if self.training:
            mv_num = 3
            if mv_status == 2 and mv_status == 3:
                mv_num = 2
            targets_dict = self.assign_common_targets(
                data_dict=data_dict, targets_dict=targets_dict, mv_num=mv_num
            )
            # to forward_ret_dict
            self.forward_ret_dict['consis_cls_preds'] = targets_dict['consis_cls_preds']
            self.forward_ret_dict['consis_box_preds'] = targets_dict['consis_box_preds']
            self.forward_ret_dict['consis_cls_labels'] = targets_dict['consis_cls_labels']
            self.forward_ret_dict['consis_box_labels'] = targets_dict['consis_box_labels']
            self.forward_ret_dict['common_sum'] = targets_dict['common_sum']
        # print('stop:', data_dict['stop'])

        return data_dict
