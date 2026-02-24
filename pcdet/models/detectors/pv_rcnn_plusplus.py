from .detector3d_template import Detector3DTemplate


class PVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            # a = {}
            # print("stop:", a[10])
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict


class MVPVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            if ('MV_CLASS' in self.roi_head.model_cfg):
                mv_status = 1  # a1 and a2
                if not ('a1' in self.roi_head.model_cfg['MV_CLASS']):
                    mv_status = 3  # only a2
                if not ('a2' in self.roi_head.model_cfg['MV_CLASS']):
                    mv_status = 2  # only a1
                if mv_status == 1:
                    circle_num = 3
                elif mv_status == 2 or mv_status == 3:
                    circle_num = 2
                mv_rois = []
                mv_roi_labels = []
                mv_roi_scores = []
                mv_roi_targets_dict = []
                for i in range(circle_num):
                    batch_dict['rois'] = batch_dict['mv_rois'][i]
                    batch_dict['roi_scores'] = batch_dict['mv_roi_scores'][i]
                    batch_dict['roi_labels'] = batch_dict['mv_roi_labels'][i]
                    targets_dict = self.roi_head.assign_targets(batch_dict)
                    mv_rois.append(targets_dict['rois'])
                    mv_roi_labels.append(targets_dict['roi_labels'])
                    mv_roi_scores.append(targets_dict['roi_scores'])
                    mv_roi_targets_dict.append(targets_dict)
                batch_dict.pop('mv_rois', None)
                batch_dict.pop('mv_roi_labels', None)
                batch_dict.pop('mv_roi_scores', None)
                batch_dict['mv_rois'] = mv_rois
                batch_dict['mv_roi_labels'] = mv_roi_labels
                batch_dict['mv_roi_scores'] = mv_roi_scores
                batch_dict['mv_roi_targets_dict'] = mv_roi_targets_dict
                # print('@@@@@@@@@@@@@@ mv_roi_targets_dict:', batch_dict['mv_roi_targets_dict'])
                # print('@@@@@@@@@@@@@@ batch_dict:', batch_dict)
                batch_dict.pop('rois', None)
                batch_dict.pop('roi_scores', None)
                batch_dict.pop('roi_labels', None)
            else:
                targets_dict = self.roi_head.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_targets_dict'] = targets_dict
                num_rois_per_scene = targets_dict['rois'].shape[1]
                # print('@@@@@@@@@@@ batch_dict:', batch_dict)
                # print('@@@@@@@@@@@ rois_size:', batch_dict['rois'].shape)
                # print('@@@@@@@@@@@ roi_labels_size:', batch_dict['roi_labels'].shape)
                if 'roi_valid_num' in batch_dict:
                    # print('@@@@@@@@@@@@ done1') # no
                    batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]
        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)
        # print('@@@@@@@@@@@@ batch_dict:', batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            # print('tb_dict:', tb_dict)
            # a = {}
            # print("stop:", a[10])
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # a = {}
            # print("stop:", a[10])
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_consis, tb_dict = self.dense_head.get_consis_loss(tb_dict)
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_consis + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
