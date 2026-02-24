from .detector3d_template import Detector3DTemplate

class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # print('batch_dict:', batch_dict)
        # PointNet2MSG：add point_features，point_coords
        # PointHeadBox：add point_cls_scores，batch_cls_preds, batch_box_preds, batch_index, cls_preds_normalized = False
        # PointRCNNHead：add rois，roi_scores, roi_labels, has_class_labels = True
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            # print('batch_dict:', batch_dict)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict

        else:

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict


class MVPointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # print('batch_dict:', batch_dict)
        # PointNet2MSG：add point_features，point_coords
        # PointHeadBox：add point_cls_scores，batch_cls_preds, batch_box_preds, batch_index, cls_preds_normalized = False
        # PointRCNNHead：add rois，roi_scores, roi_labels, has_class_labels = True
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            # print('batch_dict:', batch_dict)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict

        else:

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_consis, tb_dict = self.point_head.get_consis_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn + loss_consis
        # print('tb_dict:', tb_dict)
        return loss, tb_dict, disp_dict