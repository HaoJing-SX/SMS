import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class HeightCompressionMultiView(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        if ('MV_CLASS' in self.model_cfg):
            if ('a1' in self.model_cfg['MV_CLASS']):
                encoded_spconv_tensor = batch_dict['encoded_spconv_tensor_a1']
                spatial_features = encoded_spconv_tensor.dense()
                N, C, D, H, W = spatial_features.shape
                spatial_features = spatial_features.view(N, C * D, H, W)
                batch_dict['spatial_features_a1'] = spatial_features
            if ('a2' in self.model_cfg['MV_CLASS']):
                encoded_spconv_tensor = batch_dict['encoded_spconv_tensor_a2']
                spatial_features = encoded_spconv_tensor.dense()
                N, C, D, H, W = spatial_features.shape
                spatial_features = spatial_features.view(N, C * D, H, W)
                batch_dict['spatial_features_a2'] = spatial_features

        # print('batch_dict:', batch_dict)
        # print('spatial_features_size:', batch_dict['spatial_features'].shape) # [b, 128*2, 200, 176]
        # print('spatial_features_a1_size:', batch_dict['spatial_features_a1'].shape) # [b, 128*2, 200, 176]
        # print('spatial_features_a2_size:', batch_dict['spatial_features_a2'].shape) # [b, 128*2, 200, 176]
        return batch_dict
