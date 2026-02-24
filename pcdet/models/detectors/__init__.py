from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN, MVPointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN, MVPVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus, MVPVRCNNPlusPlus
from .pointrcnn_single import PointRCNN_SINGLE

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'MVPVRCNN': MVPVRCNN,
    'PointPillar': PointPillar,
    'MVPointRCNN': MVPointRCNN,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MVPVRCNNPlusPlus': MVPVRCNNPlusPlus,
    'PointRCNN_SINGLE':PointRCNN_SINGLE,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
