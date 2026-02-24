import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch
import math
import os
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.augmentor import augmentor_utils

home_path = os.environ['HOME']
root_path = home_path + '/Documents/kitti'

def read_pcd_from_ply(fileName):
    f = open(fileName)
    for i in range(9):
        head = f.readline()
    points = []
    for line in f.readlines():
        line = line.strip('\n')
        oneline = line.split(' ')
        points.append([float(oneline[0]), float(oneline[1]), float(oneline[2])])
    points = np.array(points, dtype=np.float32)
    return points

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.ply':
            points = read_pcd_from_ply(self.sample_file_list[index])
            point_rp = np.zeros(points.shape[0], dtype=float).reshape(-1, 1)  # initialize 0
            # print('point_rp:', point_rp.shape)
            points = np.concatenate((points, point_rp), axis=1)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        # print('index:', index)
        # print('sample_file_list:', self.sample_file_list[index])
        print('input_dict_points:', input_dict['points'].shape)
        # print('input_dict_points_len:', len(input_dict['points'])) # 11W+

        data_dict = self.prepare_data(data_dict=input_dict)
        # print('datadict_points:', data_dict['points'])
        # print('datadict_points_len:', len(data_dict['points'])) # 16384

        # rot_angle = 0
        # input_dict['points'] = augmentor_utils.point_rotation(
        #     input_dict['points'], rot_range=round((rot_angle / 180) * math.pi, 8)
        # )

        # print('stop:', input_dict['a'])

        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    parser.add_argument('--result', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--start', type=str, default="")
    parser.add_argument('--item_name', type=str, default='ours_')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])

            # print('points:', len(data_dict['points']))
            # print('data_dict:', data_dict)

            load_data_to_gpu(data_dict)
            # print('data_dict:', data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # print('pred_dicts:', pred_dicts)
            # print('roi_labels:', data_dict['roi_labels'].size())

            # if data_dict.get('points', None) is not None:
            # points = data_dict['points']
            # points_np = data_dict['points'].cpu().numpy()
            # pts_depth = np.linalg.norm(points_np[:, 1:4], axis=1)
            # pts_middle_flag = (pts_depth < 40.0) & (pts_depth >= 15.0)
            # pts_near_flag = pts_depth < 15.0
            # far_choice = np.where((pts_near_flag == 0) & (pts_middle_flag == 0))[0]
            # mid_choice = np.where((pts_near_flag == 0) & (pts_middle_flag == 1))[0]
            # near_choice = np.where(pts_near_flag == 1)[0]
            # data_dict['points_far'] = points[far_choice]
            # data_dict['points_middle'] = points[mid_choice]
            # data_dict['points_near'] = points[near_choice]

            # limit_z = [-3, 1]
            # mask = common_utils.mask_points_by_ground(data_dict['points'], limit_z)
            # data_dict['points'] = data_dict['points'][mask]

            # points = data_dict['points']
            # range_z = [points[0][3], points[0][3]]
            # for i in range(1, len(points)):
            #     _, mid_x, mid_y, mid_z, _ = points[i]
            #     # z
            #     if mid_z > range_z[1]:
            #         range_z[1] = mid_z
            #     elif mid_z < range_z[0]:
            #         range_z[0] = mid_z
            #
            # limit_z[0] = range_z[0] + 0.3
            #
            # mask = common_utils.mask_points_by_ground(data_dict['points'], limit_z)
            # data_dict['points'] = data_dict['points'][mask]

            V.draw_scenes(
                points=data_dict['point_coords'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

            # # read result.pkl and show 3d
            # with open(args.result, 'rb') as file:
            #     result_list = pickle.load(file)
            # for result in result_list:
            #     frame_id = result['frame_id']
            #     if args.start != '' and float(frame_id) == float(args.start):
            #         # print("########################### {} ###################################".format(frame_id))
            #         pred_boxes = torch.tensor(result['boxes_lidar'])
            #         pred_scores = torch.tensor(result['score'])
            #         pred_labels = torch.tensor(np.array([{'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}[x] for x in result['name']]))

            # print("label:", pred_labels)
            # print("point_size", data_dict['points'][:, 1:4].shape)
            # print("data_dict:", data_dict)
            # a = torch.tensor([[])
            # b = torch.tensor([[]])
            # pred_boxes = torch.cat((pred_boxes, a, b), dim=0)
            # d = torch.tensor([1, 1])
            # pred_scores = torch.cat((pred_scores, d), dim=0)
            # pred_labels = torch.cat((pred_labels, d), dim=0)
            # for i in range(pred_boxes.shape[0]):
            #     print(i, " x:", pred_boxes[i])
            # print("scores:", pred_scores)
            # print("labels:", pred_labels)

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:4], ref_boxes=pred_boxes,
            #     ref_scores=pred_scores, ref_labels=pred_labels
            # )
            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')

if __name__ == '__main__':
    main()
