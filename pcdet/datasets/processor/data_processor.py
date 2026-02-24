from functools import partial

import numpy as np
import math
from skimage import transform

from ...utils import box_utils, common_utils
from ..augmentor import augmentor_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except: # use
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else: # use
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features # 4
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        # print('num_point_features', self.num_point_features)

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)
        # print('data_processor_queue:', self.data_processor_queue)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]

        return data_dict

    def copy_data(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.copy_data, config=config)
        # print('data_dict', data_dict)
        if ('a1' in config['MV_CLASS']):
            data_dict['points_a1'] = data_dict['points'].copy()
            if data_dict.get('gt_boxes', None) is not None:
                data_dict['gt_boxes_a1'] = data_dict['gt_boxes'].copy()
        if ('a2' in config['MV_CLASS']):
            data_dict['points_a2'] = data_dict['points'].copy()
            if data_dict.get('gt_boxes', None) is not None:
                data_dict['gt_boxes_a2'] = data_dict['gt_boxes'].copy()
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        data_dict['ori_points'] = data_dict['points'].copy()
        # print('data_ori_points_size:', data_dict['ori_points'].shape)
        high_density_rl = 10
        low_density_rs = 30
        low_density_rl = 40
        density_str = ['density:']

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]

        if ('MV_CLASS' in config):
            if ('a1' in config['MV_CLASS']):
                # a1
                points_a1 = data_dict['points_a1']
                radius = np.array(np.linalg.norm(points_a1[:, 0:2], axis=1))
                far_points = points_a1[radius > 40]
                # upsample 10%
                far_choice = np.arange(0, len(far_points), dtype=np.int32)
                extra_choice = np.random.choice(far_choice, int(len(far_points) * 0.1), replace=False)
                far_choice = np.concatenate((far_choice, extra_choice), axis=0)
                finall_points = [far_points[far_choice]]
                # print('finall_points:', finall_points[0].shape)
                # for i in range(10):
                #     rs = 4 * i
                #     rl = 4 * (i + 1)
                for i in range(8):
                    rs = 5 * i
                    rl = 5 * (i + 1)
                    mid_circle_flag = (radius > rs) & (radius <= rl)
                    mid_circle_points = points_a1[mid_circle_flag]
                    # print('circle_points_num:', i, len(mid_circle_points))
                    mid_area = round(math.pi * (rl ** 2 - rs ** 2), 3)
                    # print('mid_area:', i, mid_area)
                    mid_density = round(len(mid_circle_points) / mid_area, 3)
                    density_str.append(mid_density)

                    # if mid_density <= 5:
                    #     print('density:', mid_density, i, 'point_num:', len(mid_circle_points))
                    #     mid_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    #     for j in range(len(mid_circle_points)):
                    #         z = mid_circle_points[j][2]
                    #         if z >= -4 and z < -3.5:
                    #             mid_num[0] += 1
                    #         elif z >= -3.5 and z < -3:
                    #             mid_num[1] += 1
                    #         elif z >= -3 and z < -2.5:
                    #             mid_num[2] += 1
                    #         elif z >= -2.5 and z < -2:
                    #             mid_num[3] += 1
                    #         elif z >= -2 and z < -1.5:
                    #             mid_num[4] += 1
                    #         elif z >= -1.5 and z < -1:
                    #             mid_num[5] += 1
                    #         elif z >= -1 and z < -0.5:
                    #             mid_num[6] += 1
                    #         elif z >= -0.5 and z < 0:
                    #             mid_num[7] += 1
                    #         elif z >= 0 and z < 0.5:
                    #             mid_num[8] += 1
                    #         elif z >= 0.5 and z < 1:
                    #             mid_num[9] += 1
                    #         elif z >= 1 and z < 1.5:
                    #             mid_num[10] += 1
                    #         elif z >= 1.5 and z < 2:
                    #             mid_num[11] += 1
                    #     print('z_num:', mid_num, 'k_sum:', np.sum(mid_num[5:9]))

                    if mid_density <= 5 and rs != 0:
                        # upsample 10%
                        # mid_choice = np.arange(0, len(mid_circle_points), dtype=np.int32)
                        # extra_choice = np.random.choice(mid_choice, int(len(mid_circle_points) * 0.1), replace=False)
                        # mid_choice = np.concatenate((mid_choice, extra_choice), axis=0)
                        # mid_circle_points = mid_circle_points[mid_choice]

                        # # key_point upsample 15%
                        key_point_flag = (mid_circle_points[:, 2] >= -1.5) & (mid_circle_points[:, 2] < 0.5) # -1.5<=z<0.5
                        key_points = mid_circle_points[key_point_flag]
                        if len(key_points) >= 15:
                            mid_choice = np.arange(0, len(key_points), dtype=np.int32)
                            extra_choice = np.random.choice(mid_choice, int(len(key_points) * 0.15), replace=False)
                            extra_points =  key_points[extra_choice]
                            # extra_points[:, 0] = np.round(extra_points[:, 0] - 0.01, 3) # x平移0.01
                            mid_circle_points = np.concatenate((mid_circle_points, extra_points), axis=0)

                        if rs < low_density_rs:
                            low_density_rs = rs
                    elif mid_density <= 5 and rs == 0:
                        mid_choice = np.arange(0, len(mid_circle_points), dtype=np.int32)
                        mid_circle_points = mid_circle_points[mid_choice]
                    elif mid_density > 8 and mid_density <= 15:
                        # downsample 10%
                        mid_choice = np.arange(0, len(mid_circle_points), dtype=np.int32)
                        mid_choice = np.random.choice(mid_choice, int(len(mid_circle_points) * 0.9), replace=False)
                        mid_circle_points = mid_circle_points[mid_choice]
                        if rl > high_density_rl:
                            high_density_rl = rl
                    elif mid_density > 15:
                        # downsample 15%
                        mid_choice = np.arange(0, len(mid_circle_points), dtype=np.int32)
                        mid_choice = np.random.choice(mid_choice, int(len(mid_circle_points) * 0.85), replace=False)
                        mid_circle_points = mid_circle_points[mid_choice]
                        if rl > high_density_rl:
                            high_density_rl = rl
                    finall_points.append(mid_circle_points)
                points_a1 = np.concatenate(finall_points, axis=0)

                if num_points < len(points_a1):
                    pts_depth = np.linalg.norm(points_a1[:, 0:3], axis=1)
                    pts_near_flag = pts_depth < 40.0
                    far_idxs_choice = np.where(pts_near_flag == 0)[0]
                    near_idxs = np.where(pts_near_flag == 1)[0]
                    choice = []
                    if num_points > len(far_idxs_choice):
                        near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                            if len(far_idxs_choice) > 0 else near_idxs_choice
                    else:
                        choice = np.arange(0, len(points_a1), dtype=np.int32)
                        choice = np.random.choice(choice, num_points, replace=False)
                    np.random.shuffle(choice)
                else:
                    choice = np.arange(0, len(points_a1), dtype=np.int32)
                    if num_points > len(points_a1):
                        extra_choice = np.random.choice(choice, num_points - len(points_a1), replace=False)
                        choice = np.concatenate((choice, extra_choice), axis=0)
                    np.random.shuffle(choice)
                    # print('done2')
                data_dict['points_a1'] = points_a1[choice]
                # print('point_a1_size:', len(data_dict['points_a1']))

            if ('a2' in config['MV_CLASS']):
                # a2
                # local grid downsampling ground points
                # x (0,35) interval 5, y (-35,35) interval 10
                points_a2 = data_dict['points_a2']
                # print('a2:', points_a2.shape)
                limit_range = self.point_cloud_range
                mask = (points_a2[:, 0] >= limit_range[0]) & (points_a2[:, 0] <= limit_range[3]) \
                       & (points_a2[:, 1] >= limit_range[1]) & (points_a2[:, 1] <= limit_range[4]) \
                       & (points_a2[:, 2] >= limit_range[2]) & (points_a2[:, 2] <= limit_range[5])
                points_a2 = points_a2[mask]
                points_a2_idxs = np.arange(0, len(points_a2), dtype=np.int32)
                grid_label = np.zeros(len(points_a2))
                # print('a2:', points_a2.shape)
                point_in_grid_flag = (points_a2[:, 0] > 0) & (points_a2[:, 0] < 35.0) \
                                     & (points_a2[:, 1] > -35.0) & (points_a2[:, 1] < 35.0)
                grid_points_a2 = points_a2[point_in_grid_flag]
                # print('a2_grid:', grid_points_a2.shape)
                no_grid_points_a2 = points_a2[~point_in_grid_flag]
                # print('a2_no_grid:', no_grid_points_a2.shape)

                # for i in range(7):
                #     for j in range(14):
                #         mid_grid_flag = (grid_points_a2[:, 0] > (5 * i)) & (grid_points_a2[:, 0] <= (5 * (i + 1))) \
                #                         & (grid_points_a2[:, 1] > (-35 + 5 * j)) \
                #                         & (grid_points_a2[:, 1] <= (-35 + 5 * (j + 1)))
                #         a = len(points_a2[point_in_grid_flag][mid_grid_flag])
                #         mid_grid_idxs = points_a2_idxs[point_in_grid_flag][mid_grid_flag]
                #         grid_label[mid_grid_idxs] = 14 * i + j + 1
                # finall_points = [no_grid_points_a2]
                # for i in range(98):
                #     mid_grid_points = np.array(sorted(points_a2[grid_label == (i + 1)], key=lambda x:x[2]))
                #     if len(mid_grid_points) > 100:
                #         # if (i >= 2 & i <= 11) and (i >= 16 & i <= 25) and (i >= 31 & i <= 38) and \
                #         #     (i >= 46 & i <= 51) and (i >= 61 & i <= 64):
                #         #     mid_thresh = mid_grid_points[0][2] + 0.15
                #         # else:
                #         mid_thresh = mid_grid_points[0][2] + 0.15
                #         mid_grid_no_ground_flag = mid_grid_points[:, 2] > mid_thresh
                #         mid_grid_points = mid_grid_points[mid_grid_no_ground_flag]
                #         # print('grid_points_size:', i, len(mid_grid_points))
                #     if len(mid_grid_points) > 0:
                #         finall_points.append(mid_grid_points)
                # points_a2 = np.concatenate(finall_points, axis=0)

                for i in range(7):
                    for j in range(7):
                        mid_grid_flag = (grid_points_a2[:, 0] > (5 * i)) & (grid_points_a2[:, 0] <= (5 * (i + 1))) \
                                        & (grid_points_a2[:, 1] > (-35 + 10 * j)) \
                                        & (grid_points_a2[:, 1] <= (-35 + 10 * (j + 1)))
                        a = len(points_a2[point_in_grid_flag][mid_grid_flag])
                        # print((7 * i + j + 1), a)
                        mid_grid_idxs = points_a2_idxs[point_in_grid_flag][mid_grid_flag]
                        grid_label[mid_grid_idxs] = 7 * i + j + 1
                finall_points = [no_grid_points_a2]
                for i in range(49):
                    mid_grid_points = np.array(sorted(points_a2[grid_label == (i + 1)], key=lambda x:x[2]))
                    if len(mid_grid_points) > 100:
                        # if (i >= 1 & i <= 5) and (i >= 8 & i <= 12) and (i >= 15 & i <= 19) and \
                        #     (i >= 23 & i <= 25) and (i == 31):
                        #     mid_thresh = mid_grid_points[0][2] + 0.2
                        # else:
                        mid_thresh = mid_grid_points[0][2] + 0.2
                        mid_grid_no_ground_flag = mid_grid_points[:, 2] > mid_thresh
                        mid_grid_points = mid_grid_points[mid_grid_no_ground_flag]
                        # print('grid_points_size:', i, len(mid_grid_points))
                    if len(mid_grid_points) > 0:
                        finall_points.append(mid_grid_points)
                points_a2 = np.concatenate(finall_points, axis=0)

                if num_points < len(points_a2):
                    pts_depth = np.linalg.norm(points_a2[:, 0:3], axis=1)
                    pts_near_flag = pts_depth < 40.0
                    far_idxs_choice = np.where(pts_near_flag == 0)[0]
                    near_idxs = np.where(pts_near_flag == 1)[0]
                    choice = []
                    if num_points > len(far_idxs_choice):
                        near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                            if len(far_idxs_choice) > 0 else near_idxs_choice
                    else:
                        choice = np.arange(0, len(points_a2), dtype=np.int32)
                        choice = np.random.choice(choice, num_points, replace=False)
                    np.random.shuffle(choice)
                else:
                    choice = np.arange(0, len(points_a2), dtype=np.int32)
                    if num_points > len(points_a2):
                        extra_choice = np.random.choice(choice, num_points - len(points_a2), replace=True)
                        choice = np.concatenate((choice, extra_choice), axis=0)
                    np.random.shuffle(choice)
                data_dict['points_a2'] = points_a2[choice]

            mid_points = data_dict['ori_points']
            mid_radius = np.array(np.linalg.norm(mid_points[:, 0:2], axis=1))
            mid_hd_flag = (mid_radius > 0) & (mid_radius <= high_density_rl)
            mid_sd_flag = (mid_radius > low_density_rs) & (mid_radius <= low_density_rl)
            data_dict['hd_ori_points'] = mid_points[mid_hd_flag]
            data_dict['sd_ori_points'] = mid_points[mid_sd_flag]

            mid_points = data_dict['points']
            mid_radius = np.array(np.linalg.norm(mid_points[:, 0:2], axis=1))
            mid_hd_flag = (mid_radius > 0) & (mid_radius <= high_density_rl)
            mid_sd_flag = (mid_radius > low_density_rs) & (mid_radius <= low_density_rl)
            data_dict['hd_points'] = mid_points[mid_hd_flag]
            data_dict['sd_points'] = mid_points[mid_sd_flag]

            if ('a1' in config['MV_CLASS']):
                mid_points = data_dict['points_a1']
                mid_radius = np.array(np.linalg.norm(mid_points[:, 0:2], axis=1))
                mid_hd_flag = (mid_radius > 0) & (mid_radius <= high_density_rl)
                mid_sd_flag = (mid_radius > low_density_rs) & (mid_radius <= low_density_rl)
                data_dict['hd_points_a1'] = mid_points[mid_hd_flag]
                data_dict['sd_points_a1'] = mid_points[mid_sd_flag]

            if ('a2' in config['MV_CLASS']):
                mid_points = data_dict['points_a2']
                mid_radius = np.array(np.linalg.norm(mid_points[:, 0:2], axis=1))
                mid_hd_flag = (mid_radius > 0) & (mid_radius <= high_density_rl)
                mid_sd_flag = (mid_radius > low_density_rs) & (mid_radius <= low_density_rl)
                data_dict['hd_points_a2'] = mid_points[mid_hd_flag]
                data_dict['sd_points_a2'] = mid_points[mid_sd_flag]

            # print('point_a2_size:', len(data_dict['points_a2']))
            # print('points:', id(data_dict['points']))
            # print('points_a1:', id(data_dict['points_a1']))
            # print('points_a2:', id(data_dict['points_a2']))
            # print('gt_boxes:', id(data_dict['gt_boxes']))
            # print('gt_boxes_a1:', id(data_dict['gt_boxes_a1']))
            # print('gt_boxes_a2:', id(data_dict['gt_boxes_a2']))
            # print('done1')

            # print('high_density_rl:', high_density_rl)
            # print('low_density_rs:', low_density_rs)
            # print('density_str:', density_str)
            # print('ori_points_size:', data_dict['ori_points'].shape)
            # print('hd_ori_points_size:', data_dict['hd_ori_points'].shape)
            # print('sd_ori_points_size:', data_dict['sd_ori_points'].shape)
            # print('points_size:', data_dict['points'].shape)
            # print('hd_points_size:', data_dict['hd_points'].shape)
            # print('sd_points_size:', data_dict['sd_points'].shape)
            # print('points_a1_size:', data_dict['points_a1'].shape)
            # print('hd_points_a1_size:', data_dict['hd_points_a1'].shape)
            # print('sd_points_a1_size:', data_dict['sd_points_a1'].shape)
            # print('points_a2_size:', data_dict['points_a2'].shape)
            # print('hd_points_a2_size:', data_dict['hd_points_a2'].shape)
            # print('sd_points_a2_size:', data_dict['sd_points_a2'].shape)

        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

            if ('MV_CLASS' in config):
                if ('a1' in config['MV_CLASS']):
                    # a1
                    points = data_dict['points_a1']
                    shuffle_idx = np.random.permutation(points.shape[0])
                    points = points[shuffle_idx]
                    data_dict['points_a1'] = points
                    # data_dict['points_a1_ori'] = data_dict['points_a1'].copy()
                if ('a2' in config['MV_CLASS']):
                    # a2
                    points = data_dict['points_a2']
                    shuffle_idx = np.random.permutation(points.shape[0])
                    points = points[shuffle_idx]
                    data_dict['points_a2'] = points

        return data_dict

    def world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.world_rotation, config=config)
        rot_angle = config['WORLD_ROT_ANGLE']
        gt_boxes = [data_dict['gt_boxes_a1'], data_dict['gt_boxes_a2']]
        data_points = [data_dict['points_a1'], data_dict['points_a2']]
        gt_boxes, points = augmentor_utils.global_rotation2(gt_boxes, data_points, rot_angle=rot_angle)
        data_dict['gt_boxes_a1'] = gt_boxes[0]
        data_dict['gt_boxes_a2'] = gt_boxes[1]
        data_dict['points_a1'] = points[0]
        data_dict['points_a2'] = points[1]
        data_dict['rot_angle'] = rot_angle
        # print('points:', data_dict['points'])
        # print('points_a1:', data_dict['points_a1'])
        # print('points_a2:', data_dict['points_a2'])
        # print('gt_boxes:', data_dict['gt_boxes'])
        # print('gt_boxes_a1:', data_dict['gt_boxes_a1'])
        # print('gt_boxes_a2:', data_dict['gt_boxes_a2'])
        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        if ('MV_CLASS' in config):
            if ('a1' in config['MV_CLASS']):
                points = data_dict['points_a1']
                voxel_output = self.voxel_generator.generate(points)
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

                data_dict['voxels_a1'] = voxels
                data_dict['voxel_a1_coords'] = coordinates
                data_dict['voxel_a1_num_points'] = num_points
            if ('a2' in config['MV_CLASS']):
                points = data_dict['points_a2']
                voxel_output = self.voxel_generator.generate(points)
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

                data_dict['voxels_a2'] = voxels
                data_dict['voxel_a2_coords'] = coordinates
                data_dict['voxel_a2_num_points'] = num_points

            mv_status = 1  # a1 and a2
            if not ('a1' in config['MV_CLASS']):
                mv_status = 3  # only a2
                # print('a1:', True)
            if not ('a2' in config['MV_CLASS']):
                mv_status = 2  # only a1
                # print('a2:', True)
            # print('@@@@@@@@@@@@@ mv_status:', mv_status)

            if self.mode == 'train' and ('CONSIS_VOXEL_SAMPLING' in config):
                if config['CONSIS_VOXEL_SAMPLING'] == True:
                    point_range = self.point_cloud_range
                    voxel_size = config.VOXEL_SIZE
                    x_num = int((point_range[3] - point_range[0]) / voxel_size[0])
                    y_num = int((point_range[4] - point_range[1]) / voxel_size[1])
                    z_num = int((point_range[5] - point_range[2]) / voxel_size[2])
                    total_num = x_num * y_num * z_num
                    voxel_hash_index = np.arange(0, total_num, 1) #(0~90111999)
                    # print('voxel_hash_index_size:', voxel_hash_index.shape)
                    if mv_status == 1: # a1 and a2
                        voxel_index = [np.arange(0, len(data_dict['voxels']), 1), np.arange(0, len(data_dict['voxels_a1']), 1),
                                       np.arange(0, len(data_dict['voxels_a2']), 1)] #(0~n0-1, 0~n1-1, 0~n2-1)
                        coords = [np.array(data_dict['voxel_coords']), np.array(data_dict['voxel_a1_coords']),
                                  np.array(data_dict['voxel_a2_coords'])] #([nz, ny, nx], a1[nz, ny, nx], a2[nz, ny, nx])
                        voxel_coords_product = [coords[0][:, 0] + coords[0][:, 1] * z_num + coords[0][:, 2] * z_num * y_num,
                                                coords[1][:, 0] + coords[1][:, 1] * z_num + coords[1][:, 2] * z_num * y_num,
                                                coords[2][:, 0] + coords[2][:, 1] * z_num + coords[2][:, 2] * z_num * y_num
                                                ]#(hash_index, a1_hash_index, a2_hash_index)
                        voxel_hash_ori_bool = [(voxel_hash_index == -1), (voxel_hash_index == -1), (voxel_hash_index == -1)]
                        voxel_hash_ori_index = [np.ones(total_num, dtype=int) * -1, np.ones(total_num, dtype=int) * -1,
                                                np.ones(total_num, dtype=int) * -1]
                        voxel_hash_ori_marknum = [np.zeros(total_num, dtype=int), np.zeros(total_num, dtype=int),
                                                np.zeros(total_num, dtype=int)]
                        for i in range(3):
                            voxel_hash_ori_bool[i][voxel_coords_product[i]] = True
                            voxel_hash_ori_index[i][voxel_coords_product[i]] = voxel_index[i]
                            voxel_hash_ori_marknum[i][voxel_coords_product[i]] += 1
                        voxel_hash_common_bool = voxel_hash_ori_bool[0] & voxel_hash_ori_bool[1] & voxel_hash_ori_bool[2]
                        common_voxel_indexs= [voxel_hash_ori_index[0][voxel_hash_common_bool],
                                              voxel_hash_ori_index[1][voxel_hash_common_bool],
                                              voxel_hash_ori_index[2][voxel_hash_common_bool]]
                        common_voxels = [data_dict['voxels'][common_voxel_indexs[0]],
                                         data_dict['voxels_a1'][common_voxel_indexs[1]],
                                         data_dict['voxels_a2'][common_voxel_indexs[2]]]
                        common_voxel_coords = [data_dict['voxel_coords'][common_voxel_indexs[0]],
                                               data_dict['voxel_a1_coords'][common_voxel_indexs[1]],
                                               data_dict['voxel_a2_coords'][common_voxel_indexs[2]]]
                        common_voxel_num_points = [data_dict['voxel_num_points'][common_voxel_indexs[0]],
                                                   data_dict['voxel_a1_num_points'][common_voxel_indexs[1]],
                                                   data_dict['voxel_a2_num_points'][common_voxel_indexs[2]]]
                    elif mv_status == 2: # only a1
                        voxel_index = [np.arange(0, len(data_dict['voxels']), 1), np.arange(0, len(data_dict['voxels_a1']), 1)] #(0~n0-1, 0~n1-1)
                        coords = [np.array(data_dict['voxel_coords']), np.array(data_dict['voxel_a1_coords'])] #([nz, ny, nx], a1[nz, ny, nx])
                        voxel_coords_product = [coords[0][:, 0] + coords[0][:, 1] * z_num + coords[0][:, 2] * z_num * y_num,
                                                coords[1][:, 0] + coords[1][:, 1] * z_num + coords[1][:, 2] * z_num * y_num,
                                                ]#(hash_index, a1_hash_index)
                        voxel_hash_ori_bool = [(voxel_hash_index == -1), (voxel_hash_index == -1)]
                        voxel_hash_ori_index = [np.ones(total_num, dtype=int) * -1, np.ones(total_num, dtype=int) * -1]
                        for i in range(2):
                            voxel_hash_ori_bool[i][voxel_coords_product[i]] = True
                            voxel_hash_ori_index[i][voxel_coords_product[i]] = voxel_index[i]
                        voxel_hash_common_bool = voxel_hash_ori_bool[0] & voxel_hash_ori_bool[1]
                        common_voxel_indexs= [voxel_hash_ori_index[0][voxel_hash_common_bool],
                                              voxel_hash_ori_index[1][voxel_hash_common_bool]]
                        common_voxels = [data_dict['voxels'][common_voxel_indexs[0]],
                                         data_dict['voxels_a1'][common_voxel_indexs[1]]]
                        common_voxel_coords = [data_dict['voxel_coords'][common_voxel_indexs[0]],
                                               data_dict['voxel_a1_coords'][common_voxel_indexs[1]]]
                        common_voxel_num_points = [data_dict['voxel_num_points'][common_voxel_indexs[0]],
                                                   data_dict['voxel_a1_num_points'][common_voxel_indexs[1]]]
                    elif mv_status == 3: # # only a2
                        voxel_index = [np.arange(0, len(data_dict['voxels']), 1), np.arange(0, len(data_dict['voxels_a2']), 1)] #(0~n0-1, 0~n2-1)
                        coords = [np.array(data_dict['voxel_coords']), np.array(data_dict['voxel_a2_coords'])] #([nz, ny, nx], a2[nz, ny, nx])
                        voxel_coords_product = [coords[0][:, 0] + coords[0][:, 1] * z_num + coords[0][:, 2] * z_num * y_num,
                                                coords[1][:, 0] + coords[1][:, 1] * z_num + coords[1][:, 2] * z_num * y_num,
                                                ]#(hash_index, a2_hash_index)
                        voxel_hash_ori_bool = [(voxel_hash_index == -1), (voxel_hash_index == -1)]
                        voxel_hash_ori_index = [np.ones(total_num, dtype=int) * -1, np.ones(total_num, dtype=int) * -1]
                        # print('sum_bool:', np.sum(voxel_hash_ori_bool[0]), 'sum_a2_bool:', np.sum(voxel_hash_ori_bool[1]))
                        for i in range(2):
                            voxel_hash_ori_bool[i][voxel_coords_product[i]] = True
                            voxel_hash_ori_index[i][voxel_coords_product[i]] = voxel_index[i]
                        voxel_hash_common_bool = voxel_hash_ori_bool[0] & voxel_hash_ori_bool[1]
                        common_voxel_indexs= [voxel_hash_ori_index[0][voxel_hash_common_bool],
                                              voxel_hash_ori_index[1][voxel_hash_common_bool]]
                        common_voxels = [data_dict['voxels'][common_voxel_indexs[0]],
                                         data_dict['voxels_a2'][common_voxel_indexs[1]]]
                        common_voxel_coords = [data_dict['voxel_coords'][common_voxel_indexs[0]],
                                               data_dict['voxel_a2_coords'][common_voxel_indexs[1]]]
                        common_voxel_num_points = [data_dict['voxel_num_points'][common_voxel_indexs[0]],
                                                   data_dict['voxel_a2_num_points'][common_voxel_indexs[1]]]
                    # for i in range(10):
                        # print('common_voxels0:', i, ' ', common_voxels[0][i])
                        # print('common_voxels1:', i, ' ', common_voxels[1][i])
                        # print('common_voxels2:', i, ' ', common_voxels[2][i])
                    #     print('common_voxel_coords00:', common_voxel_coords[0][i])
                    #     print('common_voxel_coords10:', common_voxel_coords[1][i])
                    #     print('common_voxel_coords20:', common_voxel_coords[2][i])
                    # print('common_voxel_num_points00:', common_voxel_num_points[0][0])
                    # print('common_voxel_num_points10:', common_voxel_num_points[1][0])
                    # print('common_voxel_num_points20:', common_voxel_num_points[2][0])
                    # print('common_voxels0_len:', len(common_voxels[0]))
                    # print('common_voxels1_len:', len(common_voxels[1]))
                    # print('common_voxels2_len:', len(common_voxels[2]))

                    if ('VOXEL_TO_BEV8X' in config):
                        if config['VOXEL_TO_BEV8X'] == True:
                            # print('done_bev')
                            if mv_status == 1:  # a1 and a2
                                voxel_hash_add = voxel_hash_ori_marknum[0] + voxel_hash_ori_marknum[1] \
                                                 + voxel_hash_ori_marknum[2]
                                voxel_hash_no_common_bool = (voxel_hash_add == 1) + (voxel_hash_add == 2)
                                voxel_hash_common_bool = (voxel_hash_add == 3)
                                # print('voxel_hash_no_common_sum:', voxel_hash_no_common_bool.sum())  # a
                                # print('voxel_hash_common_sum:', voxel_hash_common_bool.sum())  # b
                                # print('voxel_hash_add_max:', voxel_hash_add.max())  # 3
                                voxel_hash_index_no_common = voxel_hash_index[voxel_hash_no_common_bool]
                                voxel_hash_index_common = voxel_hash_index[voxel_hash_common_bool]
                                # print('voxel_hash_index_no_common_size:', voxel_hash_index_no_common.shape)  # a
                                # print('voxel_hash_index_common_size:', voxel_hash_index_common.shape)  # b
                                # print('voxel_hash_index_no_common_max:', voxel_hash_index_no_common.max())
                                # print('voxel_hash_index_common_max:', voxel_hash_index_common.max())
                                voxel_hash_index_no_common_bev8x = (voxel_hash_index_no_common / z_num / 64).astype(int)
                                voxel_hash_index_common_bev8x = (voxel_hash_index_common / z_num / 64).astype(int)
                                # print('voxel_hash_index_no_common_bev8x_size:', voxel_hash_index_no_common_bev8x.shape)  # a
                                # print('voxel_hash_index_common_bev8x_size:', voxel_hash_index_common_bev8x.shape)  # b
                                # print('voxel_hash_index_no_common_bev8x_max:', voxel_hash_index_no_common_bev8x.max())
                                # print('voxel_hash_index_common_bev8x_max:', voxel_hash_index_common_bev8x.max())
                                bev8x_xnum = int(x_num / 8)
                                bev8x_ynum = int(y_num / 8)
                                bev8x_totalnum = bev8x_xnum * bev8x_ynum
                                # print('bev8x_xnum:', bev8x_xnum)  # 176
                                # print('bev8x_ynum:', bev8x_ynum)  # 200
                                # print('bev8x_totalnum:', bev8x_totalnum)  # 35200
                                bev8x_hash_index = np.arange(0, bev8x_totalnum, 1)  # (0~35200)
                                bev8x_hash_bool = (bev8x_hash_index == -1)
                                # bev8x_hash_marknum_sum = np.zeros(bev8x_totalnum, dtype=float)
                                # bev8x_hash_marknum_common = np.zeros(bev8x_totalnum, dtype=float)
                                # voxel_hash_index_no_common_bev8x.sort()
                                # voxel_hash_index_common_bev8x.sort()
                                # for i in range(len(voxel_hash_index_no_common_bev8x)):
                                #     bev8x_hash_marknum_sum[voxel_hash_index_no_common_bev8x[i]] += 1
                                # for i in range(len(voxel_hash_index_common_bev8x)):
                                #     bev8x_hash_marknum_sum[voxel_hash_index_common_bev8x[i]] += 1
                                #     bev8x_hash_marknum_common[voxel_hash_index_common_bev8x[i]] += 1

                                # print('bev8x_hash_marknum_sum_size:', bev8x_hash_marknum_sum.shape)  # 35200
                                # print('bev8x_hash_marknum_common_size:', bev8x_hash_marknum_common.shape)  # 35200
                                # print('bev8x_hash_marknum_sum_max:', bev8x_hash_marknum_sum.max())
                                # print('bev8x_hash_marknum_common_max:', bev8x_hash_marknum_common.max())

                                # bev8x_hash_marknum_sum += 0.001
                                # bev8x_hash_marknum_common_rate = (bev8x_hash_marknum_common / bev8x_hash_marknum_sum)

                                # for i in range(10):
                                #     rate = i * 0.1
                                #     print('rate:', rate)
                                #     print('bev8x_hash_marknum_common_rate_sum:', (bev8x_hash_marknum_common_rate >= rate).sum())
                                #     print('bev8x_hash_marknum_common_rate_max:',
                                #           bev8x_hash_marknum_common_rate.max())

                                # bev8x_hash_common_index = bev8x_hash_index[bev8x_hash_marknum_common_rate >= config['COMMON_RATE']]
                                # bev8x_hash_bool[bev8x_hash_common_index] = True
                                # data_dict['common_bev8x_hash_mask'] = bev8x_hash_bool

                                common_bev8x_coords = (common_voxel_coords[0][:, 1:3] / 8).astype(int)
                                bev8x_coords_product = common_bev8x_coords[:, 0] + common_bev8x_coords[:, 1] * bev8x_ynum
                                bev8x_hash_bool[bev8x_coords_product] = True
                                data_dict['common_bev8x_hash_mask'] = bev8x_hash_bool

                    if ('VOXEL_TO_POINT' in config):
                        if config['VOXEL_TO_POINT'] == True:
                            # print('done_point')
                            if mv_status == 1:  # a1 and a2
                                common_points = []
                                mid_points = data_dict['points']
                                mid_points_a1 = data_dict['points_a1']
                                mid_points_a2 = data_dict['points_a2']
                                mid_indexs = np.arange(0, config.NUM_POINTS, 1)
                                common_point_indexs = []
                                common_point_a1_indexs = []
                                common_point_a2_indexs = []
                                for i in range(len(common_voxels[0])):
                                # for i in range(1):
                                    now_voxel_coords = common_voxel_coords[0][i]
                                    limit_x = [now_voxel_coords[2] * (config.VOXEL_SIZE[0]),
                                               (now_voxel_coords[2] + 1) * (config.VOXEL_SIZE[0])]
                                    limit_y = [now_voxel_coords[1] * (config.VOXEL_SIZE[1]) - 40,
                                               (now_voxel_coords[1] + 1) * (config.VOXEL_SIZE[1]) - 40]
                                    limit_z = [now_voxel_coords[0] * (config.VOXEL_SIZE[2]) - 3,
                                               (now_voxel_coords[0] + 1) * (config.VOXEL_SIZE[2]) - 3]
                                    now_point_indexs = mid_indexs[(mid_points[:, 0] >= limit_x[0]) & (mid_points[:, 0] <= limit_x[1]) &
                                                                  (mid_points[:, 1] >= limit_y[0]) & (mid_points[:, 1] <= limit_y[1]) &
                                                                  (mid_points[:, 2] >= limit_z[0]) & (mid_points[:, 2] <= limit_z[1])]
                                    now_point_a1_indexs = mid_indexs[(mid_points_a1[:, 0] >= limit_x[0]) & (mid_points_a1[:, 0] <= limit_x[1]) &
                                                                     (mid_points_a1[:, 1] >= limit_y[0]) & (mid_points_a1[:, 1] <= limit_y[1]) &
                                                                     (mid_points_a1[:, 2] >= limit_z[0]) & (mid_points_a1[:, 2] <= limit_z[1])]
                                    now_point_a2_indexs = mid_indexs[(mid_points_a2[:, 0] >= limit_x[0]) & (mid_points_a2[:, 0] <= limit_x[1]) &
                                                                     (mid_points_a2[:, 1] >= limit_y[0]) & (mid_points_a2[:, 1] <= limit_y[1]) &
                                                                     (mid_points_a2[:, 2] >= limit_z[0]) & (mid_points_a2[:, 2] <= limit_z[1])]

                                    p_num0 = common_voxel_num_points[0][i]
                                    p_num1 = common_voxel_num_points[1][i]
                                    p_num2 = common_voxel_num_points[2][i]
                                    is_common_point_exist = False
                                    for j in range(p_num0):
                                        mid_x0, mid_y0, mid_z0, mid_r0 = common_voxels[0][i][j][0], common_voxels[0][i][j][1], \
                                                                         common_voxels[0][i][j][2], common_voxels[0][i][j][3]
                                        common_flag1 = False
                                        common_flag2 = False
                                        for k in range(p_num1):
                                            mid_x1, mid_y1, mid_z1, mid_r1 = common_voxels[1][i][k][0], common_voxels[1][i][k][1], \
                                                                             common_voxels[1][i][k][2], common_voxels[1][i][k][3]
                                            common_flag1 = (abs(mid_x1 - mid_x0) < 0.001) & (abs(mid_y1 - mid_y0) < 0.001) & \
                                                           (abs(mid_z1 - mid_z0) < 0.001) & (abs(mid_r1 - mid_r0) < 0.001)
                                            if common_flag1:
                                                break
                                        for k in range(p_num2):
                                            mid_x2, mid_y2, mid_z2, mid_r2 = common_voxels[2][i][k][0], common_voxels[2][i][k][1], \
                                                                             common_voxels[2][i][k][2], common_voxels[2][i][k][3]
                                            common_flag2 = (abs(mid_x2 - mid_x0) < 0.001) & (abs(mid_y2 - mid_y0) < 0.001) & \
                                                           (abs(mid_z2 - mid_z0) < 0.001) & (abs(mid_r2 - mid_r0) < 0.001)
                                            if common_flag2:
                                                break
                                        if common_flag1 & common_flag2:
                                            is_common_point_exist = True
                                            now_common_point = common_voxels[0][i][j]
                                            break

                                    if is_common_point_exist:
                                        if len(now_point_indexs) > 0 and len(now_point_a1_indexs) > 0 and len(now_point_a2_indexs) > 0:
                                            for j in range(len(now_point_indexs)):
                                                mid_flag1 = False
                                                mid_flag1 = (abs(now_common_point[0] - mid_points[now_point_indexs[j]][0]) < 0.001) & \
                                                           (abs(now_common_point[1] - mid_points[now_point_indexs[j]][1]) < 0.001) & \
                                                           (abs(now_common_point[2] - mid_points[now_point_indexs[j]][2]) < 0.001) & \
                                                           (abs(now_common_point[3] - mid_points[now_point_indexs[j]][3]) < 0.001)
                                                if mid_flag1:
                                                    now_index1 = now_point_indexs[j]
                                                    break
                                            for j in range(len(now_point_a1_indexs)):
                                                mid_flag2 = False
                                                mid_flag2 = (abs(now_common_point[0] - mid_points_a1[now_point_a1_indexs[j]][0]) < 0.001) & \
                                                           (abs(now_common_point[1] - mid_points_a1[now_point_a1_indexs[j]][1]) < 0.001) & \
                                                           (abs(now_common_point[2] - mid_points_a1[now_point_a1_indexs[j]][2]) < 0.001) & \
                                                           (abs(now_common_point[3] - mid_points_a1[now_point_a1_indexs[j]][3]) < 0.001)
                                                if mid_flag2:
                                                    now_index2 = now_point_a1_indexs[j]
                                                    break
                                            for j in range(len(now_point_a2_indexs)):
                                                mid_flag3 = False
                                                mid_flag3 = (abs(now_common_point[0] - mid_points_a2[now_point_a2_indexs[j]][0]) < 0.001) & \
                                                           (abs(now_common_point[1] - mid_points_a2[now_point_a2_indexs[j]][1]) < 0.001) & \
                                                           (abs(now_common_point[2] - mid_points_a2[now_point_a2_indexs[j]][2]) < 0.001) & \
                                                           (abs(now_common_point[3] - mid_points_a2[now_point_a2_indexs[j]][3]) < 0.001)
                                                if mid_flag3:
                                                    now_index3 = now_point_a2_indexs[j]
                                                    break
                                            if mid_flag1 & mid_flag2 & mid_flag3:
                                                common_points.append(now_common_point)
                                                common_point_indexs.append(now_index1)
                                                common_point_a1_indexs.append(now_index2)
                                                common_point_a2_indexs.append(now_index3)

                                if len(common_points) > 1:
                                    common_points = np.vstack(common_points)
                                    common_point_indexs = np.stack(common_point_indexs)
                                    common_point_a1_indexs = np.stack(common_point_a1_indexs)
                                    common_point_a2_indexs = np.stack(common_point_a2_indexs)

                                data_dict['common_points'] = common_points
                                data_dict['common_point_indexs'] = common_point_indexs
                                data_dict['common_point_a1_indexs'] = common_point_a1_indexs
                                data_dict['common_point_a2_indexs'] = common_point_a2_indexs
                            elif mv_status == 2: # only a1
                                common_points = []
                                mid_points = data_dict['points']
                                mid_points_a1 = data_dict['points_a1']
                                mid_indexs = np.arange(0, config.NUM_POINTS, 1)
                                common_point_indexs = []
                                common_point_a1_indexs = []
                                for i in range(len(common_voxels[0])):
                                    # for i in range(1):
                                    now_voxel_coords = common_voxel_coords[0][i]
                                    limit_x = [now_voxel_coords[2] * (config.VOXEL_SIZE[0]),
                                               (now_voxel_coords[2] + 1) * (config.VOXEL_SIZE[0])]
                                    limit_y = [now_voxel_coords[1] * (config.VOXEL_SIZE[1]) - 40,
                                               (now_voxel_coords[1] + 1) * (config.VOXEL_SIZE[1]) - 40]
                                    limit_z = [now_voxel_coords[0] * (config.VOXEL_SIZE[2]) - 3,
                                               (now_voxel_coords[0] + 1) * (config.VOXEL_SIZE[2]) - 3]
                                    now_point_indexs = mid_indexs[(mid_points[:, 0] >= limit_x[0]) & (mid_points[:, 0] <= limit_x[1]) &
                                                                  (mid_points[:, 1] >= limit_y[0]) & (mid_points[:, 1] <= limit_y[1]) &
                                                                  (mid_points[:, 2] >= limit_z[0]) & (mid_points[:, 2] <= limit_z[1])]
                                    now_point_a1_indexs = mid_indexs[(mid_points_a1[:, 0] >= limit_x[0]) & (mid_points_a1[:, 0] <= limit_x[1]) &
                                                                     (mid_points_a1[:, 1] >= limit_y[0]) & (mid_points_a1[:, 1] <= limit_y[1]) &
                                                                     (mid_points_a1[:, 2] >= limit_z[0]) & (mid_points_a1[:, 2] <= limit_z[1])]
                                    p_num0 = common_voxel_num_points[0][i]
                                    p_num1 = common_voxel_num_points[1][i]
                                    is_common_point_exist = False
                                    for j in range(p_num0):
                                        mid_x0, mid_y0, mid_z0, mid_r0 = common_voxels[0][i][j][0], common_voxels[0][i][j][1], \
                                                                         common_voxels[0][i][j][2], common_voxels[0][i][j][3]
                                        common_flag1 = False
                                        for k in range(p_num1):
                                            mid_x1, mid_y1, mid_z1, mid_r1 = common_voxels[1][i][k][0], common_voxels[1][i][k][1], \
                                                                             common_voxels[1][i][k][2], common_voxels[1][i][k][3]
                                            common_flag1 = (abs(mid_x1 - mid_x0) < 0.001) & (abs(mid_y1 - mid_y0) < 0.001) & \
                                                           (abs(mid_z1 - mid_z0) < 0.001) & (abs(mid_r1 - mid_r0) < 0.001)
                                            if common_flag1:
                                                break
                                        if common_flag1:
                                            is_common_point_exist = True
                                            now_common_point = common_voxels[0][i][j]
                                            break

                                    if is_common_point_exist:
                                        if len(now_point_indexs) > 0 and len(now_point_a1_indexs) > 0:
                                            for j in range(len(now_point_indexs)):
                                                mid_flag1 = False
                                                mid_flag1 = (abs(now_common_point[0] - mid_points[now_point_indexs[j]][0]) < 0.001) & \
                                                            (abs(now_common_point[1] - mid_points[now_point_indexs[j]][1]) < 0.001) & \
                                                            (abs(now_common_point[2] - mid_points[now_point_indexs[j]][2]) < 0.001) & \
                                                            (abs(now_common_point[3] - mid_points[now_point_indexs[j]][3]) < 0.001)
                                                if mid_flag1:
                                                    now_index1 = now_point_indexs[j]
                                                    break
                                            for j in range(len(now_point_a1_indexs)):
                                                mid_flag2 = False
                                                mid_flag2 = (abs(
                                                    now_common_point[0] - mid_points_a1[now_point_a1_indexs[j]][0]) < 0.001) & \
                                                            (abs(now_common_point[1] - mid_points_a1[now_point_a1_indexs[j]][
                                                                1]) < 0.001) & \
                                                            (abs(now_common_point[2] - mid_points_a1[now_point_a1_indexs[j]][
                                                                2]) < 0.001) & \
                                                            (abs(now_common_point[3] - mid_points_a1[now_point_a1_indexs[j]][3]) < 0.001)
                                                if mid_flag2:
                                                    now_index2 = now_point_a1_indexs[j]
                                                    break
                                            if mid_flag1 & mid_flag2:
                                                common_points.append(now_common_point)
                                                common_point_indexs.append(now_index1)
                                                common_point_a1_indexs.append(now_index2)

                                if len(common_points) > 1:
                                    common_points = np.vstack(common_points)
                                    common_point_indexs = np.stack(common_point_indexs)
                                    common_point_a1_indexs = np.stack(common_point_a1_indexs)

                                data_dict['common_points'] = common_points
                                data_dict['common_point_indexs'] = common_point_indexs
                                data_dict['common_point_a1_indexs'] = common_point_a1_indexs
                            elif mv_status == 3:  # only a2
                                common_points = []
                                mid_points = data_dict['points']
                                mid_points_a2 = data_dict['points_a2']
                                mid_indexs = np.arange(0, config.NUM_POINTS, 1)
                                common_point_indexs = []
                                common_point_a2_indexs = []
                                for i in range(len(common_voxels[0])):
                                    # for i in range(1):
                                    now_voxel_coords = common_voxel_coords[0][i]
                                    limit_x = [now_voxel_coords[2] * (config.VOXEL_SIZE[0]),
                                               (now_voxel_coords[2] + 1) * (config.VOXEL_SIZE[0])]
                                    limit_y = [now_voxel_coords[1] * (config.VOXEL_SIZE[1]) - 40,
                                               (now_voxel_coords[1] + 1) * (config.VOXEL_SIZE[1]) - 40]
                                    limit_z = [now_voxel_coords[0] * (config.VOXEL_SIZE[2]) - 3,
                                               (now_voxel_coords[0] + 1) * (config.VOXEL_SIZE[2]) - 3]
                                    now_point_indexs = mid_indexs[(mid_points[:, 0] >= limit_x[0]) & (mid_points[:, 0] <= limit_x[1]) &
                                                                  (mid_points[:, 1] >= limit_y[0]) & (mid_points[:, 1] <= limit_y[1]) &
                                                                  (mid_points[:, 2] >= limit_z[0]) & (mid_points[:, 2] <= limit_z[1])]
                                    now_point_a2_indexs = mid_indexs[
                                        (mid_points_a2[:, 0] >= limit_x[0]) & (mid_points_a2[:, 0] <= limit_x[1]) &
                                        (mid_points_a2[:, 1] >= limit_y[0]) & (mid_points_a2[:, 1] <= limit_y[1]) &
                                        (mid_points_a2[:, 2] >= limit_z[0]) & (mid_points_a2[:, 2] <= limit_z[1])]
                                    p_num0 = common_voxel_num_points[0][i]
                                    p_num2 = common_voxel_num_points[1][i]
                                    is_common_point_exist = False
                                    for j in range(p_num0):
                                        mid_x0, mid_y0, mid_z0, mid_r0 = common_voxels[0][i][j][0], common_voxels[0][i][j][1], \
                                                                         common_voxels[0][i][j][2], common_voxels[0][i][j][3]
                                        common_flag2 = False
                                        for k in range(p_num2):
                                            mid_x2, mid_y2, mid_z2, mid_r2 = common_voxels[1][i][k][0], common_voxels[1][i][k][1], \
                                                                             common_voxels[1][i][k][2], common_voxels[1][i][k][3]
                                            common_flag2 = (abs(mid_x2 - mid_x0) < 0.001) & (abs(mid_y2 - mid_y0) < 0.001) & \
                                                           (abs(mid_z2 - mid_z0) < 0.001) & (abs(mid_r2 - mid_r0) < 0.001)
                                            if common_flag2:
                                                break
                                        if common_flag2:
                                            is_common_point_exist = True
                                            now_common_point = common_voxels[0][i][j]
                                            break

                                    if is_common_point_exist:
                                        if len(now_point_indexs) > 0 and len(now_point_a2_indexs) > 0:
                                            for j in range(len(now_point_indexs)):
                                                mid_flag1 = False
                                                mid_flag1 = (abs(now_common_point[0] - mid_points[now_point_indexs[j]][0]) < 0.001) & \
                                                            (abs(now_common_point[1] - mid_points[now_point_indexs[j]][1]) < 0.001) & \
                                                            (abs(now_common_point[2] - mid_points[now_point_indexs[j]][2]) < 0.001) & \
                                                            (abs(now_common_point[3] - mid_points[now_point_indexs[j]][3]) < 0.001)
                                                if mid_flag1:
                                                    now_index1 = now_point_indexs[j]
                                                    break
                                            for j in range(len(now_point_a2_indexs)):
                                                mid_flag3 = False
                                                mid_flag3 = (abs(
                                                    now_common_point[0] - mid_points_a2[now_point_a2_indexs[j]][0]) < 0.001) & \
                                                            (abs(now_common_point[1] - mid_points_a2[now_point_a2_indexs[j]][
                                                                1]) < 0.001) & \
                                                            (abs(now_common_point[2] - mid_points_a2[now_point_a2_indexs[j]][
                                                                2]) < 0.001) & \
                                                            (abs(
                                                                now_common_point[3] - mid_points_a2[now_point_a2_indexs[j]][3]) < 0.001)
                                                if mid_flag3:
                                                    now_index3 = now_point_a2_indexs[j]
                                                    break
                                            if mid_flag1 & mid_flag3:
                                                common_points.append(now_common_point)
                                                common_point_indexs.append(now_index1)
                                                common_point_a2_indexs.append(now_index3)

                                if len(common_points) > 1:
                                    common_points = np.vstack(common_points)
                                    common_point_indexs = np.stack(common_point_indexs)
                                    common_point_a2_indexs = np.stack(common_point_a2_indexs)

                                data_dict['common_points'] = common_points
                                data_dict['common_point_indexs'] = common_point_indexs
                                data_dict['common_point_a2_indexs'] = common_point_a2_indexs

                #     print("consis_sample")
                # else:
                #     print("no_consis_sample")

        # print('common_points:', common_points.shape)
        # print('common_point_indexs:', common_point_indexs.shape)
        # print('common_point_a1_indexs:', common_point_a1_indexs.shape)
        # print('common_point_a2_indexs:', common_point_a2_indexs.shape)

        if ('MV_CLASS' in config):
            if ('DISCARD_VOXELS' in config):
                if config['DISCARD_VOXELS'] == True:
                    data_dict.pop('voxels', None)
                    data_dict.pop('voxel_coords', None)
                    data_dict.pop('voxel_num_points', None)
                    if ('a1' in config['MV_CLASS']):
                        data_dict.pop('voxels_a1', None)
                        data_dict.pop('voxel_a1_coords', None)
                        data_dict.pop('voxel_a1_num_points', None)
                    if ('a2' in config['MV_CLASS']):
                        data_dict.pop('voxels_a2', None)
                        data_dict.pop('voxel_a2_coords', None)
                        data_dict.pop('voxel_a2_num_points', None)

        # print('data_dict:', data_dict)
        # print('limit_x:', limit_x)
        # print('limit_y:', limit_y)
        # print('limit_z:', limit_z)
        # print('now_point_indexs:', now_point_indexs)
        # print('now_point_a1_indexs:', now_point_a1_indexs)
        # print('now_point_a2_indexs:', now_point_a2_indexs)
        # print('now_common_point:', now_common_point)
        # print('mid_points:', mid_points[now_point_indexs])
        # print('mid_points_a1:', mid_points_a1[now_point_a1_indexs])
        # print('mid_points_a2:', mid_points_a2[now_point_a2_indexs])
        # print('common_points:', common_points.shape)
        # print('common_point_indexs:', common_point_indexs.shape)
        # print('common_point_a1_indexs:', common_point_a1_indexs.shape)
        # print('common_point_a2_indexs:', common_point_a2_indexs.shape)
        # print('voxel_hash_index:', voxel_hash_index)
        # print('voxel_index:', voxel_index)
        # print('voxel_coords_product:', voxel_coords_product)
        # print('voxel_hash_ori_bool:', voxel_hash_ori_bool)
        # print('sum_bool:', np.sum(voxel_hash_ori_bool[0]), 'sum_a1_bool:', np.sum(voxel_hash_ori_bool[1]),
        #       'sum_a2_bool:', np.sum(voxel_hash_ori_bool[2]))
        # print('voxel_hash_ori_index:', voxel_hash_ori_index)
        # print('common_bool:', np.sum(voxel_hash_common_bool))
        # print('common_index:', common_voxel_indexs)
        # print('common_index_max:', np.max(common_voxel_indexs[0]), 'common_a1_index_max:', np.max(common_voxel_indexs[1]),
        #       'common_a2_index_max:', np.max(common_voxel_indexs[2]))
        # print('common_index_size:', len(common_voxel_indexs[0]), 'common_a1_index_size:', len(common_voxel_indexs[1]),
        #       'common_a2_index_size:', len(common_voxel_indexs[2]))
        # print('common_voxel_size:', len(common_voxels[0]), 'common_a1_voxel_size:', len(common_voxels[1]),
        #       'common_a2_voxel_size:', len(common_voxels[2]))
        # print('common_voxel_num_points:', common_voxel_num_points)
        # print('common_voxel_num_points_size:', len(common_voxel_num_points[0]),
        #       'common_a1_voxel_num_points_size:', len(common_voxel_num_points[1]),
        #       'common_a2_voxel_num_points_size:', len(common_voxel_num_points[2]))
        # print('common_points:', common_points)
        # print('common_points_size:', common_points.shape)
        # return

        # a = {}
        # print("stop:", a[10])
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
