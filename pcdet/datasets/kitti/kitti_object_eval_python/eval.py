import io as sysio

import numba
import numpy as np
import math

from .rotate_iou import rotate_iou_gpu_eval


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False,
                           printflag=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    gt_bboxes = gt_datas[:, :4]

    # if printflag:
    #     print('det_size:', det_size)
    #     print('gt_size:', gt_size)
    #     print('dt_scores:', dt_scores)
    #     print('dt_alphas:', dt_alphas)
    #     print('gt_alphas:', gt_alphas)
    #     print('dt_bboxes:', dt_bboxes)

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    # if metric == 2:
        # print('total_dt_num:', total_dt_num) # [ 7  2  7  5  6 10 10  6  1 10  6  7  9  3  1  9  3 11  3  8  3  6  9 10
                                             #  5  9  9  1  6  8  6  4  9  5 10  3  6  7  6  1  8  1  8  3  8  7 10  4
                                             #  5 11  7  4 10  4  6  3 17 15 10  8  4  9  4  7 11  4  2 11 17  8  6 20
                                             #  3 10  5  4 14  5  8 10  3  6  5 11  9  3  7  3 13  1 13  5  7 10 11  7
                                             # 14  7 12  6]
        # print('total_gt_num:', total_gt_num) # [ 1  1  2  2  4 10  5  3  3  7  1  3 10  0  1  6  1  6  3  6  1  1  8  5
                                             #  5  8  8  2  4  4  4  2  3  2 10  1  4  0  5  1  8  1  5  2  4  7  8  2
                                             #  5  9  2  2  1  4  2  1 10  8  2  1  5  5  4  2  7  2  1  9 13  5  4 13
                                             #  2  8  4  4 15  3  5 10  1  4  7  8  5  3  3  3  7  1  8  2  2  5  9  5
                                             # 12  6  9  2]
        # print('gt_annos0:', gt_annos[0])
        # print('gt_annos1:', gt_annos[1])
        # print('gt_annos2:', gt_annos[2])
        # print('gt_annos3:', gt_annos[3])
        # print('gt_annos4:', gt_annos[4])
        # print('gt_annos5:', gt_annos[5])
        # print('gt_annos6:', gt_annos[6])
        # print('gt_annos7:', gt_annos[7])
        # print('gt_annos8:', gt_annos[8])
        # print('gt_annos9:', gt_annos[9])

    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    # print('num_parts:', num_parts) # 100
    # print('split_parts:', split_parts) # 每一份的样本个数
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    # print('************eval_done %s ************' %metric)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    if metric == 2:
        # print('dt_annos_size:', len(dt_annos))
        # print('gt_annos_size:', len(gt_annos))
        # print('dt_annos:', dt_annos[0:10])
        # print('gt_annos:', gt_annos[0:10])
        # print('overlaps:', overlaps)
        # print('overlaps:', rets[0][0:5]) # 每个点云的dt和对应的gt测3diou
        # print('overlaps_size:', len(rets[0]))  # 100 每个点云的dt和对应的gt测3diou
        # print('overlaps_size:', rets[0][50].shape)
        # print('parted_overlaps:', rets[1]) # 点云的dt和gt测3diou，数量不同时处理方式不同
        # print('parted_overlaps_size:', len(rets[1]))
        # print('parted_overlaps_size:', rets[1][9])
        # print('total_dt_num:', rets[2])
        # print('total_dt_sum2:', sum(rets[2]))
        # print('total_gt_num:', rets[3])


        # print('total_gt_sum:', sum(rets[3]))

        total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
        total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

        lack_dt_car = []
        lack_dt_pedestrian = []
        lack_dt_cyclist = []
        bad_dt_car = []
        bad_dt_pedestrian = []
        bad_dt_cyclist = []

        lack_dt_car_depth = []
        lack_dt_pedestrian_depth = []
        lack_dt_cyclist_depth = []
        bad_dt_car_depth = []
        bad_dt_pedestrian_depth = []
        bad_dt_cyclist_depth = []

        lack_dt_car_z = []
        lack_dt_pedestrian_z = []
        lack_dt_cyclist_z = []
        bad_dt_car_z = []
        bad_dt_pedestrian_z = []
        bad_dt_cyclist_z = []

        for i in range(len(gt_annos)):
        # for i in range(5):
            if len(overlaps[i]) == 0: # 无目标未检出，将所有gt样本列入漏检
                # print('0dt:', dt_annos[i]['frame_id'])
                for j in range(len(gt_annos[i]['name'])):
                    if gt_annos[i]['name'][j].lower() == 'car':
                        mid_annos = [dt_annos[i]['frame_id'], gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                     gt_annos[i]['num_points_in_gt'][j]]
                        pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                    gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                        lack_dt_car_depth.append(pts_depth)
                        lack_dt_car_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                        lack_dt_car.append(mid_annos)
                    elif gt_annos[i]['name'][j].lower() == 'pedestrian':
                        mid_annos = [dt_annos[i]['frame_id'], gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                     gt_annos[i]['num_points_in_gt'][j]]
                        pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                    gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                        lack_dt_pedestrian_depth.append(pts_depth)
                        lack_dt_pedestrian_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                        lack_dt_pedestrian.append(mid_annos)
                    elif gt_annos[i]['name'][j].lower() == 'cyclist' and max_iou == 0:
                        mid_annos = [dt_annos[i]['frame_id'], gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                     gt_annos[i]['num_points_in_gt'][j]]
                        pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                    gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                        lack_dt_cyclist_depth.append(pts_depth)
                        lack_dt_cyclist_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                        lack_dt_cyclist.append(mid_annos)
            else:
                for j in range(len(gt_annos[i]['name'])):
                    if gt_annos[i]['name'][j].lower() == 'car' or gt_annos[i]['name'][j].lower() == 'pedestrian' \
                            or gt_annos[i]['name'][j].lower() == 'cyclist':
                        mid_data = overlaps[i][:, j]
                        max_iou = max(mid_data)
                        if max_iou != 0 and len(overlaps[i][:, j]) > 1:
                            max_index = mid_data.tolist().index(max_iou)
                        elif max_iou != 0 and len(overlaps[i][:, j]) == 1:
                            max_index = 0
                        if gt_annos[i]['name'][j].lower() == 'car' and max_iou > 0 and max_iou < 0.7:
                            mid_annos = [dt_annos[i]['frame_id'], round(dt_annos[i]['score'][max_index], 8),
                                         round(max_iou, 4), gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                         gt_annos[i]['num_points_in_gt'][j]]
                            # print('mid_annos0:', i, j, mid_annos)
                            pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                        gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                            bad_dt_car_depth.append(pts_depth)
                            bad_dt_car_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                            bad_dt_car.append(mid_annos)
                        elif gt_annos[i]['name'][j].lower() == 'car' and max_iou == 0:
                            mid_annos = [dt_annos[i]['frame_id'], gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                         gt_annos[i]['num_points_in_gt'][j]]
                            # print('mid_annos1:', i, j, mid_annos)
                            pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                        gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                            lack_dt_car_depth.append(pts_depth)
                            lack_dt_car_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                            lack_dt_car.append(mid_annos)

                        if gt_annos[i]['name'][j].lower() == 'pedestrian' and max_iou > 0 and max_iou < 0.5:
                            mid_annos = [dt_annos[i]['frame_id'], round(dt_annos[i]['score'][max_index], 8),
                                         round(max_iou, 4), gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                         gt_annos[i]['num_points_in_gt'][j]]
                            # print('mid_annos2:', i, j, mid_annos)
                            pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                        gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                            bad_dt_pedestrian_depth.append(pts_depth)
                            bad_dt_pedestrian_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                            bad_dt_pedestrian.append(mid_annos)
                        elif gt_annos[i]['name'][j].lower() == 'pedestrian' and max_iou == 0:
                            mid_annos = [dt_annos[i]['frame_id'], gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                         gt_annos[i]['num_points_in_gt'][j]]
                            # print('mid_annos3:', i, j, mid_annos)
                            pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                        gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                            lack_dt_pedestrian_depth.append(pts_depth)
                            lack_dt_pedestrian_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                            lack_dt_pedestrian.append(mid_annos)

                        if gt_annos[i]['name'][j].lower() == 'cyclist' and max_iou > 0 and max_iou < 0.5:
                            mid_annos = [dt_annos[i]['frame_id'], round(dt_annos[i]['score'][max_index], 8),
                                         round(max_iou, 4), gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                         gt_annos[i]['num_points_in_gt'][j]]
                            # print('mid_annos4:', i, j, mid_annos)
                            pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                        gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                            # pts_depth1 = np.linalg.norm(gt_annos[i]['location'][j], axis=0)
                            # print('bad_cyclist_depth:', pts_depth, pts_depth1)
                            bad_dt_cyclist_depth.append(pts_depth)
                            bad_dt_cyclist_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                            bad_dt_cyclist.append(mid_annos)
                        elif gt_annos[i]['name'][j].lower() == 'cyclist' and max_iou == 0:
                            mid_annos = [dt_annos[i]['frame_id'], gt_annos[i]['occluded'][j], gt_annos[i]['index'][j],
                                         gt_annos[i]['num_points_in_gt'][j]]
                            # print('mid_annos5:', i, j, mid_annos)
                            pts_depth = np.linalg.norm([gt_annos[i]['gt_boxes_lidar'][j][0],
                                                        gt_annos[i]['gt_boxes_lidar'][j][1]], axis=0)
                            lack_dt_cyclist_depth.append(pts_depth)
                            lack_dt_cyclist_z.append(gt_annos[i]['gt_boxes_lidar'][j][2])
                            lack_dt_cyclist.append(mid_annos)

                        # print('ok:', i, j)
        # print('bad_dt_car:', bad_dt_car)
        # print('bad_dt_pedestrian:', bad_dt_pedestrian)
        # print('bad_dt_cyclist:', bad_dt_cyclist)
        # print('lack_dt_car:', lack_dt_car)
        # print('lack_dt_pedestrian:', lack_dt_pedestrian)
        # print('lack_dt_cyclist:', lack_dt_cyclist)

        # mean_bad_car = np.mean(np.array(bad_dt_car_depth))
        # mean_lack_car = np.mean(np.array(lack_dt_car_depth))
        # print('mean_bad_car:', mean_bad_car)
        # print('mean_lack_car:', mean_lack_car)
        # mean_bad_pedestrian = np.mean(np.array(bad_dt_pedestrian_depth))
        # mean_lack_pedestrian = np.mean(np.array(lack_dt_pedestrian_depth))
        # print('mean_bad_pedestrian:', mean_bad_pedestrian)
        # print('mean_lack_pedestrian:', mean_lack_pedestrian)
        # mean_bad_cyclist = np.mean(np.array(bad_dt_cyclist_depth))
        # mean_lack_cyclist = np.mean(np.array(lack_dt_cyclist_depth))
        # print('mean_bad_cyclist:', mean_bad_cyclist)
        # print('mean_lack_cyclist:', mean_lack_cyclist)

        mid_resultset = [bad_dt_car, bad_dt_pedestrian, bad_dt_cyclist,
                         lack_dt_car, lack_dt_pedestrian, lack_dt_cyclist]
        mid_depthset = [bad_dt_car_depth, bad_dt_pedestrian_depth, bad_dt_cyclist_depth,
                     lack_dt_car_depth, lack_dt_pedestrian_depth, lack_dt_cyclist_depth]
        mid_zset = [bad_dt_car_z, bad_dt_pedestrian_z, bad_dt_cyclist_z,
                    lack_dt_car_z, lack_dt_pedestrian_z, lack_dt_cyclist_z]
        # gt point count
        bad_cyclist = []
        lack_cyclist = []
        gt_point_num = []
        for k in range(6):
            mid_num = [0, 0, 0, 0]
            for i in range(len(mid_resultset[k])):
                if k <= 2:
                    mid_pcount = mid_resultset[k][i][5]
                else:
                    mid_pcount = mid_resultset[k][i][3]
                if mid_pcount <= 50:
                    mid_num[0] += 1
                    if k == 2:
                        bad_cyclist.append(mid_resultset[k][i])
                    elif k == 5:
                        lack_cyclist.append(mid_resultset[k][i])
                elif mid_pcount > 50 and mid_pcount <= 150:
                    mid_num[1] += 1
                    if k == 2:
                        bad_cyclist.append(mid_resultset[k][i])
                    elif k == 5:
                        lack_cyclist.append(mid_resultset[k][i])
                elif mid_pcount > 150 and mid_pcount <= 300:
                    mid_num[2] += 1
                elif mid_pcount > 300:
                    mid_num[3] += 1
            gt_point_num.append(mid_num)
        # print('gt_point_num:', gt_point_num)
        # print('gt_point_sum:', np.sum(gt_point_num, axis=1))
        # print('bad_cyclist:', bad_cyclist)
        # print('lack_cyclist:', lack_cyclist)
        # object count
        object_count = []
        for k in range(6):
            mid_num = [0, 0, 0, 0]
            for i in range(len(mid_resultset[k])):
                if k <= 2:
                    mid_occluded = mid_resultset[k][i][3]
                else:
                    mid_occluded = mid_resultset[k][i][1]
                if mid_occluded == 0:
                    mid_num[0] += 1
                elif mid_occluded == 1:
                    mid_num[1] += 1
                elif mid_occluded == 2:
                    mid_num[2] += 1
                elif mid_occluded == 3:
                    mid_num[3] += 1
            object_count.append(mid_num)
        # print('object_count:', object_count)
        # iou
        iou_num = []
        for k in range(3):
            if k == 0:
                mid_num = [0, 0, 0, 0, 0, 0, 0]
            else:
                mid_num = [0, 0, 0, 0, 0]
            for i in range(len(mid_resultset[k])):
                mid_iou = mid_resultset[k][i][2]
                if mid_iou < 0.1:
                    mid_num[0] += 1
                elif mid_iou >= 0.1 and mid_iou < 0.2:
                    mid_num[1] += 1
                elif mid_iou >= 0.2 and mid_iou < 0.3:
                    mid_num[2] += 1
                elif mid_iou >= 0.3 and mid_iou < 0.4:
                    mid_num[3] += 1
                elif mid_iou >= 0.4 and mid_iou < 0.5:
                    mid_num[4] += 1
                elif mid_iou >= 0.5 and mid_iou < 0.6 and k == 0:
                    mid_num[5] += 1
                elif mid_iou >= 0.6 and mid_iou < 0.7 and k == 0:
                    mid_num[6] += 1
            iou_num.append(mid_num)
        # print('iou_num:', iou_num)
        # score
        score_num = []
        for k in range(3):
            mid_num = [0, 0, 0, 0, 0]
            for i in range(len(mid_resultset[k])):
                mid_score = mid_resultset[k][i][1]
                if mid_score < 0.2:
                    mid_num[0] += 1
                elif mid_score >= 0.2 and mid_score < 0.4:
                    mid_num[1] += 1
                elif mid_score >= 0.4 and mid_score < 0.6:
                    mid_num[2] += 1
                elif mid_score >= 0.6 and mid_score < 0.8:
                    mid_num[3] += 1
                elif mid_score >= 0.8:
                    mid_num[4] += 1
            score_num.append(mid_num)
        # print('score_num:', score_num)
        # depth
        depth_num = []
        for k in range(6):
            mid_num = [0, 0, 0, 0, 0, 0, 0]
            for i in range(len(mid_depthset[k])):
                mid_depth = mid_depthset[k][i]
                if mid_depth < 10:
                    mid_num[0] += 1
                elif mid_depth >= 10 and mid_depth < 20:
                    mid_num[1] += 1
                elif mid_depth >= 20 and mid_depth < 30:
                    mid_num[2] += 1
                elif mid_depth >= 30 and mid_depth < 40:
                    mid_num[3] += 1
                elif mid_depth >= 40 and mid_depth < 50:
                    mid_num[4] += 1
                elif mid_depth >= 50 and mid_depth < 60:
                    mid_num[5] += 1
                elif mid_depth >= 60:
                    mid_num[6] += 1
            depth_num.append(mid_num)
        # print('depth_num:', depth_num)
        # z
        z_num = []
        for k in range(6):
            mid_num = [0, 0, 0, 0, 0, 0]
            for i in range(len(mid_zset[k])):
                mid_z = mid_zset[k][i]
                if mid_z >= -2 and mid_z < -1.5:
                    mid_num[0] += 1
                elif mid_z >= -1.5 and mid_z < -1:
                    mid_num[1] += 1
                elif mid_z >= -1 and mid_z < -0.5:
                    mid_num[2] += 1
                elif mid_z >= -0.5 and mid_z < 0:
                    mid_num[3] += 1
                elif mid_z >= 0 and mid_z < 0.5:
                    mid_num[4] += 1
                elif mid_z >= 0.5 and mid_z < 1:
                    mid_num[5] += 1
            z_num.append(mid_num)
        # print('z_num:', z_num)

    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps) # 2
    num_class = len(current_classes) # 3
    num_difficulty = len(difficultys) # 3
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    # print('current_classes:', current_classes) # [0, 1, 2]
    # print('difficultys:', difficultys) # [0, 1, 2]
    # print('min_overlaps:', min_overlaps)
    # print('split_parts:', split_parts) # 10
    for m, current_class in enumerate(current_classes): # 汽车 行人 自行车
        for l, difficulty in enumerate(difficultys): # 简单 中等 困难
            # 根据当前难度,gt和dt的尺寸生成相应的取舍标签
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            # 前5个量的长度与gt_annos相同
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets

            for k, min_overlap in enumerate(min_overlaps[:, metric, m]): # 高iou阈值 低iou阈值 # bbox bev 3d # 汽车 行人 自行车
                thresholdss = []
                for i in range(len(gt_annos)):
                    if metric == 2 and m == 2 and l == 0 and k == 0 and i == 2:
                        isprint = True
                    else:
                        isprint = False
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False,
                        printflag=isprint)
                    tp, fp, fn, similarity, thresholds = rets
                    # print('thresholds:', i, thresholds)
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)

                # 汽车比自行车满足条件的个数多
                # if metric == 2 and m == 0 and l == 0 and k == 0:
                #     print('total_num_valid_gt:', total_num_valid_gt) # 77
                #     print('thresholdss0:', thresholdss)
                #     print('thresholds0:', thresholds)
                # if metric == 2 and m == 1 and l == 0 and k == 0:
                #     print('total_num_valid_gt:', total_num_valid_gt)  # 28
                #     print('thresholdss0:', thresholdss)
                #     print('thresholds0:', thresholds)
                # if metric == 2 and m == 2 and l == 0 and k == 0:
                #     print('total_num_valid_gt:', total_num_valid_gt)  # 10
                #     print('thresholdss0:', thresholdss)
                #     print('thresholds0:', thresholds)

                # if metric == 2:
                #     print('thresholds:', m, l, k, thresholds)

                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part

                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)

    # print('recall:', recall)
    # print('precision:', precision)
    # print('precision_size:', len(precision)) # 3 # 汽车 行人 自行车
    # print('precision0_size:', len(precision[0])) # 3 # 简单 中等 困难
    # print('precision00_size:', len(precision[0][0])) # 2 # 高iou阈值 低iou阈值
    # print('precision000_size:', len(precision[0][0][0])) # 41
    # print('orientation:', aos)

    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4): # 采样间隔4,取11个值
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]): # 除去第一个值外，取40个值
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    # print('precision0:', ret["precision"].shape)  # (3, 3, 2, 41)

    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    # print('precision1:', ret["precision"].shape) # (3, 3, 2, 41)

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])

    # print('precision2:', ret["precision"].shape)  # (3, 3, 2, 41)

    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5, 0.7], [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    # 评估运算
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
                                 f"{mAPbbox[j, 1, i]:.4f}, "
                                 f"{mAPbbox[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
                                 f"{mAPbev[j, 1, i]:.4f}, "
                                 f"{mAPbev[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                                 f"{mAP3d[j, 1, i]:.4f}, "
                                 f"{mAP3d[j, 2, i]:.4f}"))

            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                                     f"{mAPaos[j, 1, i]:.2f}, "
                                     f"{mAPaos[j, 2, i]:.2f}"))
                # if i == 0:
                   # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                   # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                   # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]

            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 1, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
                                 f"{mAPbev_R40[j, 1, i]:.4f}, "
                                 f"{mAPbev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP3d_R40[j, 2, i]:.4f}"))
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                                     f"{mAPaos_R40[j, 1, i]:.2f}, "
                                     f"{mAPaos_R40[j, 2, i]:.2f}"))
                if i == 0:
                   ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
                   ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
                   ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

            if i == 0:
                # ret_dict['%s_3d/easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
                # ret_dict['%s_3d/moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
                # ret_dict['%s_3d/hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
                # ret_dict['%s_bev/easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
                # ret_dict['%s_bev/moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
                # ret_dict['%s_bev/hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
                # ret_dict['%s_image/easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
                # ret_dict['%s_image/moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
                # ret_dict['%s_image/hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]

                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return result
