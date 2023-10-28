#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2

from yolox.yolox_onnx import YoloxONNX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)

    parser.add_argument(
        "--model",
        type=str,
        default='model/openlenda_s.onnx',
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.5,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.01,
        help='NMS IoU threshold',
    )

    parser.add_argument(
        '--grouping_th',
        type=float,
        default=0.8,
    )

    parser.add_argument("--use_gpu", action="store_true")

    args = parser.parse_args()

    return args


def main():
    # 引数解析
    args = get_args()
    cap_device = args.device

    if args.movie is not None:
        cap_device = args.movie
    image_path = args.image

    model_path = args.model
    score_th = args.score_th
    nms_th = args.nms_th

    grouping_th = args.grouping_th

    use_gpu = args.use_gpu

    # カメラ準備
    if image_path is None:
        cap = cv2.VideoCapture(cap_device)

    # モデルロード
    providers = ['CPUExecutionProvider']
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    yolox = YoloxONNX(
        model_path=model_path,
        class_score_th=score_th,
        nms_th=nms_th,
        providers=providers,
    )

    # クラスリスト読み込み
    class_names = []
    with open('classes.txt', 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    while True:
        start_time = time.time()

        # カメラキャプチャ
        if image_path is None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = cv2.imread(image_path)
        debug_image = copy.deepcopy(frame)

        # 推論実施
        bboxes, scores, class_ids = yolox(frame)

        # 同じバウンディングボックスで結果をグルーピング
        bboxes, scores, class_ids = grouping_by_iou(
            bboxes,
            scores,
            class_ids,
            grouping_th,
        )

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            bboxes,
            scores,
            class_ids,
            class_names,
        )

        # 画面反映
        cv2.imshow('OpenLenda ONNX Sample', debug_image)

        # キー処理(ESC：終了)
        if image_path is None:
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
        else:
            cv2.waitKey(0)
            break

    if image_path is None:
        cap.release()
    cv2.destroyAllWindows()


def iou(bbox1, bbox2):
    # bbox = [xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    bx_mn, by_mn, bx_mx, by_mx = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w * h

    iou = intersect / (a_area + b_area - intersect)
    return iou


def grouping_by_iou(bboxes, scores, class_ids, grouping_th):
    grouping_bboxes = []
    grouping_scores = []
    grouping_class_id = []
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        if len(grouping_bboxes) == 0:
            grouping_bboxes.append(bbox.tolist())
            grouping_scores.append([score])
            grouping_class_id.append([class_id])
            continue

        append_flag = False
        for index, g_bbox in enumerate(grouping_bboxes):
            if iou(g_bbox, bbox) > grouping_th:
                grouping_scores[index].append(score)
                grouping_class_id[index].append(class_id)
                append_flag = True
                break
        if not append_flag:
            grouping_bboxes.append(bbox.tolist())
            grouping_scores.append([score])
            grouping_class_id.append([class_id])

    return grouping_bboxes, grouping_scores, grouping_class_id


def draw_debug(
    image,
    elapsed_time,
    bboxes,
    scores,
    class_ids,
    class_names,
):
    debug_image = copy.deepcopy(image)

    for bbox, score_list, class_id_list in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=2,
        )

        # クラスID、スコア
        for index, (
                score,
                class_id,
        ) in enumerate(zip(score_list, class_id_list)):
            class_name = str(class_names[int(class_id)])
            score_text = '%.2f' % score
            text = '%s:%s' % (class_name, score_text)

            debug_image = cv2.putText(
                debug_image,
                text,
                (x1, y2 + 20 + (24 * index)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                thickness=2,
            )

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
