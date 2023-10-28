#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np
import onnxruntime


class YoloxONNX(object):
    def __init__(
        self,
        model_path='yolox_nano.onnx',
        class_score_th=0.3,
        nms_th=0.45,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ):
        # 閾値
        self.class_score_th = class_score_th
        self.nms_th = nms_th

        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        # 入力サイズ
        self.input_shape = self.input_detail.shape[2:]

    def __call__(self, image):
        temp_image = copy.deepcopy(image)

        # 前処理
        image, ratio = self._preprocess(temp_image, self.input_shape)

        # 推論実施
        results = self.onnx_session.run(
            None,
            {self.input_name: image[None, :, :, :]},
        )

        # 後処理
        bboxes, scores, class_ids = self._postprocess(
            np.array(results),
            self.input_shape,
            num_classes=8,
            conf_thre=self.class_score_th,
            nms_thre=self.nms_th,
            ratio=ratio,
        )

        return bboxes, scores, class_ids

    def _preprocess(self, image, input_size, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_image = np.ones(
                (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_image = np.ones(input_size, dtype=np.uint8) * 114

        ratio = min(input_size[0] / image.shape[0],
                    input_size[1] / image.shape[1])
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_image = resized_image.astype(np.uint8)

        padded_image[:int(image.shape[0] * ratio), :int(image.shape[1] *
                                                        ratio)] = resized_image
        padded_image = padded_image.transpose(swap)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)

        return padded_image, ratio

    def _postprocess(
        self,
        prediction,
        image_size,
        num_classes,
        conf_thre=0.7,
        nms_thre=0.45,
        ratio=1.0,
    ):
        # ストライド処理
        grids = []
        expanded_strides = []
        strides = [8, 16, 32]
        hsizes = [image_size[0] // stride for stride in strides]
        wsizes = [image_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        prediction[..., :2] = (prediction[..., :2] + grids) * expanded_strides
        prediction[..., 2:4] = np.exp(prediction[..., 2:4]) * expanded_strides

        # TODO: 矢印のみの推論を弾くような処理が必要
        # box_corner = prediction.new(prediction.shape)
        prediction = prediction[0]
        box_corner = np.copy(prediction)
        # xの中心座標から左上の座標、右下の座標に変換
        # 左上
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        # 右下
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        # 予測結果の座標を変換して、box_cornerに格納
        prediction[:, :, :4] = box_corner[:, :, :4]

        outputs = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.shape[0]:
                continue
            # Get score and class with highest confidence
            # 1. 閾値を超える各クラスの確信度に対するマスクを取得
            conf_mask_multi = (
                image_pred[:, 5:5 + num_classes] *
                np.expand_dims(image_pred[:, 4], 1)) >= conf_thre

            # 2. マスクを使用して、対応するクラスの確信度とクラスインデックスを取得
            class_conf_multi = image_pred[:,
                                          5:5 + num_classes][conf_mask_multi]
            class_idx_multi = (conf_mask_multi.nonzero()[1]).astype(float)
            class_idx_multi = np.expand_dims(class_idx_multi, 1)

            # 3. detections_multiテンソルを作成
            detections_multi = np.concatenate(
                (image_pred[:, :5].repeat(
                    np.sum(conf_mask_multi, axis=1),
                    axis=0,
                ), np.expand_dims(class_conf_multi, 1), class_idx_multi),
                1,
            )
            # 4. NMSを実行 ここではクラスごとに実行する(red と arrowが混在している場合に対応するため)
            bboxes = detections_multi[:, :4]
            scores = (detections_multi[:, 4] * detections_multi[:, 5])
            class_ids = detections_multi[:, 6]
            indexes = []
            # TODO: マルチクラスNMSの実装は暫定対応
            for class_id in np.unique(class_ids):
                multi_nm_out_index_ = cv2.dnn.NMSBoxes(
                    bboxes[class_ids == class_id].tolist(),
                    scores[class_ids == class_id].tolist(),
                    conf_thre,
                    nms_thre,
                )
                indexes.append(
                    np.where(class_ids == class_id)[0][np.squeeze(
                        multi_nm_out_index_)])
            multi_nm_out_index = np.array(indexes).flatten()

            if len(multi_nm_out_index) > 0:
                detections_multi = detections_multi[multi_nm_out_index]

                if outputs[i] is None:
                    outputs[i] = detections_multi
                else:
                    outputs[i] = np.concatenate((outputs[i], detections_multi))

            # 5. レターボックスの縮小を元に戻す
            output_bboxes = []
            output_scores = []
            output_class_ids = []
            if outputs[i] is not None:
                for output in outputs[i]:
                    output_bboxes.append(output[:4].tolist())
                    output_scores.append(output[4] * output[5])
                    output_class_ids.append(output[6])

                for index, bbox in enumerate(output_bboxes):
                    output_bboxes[index][0] = int(bbox[0] / ratio)
                    output_bboxes[index][1] = int(bbox[1] / ratio)
                    output_bboxes[index][2] = int(bbox[2] / ratio)
                    output_bboxes[index][3] = int(bbox[3] / ratio)

                output_bboxes = np.array(output_bboxes)
                output_scores = np.array(output_scores)
                output_class_ids = np.array(output_class_ids)

        return output_bboxes, output_scores, output_class_ids
