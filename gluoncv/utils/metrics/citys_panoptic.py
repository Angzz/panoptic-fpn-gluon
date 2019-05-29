"""MS COCO Panoptic Segmentation Evaluate Metrics."""
from __future__ import absolute_import

import sys
import io
import os
import json
from os import path as osp
import warnings

import cv2
import numpy as np
import mxnet as mx


class CitysPanopticMetric(mx.metric.EvalMetric):
    """Instance segmentation metric for COCO bbox and segm task.
    Will return box summary, box metric, seg summary and seg metric.

    Parameters
    ----------
    dataset : instance of gluoncv.data.COCOInstance
        The validation dataset.
    save_prefix : str
        Prefix for the saved JSON results.
    use_time : bool
        Append unique datetime string to created JSON file name if ``True``.
    cleanup : bool
        Remove created JSON file if ``True``.
    score_thresh : float
        Detection results with confident scores smaller than ``score_thresh`` will
        be discarded before saving to results.

    """
    def __init__(self, dataset, save_prefix, max_dets=100, use_time=True,
                 cleanup=False, score_thresh=1e-3, panoptic_score_thresh=0.5,
                 root=osp.expanduser('~/.mxnet/datasets/citys')):
        super(CitysPanopticMetric, self).__init__('CityesPanoptic')
        self.dataset = dataset
        self._img_ids = dataset.images
        self._max_dets = max_dets
        self._current_id = 0
        self._cleanup = cleanup
        self._root = root
        self._results = {'annotations': []} # for generate predict json
        self._panoptic_images = {} # for generate predict images
        self._inst_mapping = {0: 24, 1: 25, 2: 26, 3: 27, \
                              4: 28, 5: 31, 6: 32, 7: 33}
        self._stuff_mapping = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13,
                               5: 17, 6: 19, 7: 20, 8: 21, 9: 22, \
                               10: 23, 11: 24, 12: 25, 13: 26, \
                               14: 27, 15: 28, 16: 31, 17: 32, 18: 33}
        self._stuff_inst_mapping = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, \
                                    16: 5, 17: 6, 18: 7}
        self._score_thresh = score_thresh
        self._panoptic_score_thresh = panoptic_score_thresh
        self._min_thing_area = 0
        self._min_stuff_area = 0

        if use_time:
            import datetime
            t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        else:
            t = ''
        self._filename = osp.abspath(osp.expanduser(save_prefix) + t + '.json')
        try:
            f = open(self._filename, 'w')
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()

        self._save_imgpath = osp.abspath(osp.expanduser(save_prefix) + t)
        if not os.path.exists(self._save_imgpath):
            os.mkdir(self._save_imgpath)
        self._eval_result_file = osp.abspath(osp.expanduser(save_prefix) + t + '_result.json')

    def __del__(self):
        if self._cleanup:
            try:
                os.remove(self._filename)
            except IOError as err:
                warnings.warn(str(err))

    def reset(self):
        self._current_id = 0
        self._results = {'annotations': []}
        self._panoptic_images = {}

    def _dump_json(self):
        """Write coco json file"""
        if not self._current_id == len(self._img_ids):
            warnings.warn(
                'Recorded {} out of {} validation images, incomplete results'.format(
                    self._current_id, len(self._img_ids)))
        try:
            with open(self._filename, 'w') as f:
                json.dump(self._results, f)
        except IOError as e:
            raise RuntimeError("Unable to dump json file, ignored. What(): {}".format(str(e)))

    def _dump_image(self):
        """Write coco json file"""
        if not self._current_id == len(self._img_ids):
            warnings.warn(
                'Recorded {} out of {} validation images, incomplete results'.format(
                    self._current_id, len(self._img_ids)))
        try:
            for im_name, im in self._panoptic_images.items():
                cv2.imwrite(osp.join(self._save_imgpath, im_name), im)
        except IOError as e:
            raise RuntimeError("Unable to dump images, ignored. What(): {}".format(str(e)))


    def _update(self):
        """Save necessary results for evaluation."""
        print("Saving prediction json files...")
        self._dump_json()
        print("Saving prediction json files done...")
        print("Saving prediction images...")
        self._dump_image()
        print("Saving prediction images done...")

    def get(self):
        """Get evaluation metrics. """
        self._update()
        print("Starting evaluation...")
        from .evalPanopticSemanticLabeling import evaluatePanoptic
        gt_path = osp.join(self._root, 'gtFine')
        gt_folder = osp.join(gt_path, 'cityscapes_panoptic_val')
        gt_json = osp.join(gt_path, 'cityscapes_panoptic_val.json')
        pred_folder = self._save_imgpath
        pred_json = self._filename
        result_file = self._eval_result_file
        results = evaluatePanoptic(gt_json, gt_folder, pred_json, pred_folder, result_file)
        return results

    def panoptic_merge(self, insts, segms, dets):
        '''
        insts : [N, 28, 28]
        segms : [H, W]
        dets : [N, 6]

        return : category_id, id
        Note : for stuff : category_id == id
               for thing : category_id == id // 1000
        '''
        panoptic = np.zeros(segms.shape + (3,), dtype=np.uint16)
        unique_cls = np.unique(segms)
        stuff = np.zeros_like(segms)
        for _cls in unique_cls:
            if _cls in self._stuff_inst_mapping:
                stuff[segms == _cls] = 255
            else:
                stuff[segms == _cls] = self._stuff_mapping[_cls]
        panoptic[:, :, 2] = stuff

        # Merge Thing
        for _cls in self._inst_mapping:
            sdet = dets[dets[:,-1] == _cls]
            sinst = insts[dets[:,-1] == _cls]
            inst_id = 0
            for i, inst in enumerate(sinst):
                score = sdet[i, -2]
                if score >= self._panoptic_score_thresh:
                    inst_map = panoptic[:, :, 1]
                    valid_area = (inst_map == 0) & (inst == 1)
                    if np.count_nonzero(valid_area) > self._min_thing_area:
                        thing_cls = self._inst_mapping[_cls]
                        panoptic[:, :, 1][valid_area] = thing_cls * 1000 + inst_id
                        panoptic[:, :, 2][valid_area] = thing_cls * 1000 + inst_id
                        inst_id += 1

        # Merge Stuff
        stuff_map = panoptic[:, :, 1] == 0
        stuff_cls = np.unique(panoptic[:, :, 2][stuff_map])
        for _cls in stuff_cls:
            if _cls >= 0:
                stuff_seg = (panoptic[:, :, 2] == _cls).astype(np.uint8)
                num, componets = cv2.connectedComponents(stuff_seg)
                for i in range(num):
                    if i > 0:
                        com_map = componets == i
                        if np.count_nonzero(com_map) <= self._min_stuff_area:
                            panoptic[:, :, 2][com_map] = 255

        # Convert 255 to Unlabeled
        panoptic[panoptic == 255] = 0
        return panoptic

    # pylint: disable=arguments-differ, unused-argument
    def update(self, pred_bboxes, pred_labels, pred_scores, pred_masks, pred_segms):
        """Update internal buffer with latest predictions.
        Note that the statistics are not available until you call self.get() to return
        the metrics.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        pred_masks: mxnet.NDArray or numpy.ndarray
            Prediction masks with *original* shape `H, W`.
        pred_segms: mxnet.NDArray or numpy.ndarray
            Prediction segms with *original* shape `H, W`.
        """
        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        # mask must be the same as image shape, so no batch dimension is supported
        pred_bbox, pred_label, pred_score, pred_mask = [
            as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores, pred_masks]]
        # filter out padded detection & low confidence detections
        valid_pred = np.where((pred_label >= 0) & (pred_score >= self._score_thresh))[0]
        pred_bbox = pred_bbox[valid_pred].astype('float32')
        pred_label = pred_label.flat[valid_pred].astype('int32')
        pred_score = pred_score.flat[valid_pred].astype('float32')
        pred_mask = pred_mask[valid_pred].astype('uint8')
        valid_idx = np.argsort(-pred_score)[:self._max_dets]
        pred_score = pred_score[valid_idx]
        pred_bbox = pred_bbox[valid_idx]
        pred_label = pred_label[valid_idx]
        pred_mask = pred_mask[valid_idx]
        pred_segm = pred_segms.astype('uint8')
        pred_dets = np.concatenate(
                [pred_bbox, pred_score[:, np.newaxis], pred_label[:, np.newaxis]], axis=1)
        panoptic = self.panoptic_merge(pred_mask, pred_segm, pred_dets)
        imgid = self._img_ids[self._current_id]
        self._current_id += 1
        file_name = imgid.split('/')[-1]
        img_name = file_name.replace('leftImg8bit', 'gtFine_panoptic')
        img_id = file_name.replace('_leftImg8bit.png', '')
        segments_info = []
        unique_cls = np.unique(panoptic[:, :, 2])
        for _cls in unique_cls:
            if _cls > 0: # 0 is unlabeled
                annots = {}
                annots['category_id'] = int(_cls // 1000 if _cls > 24 else _cls)
                annots['id'] = int(_cls)
                segments_info.append(annots)

        # Convert to coco panoptic format
        panoptic[:, :, 1] = panoptic[:, :, 1] // 256
        panoptic[:, :, 2] = panoptic[:, :, 2] % 256
        panoptic = panoptic.astype('uint8')
        # Save
        self._panoptic_images[img_name] = panoptic
        self._results['annotations'].append({'file_name': img_name,
                                             'image_id': img_id,
                                             'segments_info': segments_info})
