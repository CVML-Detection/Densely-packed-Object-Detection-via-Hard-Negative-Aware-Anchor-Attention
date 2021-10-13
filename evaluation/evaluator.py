import os
import csv
from evaluation.cal_acc import get_dt_json, get_gt_json, summarize
from evaluation.coco_custom import CustomCOCO
from pycocotools.cocoeval import COCOeval
from utils import copy_and_add_header_anno


class Evaluator(object):
    def __init__(self, opts):
        self.opts = opts
        self.data_type = opts.data_type

        # for VOC
        self.det_img_name = list()
        self.det_additional = list()
        self.det_boxes = list()
        self.det_labels = list()
        self.det_scores = list()

        # for COCO
        self.results = list()
        self.img_ids = list()

    def get_info(self, info):
        if self.data_type == 'sku':

            (pred_boxes, pred_labels, pred_scores, img_names, additional_info) = info

            self.det_img_name.append(img_names)
            self.det_additional.append(additional_info)

            w = additional_info[0]
            h = additional_info[1]

            # x1 y1 x2 y2
            pred_boxes[:, 0] *= w
            pred_boxes[:, 1] *= h
            pred_boxes[:, 2] *= w
            pred_boxes[:, 3] *= h

            self.det_boxes.append(pred_boxes.cpu())
            self.det_labels.append(pred_labels.cpu())
            self.det_scores.append(pred_scores.cpu())

    def evaluate(self, dataset):
        if self.data_type == 'sku':

            split = dataset.split
            csv_data_lst = []
            csv_data_lst.append(['image_name', 'x1', 'y1', 'x2', 'y2', 'score'])
            for img_name, detection, score in zip(self.det_img_name, self.det_boxes, self.det_scores):

                # 1. convert from ascii int name to string
                img_name_ascii = img_name[0]
                img_name_ascii = img_name_ascii.numpy()
                img_name_from_ascii = [chr(c) for c in img_name_ascii]
                img_name = ''.join(img_name_from_ascii)
                img_name = img_name + '.jpg'

                for det, sco in zip(detection, score):
                    row = [img_name, str(det[0].item()), str(det[1].item()), str(det[2].item()), str(det[3].item()),
                           str(sco.item())]
                    csv_data_lst.append(row)

            # --- make gt_annotations ---
            data_root = self.opts.data_root
            os.makedirs('./gt', exist_ok=True)
            eval_no_hd_gt_path = './gt/annotations_no_hd_{}.csv'.format(split)
            eval_gt_path = './gt/annotations_{}.csv'.format(split)
            copy_and_add_header_anno(data_root, eval_gt_path, eval_no_hd_gt_path, split)
            # --- make gt_annotations ---

            os.makedirs('./pred', exist_ok=True)
            res_file = os.path.join('./pred', 'predictions_{}.csv'.format(split))

            with open(res_file, 'w') as fl_csv:
                writer = csv.writer(fl_csv)
                writer.writerows(csv_data_lst)
            print("Saved output.csv file")

            gt_path = './gt/annotations_{}.csv'.format(split)
            pred_path = './pred/predictions_{}.csv'.format(split)

            gt_json, imgIds = get_gt_json(gt_path)
            dt_json = get_dt_json(pred_path, imgIds)

            gt_coco_format = CustomCOCO(gt_json)
            dt_coco_format = CustomCOCO(dt_json)

            # running evaluation
            cocoEval = COCOeval(gt_coco_format, dt_coco_format, iouType='bbox')
            maxDets = 300
            cocoEval.params.maxDets = [maxDets, maxDets, maxDets]

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            print('AP for 300 detections.')
            mAP = summarize(cocoEval, ap=1, maxDets=maxDets)
            mAP_50 = summarize(cocoEval, ap=1, maxDets=maxDets, iouThr=0.5)
            mAP_75, P_50 = summarize(cocoEval, ap=1, maxDets=maxDets, iouThr=0.75)
            AR300 = summarize(cocoEval, ap=0, maxDets=maxDets)
            AR300_50 = summarize(cocoEval, ap=0, maxDets=maxDets, iouThr=0.5)

            print("AP : ", mAP)
            print("AP.50 : ", mAP_50)
            print("AP.75 : ", mAP_75)
            print("AR300 : ", AR300)
            print("AR300_50 : ", AR300_50)
            print("P_50 : ", P_50)

            results = [mAP, mAP_50, mAP_75, AR300, AR300_50, P_50]

        return results