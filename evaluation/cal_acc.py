from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import pandas as pd
import numpy as np


def get_entry_dict():
    return {
        "segmentation": None,
        "iscrowd": 0,
        "image_id": None,
        "category_id": 1,
        "id": None,
        "bbox": None,
        "area": None
    }


def get_gt_json(gt_path='./gt/annotations_test.csv'):
    annot = pd.read_csv(gt_path)
    imgIds = sorted(list(set(annot['image_name'].tolist())))
    gt_json = []
    for annotid, row in tqdm(annot.iterrows()):
        entry = get_entry_dict()
        entry['image_id'] = imgIds.index(row['image_name'])
        entry['id'] = annotid
        entry['bbox'] = [row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1']]
        entry['area'] = (row['x2'] - row['x1']) * (row['y2'] - row['y1'])
        gt_json.append(entry)
    return gt_json, imgIds


def get_dt_json(pred_path='./pred/predictions_test.csv', imgIds=None):
    dt_json = []
    dets = pd.read_csv(pred_path)
    for detid, row in tqdm(dets.iterrows()):
        entry = get_entry_dict()
        entry['image_id'] = imgIds.index(row['image_name'])
        entry['id'] = detid
        entry['bbox'] = [row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1']]
        entry['area'] = (row['x2'] - row['x1']) * (row['y2'] - row['y1'])
        entry['score'] = row['score']
        dt_json.append(entry)
    return dt_json


def summarize(eval_results, ap=1, iouThr=None, areaRng='all', maxDets=100):
    p = eval_results.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = eval_results.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
        if iouThr == 0.75:
            P_50 = s[..., 0].squeeze()[50]
    else:
        # dimension of recall: [TxKxAxM]
        s = eval_results.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))

    if iouThr == 0.75:
        return mean_s, P_50

    if iouThr == 0.5:
        return mean_s

    return mean_s


if __name__ == "__main__":
    from evaluation.coco_custom import CustomCOCO

    gt_path = './gt/annotations_val.csv'
    pred_path = './pred/predictions_val.csv'

    gt_json, imgIds = get_gt_json(gt_path)
    dt_json = get_dt_json(pred_path, imgIds)

    gt_coco_format = CustomCOCO(gt_json)
    dt_coco_format = CustomCOCO(dt_json)

    # running evaluation
    cocoEval = COCOeval(gt_coco_format, dt_coco_format, iouType='bbox')
    cocoEval.params.maxDets = [300, 300, 300]

    # if you want to evaluate on a subset
    # cocoEval.params.imgIds  = range(0,100)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print('AP for 300 detections.')
    mAP = summarize(cocoEval, ap=1, maxDets=300)
    mAP_50, P_50_ = summarize(cocoEval, ap=1, maxDets=300, iouThr=0.5)
    mAP_75, P_50 = summarize(cocoEval, ap=1, maxDets=300, iouThr=0.75)
    AR300 = summarize(cocoEval, ap=0, maxDets=300)

    print("AP : ", mAP)
    print("AP.50 : ", mAP_50)
    print("AP.75 : ", mAP_75)
    print("AR300 : ", AR300)
    print("P_50 : ", P_50)
    print("P_50_ : ", P_50_)