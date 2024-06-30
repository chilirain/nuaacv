from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

# 假设 pred_boxes 和 gt_boxes 是预测的边界框和真实的边界框
# 预测格式: [{'boxes': [[x1, y1, x2, y2], ...], 'scores': [score1, score2, ...]}, ...]
# 真实格式: [[x1, y1, x2, y2], ...]

def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def compute_average_precision(pred_boxes, gt_boxes, iou_threshold=0.5):
    all_scores = []
    all_tp_fp = []

    for pred, gt in zip(pred_boxes, gt_boxes):
        scores = pred['scores']
        pred_boxes_list = pred['boxes']
        
        tp_fp_labels = []
        for pred_box in pred_boxes_list:
            best_iou = 0
            for gt_box in gt:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou

            if best_iou >= iou_threshold:
                tp_fp_labels.append(1)  # True Positive
            else:
                tp_fp_labels.append(0)  # False Positive
        
        all_scores.extend(scores)
        all_tp_fp.extend(tp_fp_labels)

    precision, recall, _ = precision_recall_curve(all_tp_fp, all_scores)
    ap = average_precision_score(all_tp_fp, all_scores)

    return ap, precision, recall