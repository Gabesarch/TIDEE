import cv2
import numpy as np
import torch
import torch.nn.functional as F 
from PIL import Image
# import hyperparams as hyp
from arguments import args
# from util import box_ops

import ipdb
st = ipdb.set_trace


def visualize_boxes(img_, pred_boxes, pred_labels, score_img, iou_img, precision_img, recall_img,vis_name, obj_cur, gt_boxes, gt_labels, id_to_name, summ_writer, class_agnostic=False):
    confidence=0.5; rect_th=1; text_size=0.5; text_th=1
    img = img_.copy()
    for i in range(len(pred_boxes)):
        # rgb_mask = self.get_coloured_mask(pred_masks[i])
        if class_agnostic: 
            pred_class_name = 'Object'
        else:
            pred_class_name = id_to_name[int(pred_labels[i])]

        if pred_class_name != obj_cur:
            continue

        # pred_score = pred_scores[i]
        # alpha = 0.7
        # beta = (1.0 - alpha)
        # img = cv2.addWeighted(img.astype(np.int32), 1.0, rgb_mask.astype(np.int32), 0.5, 0.0)
        # where_mask = rgb_mask>0
        # img[where_mask] = rgb_mask[where_mask]
        cv2.rectangle(img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])), (int(pred_boxes[i][2]), int(pred_boxes[i][3])),(0, 255, 0), rect_th)
        cv2.putText(img,pred_class_name, (int(pred_boxes[i][0:1]), int(pred_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'score: ' + str(float(score_img)), (int(20), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'best iou: ' + str(iou_img), (int(20), int(40)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'precision: ' + str(precision_img), (int(20), int(60)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'recall: ' + str(recall_img), (int(20), int(80)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    img_torch1 = torch.from_numpy(img).cuda().permute(2,0,1) / 255. - 0.5
    img_torch1 = F.interpolate(img_torch1.unsqueeze(0), scale_factor=0.5, mode='bilinear')

    img = img_.copy()
    for i in range(len(pred_boxes)):
        # rgb_mask = self.get_coloured_mask(pred_masks[i])
        if class_agnostic: 
            pred_class_name = 'Object'
        else:
            pred_class_name = id_to_name[int(pred_labels[i])]

        # if pred_class_name != obj_cur:
        #     continue

        # pred_score = pred_scores[i]
        # alpha = 0.7
        # beta = (1.0 - alpha)
        # img = cv2.addWeighted(img.astype(np.int32), 1.0, rgb_mask.astype(np.int32), 0.5, 0.0)
        # where_mask = rgb_mask>0
        # img[where_mask] = rgb_mask[where_mask]
        cv2.rectangle(img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])), (int(pred_boxes[i][2]), int(pred_boxes[i][3])),(0, 255, 0), rect_th)
        cv2.putText(img,pred_class_name, (int(pred_boxes[i][0:1]), int(pred_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'score: ' + str(float(score_img)), (int(20), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'best iou: ' + str(iou_img), (int(20), int(40)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'precision: ' + str(precision_img), (int(20), int(60)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'recall: ' + str(recall_img), (int(20), int(80)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    img_torch2 = torch.from_numpy(img).cuda().permute(2,0,1) / 255. - 0.5
    img_torch2 = F.interpolate(img_torch2.unsqueeze(0), scale_factor=0.5, mode='bilinear')

    img = img_.copy()
    for i in range(len(gt_boxes)):
        # rgb_mask = self.get_coloured_mask(gt_masks[i])
        if class_agnostic: 
            gt_class_name = 'Object'
        else:
            gt_class_name = id_to_name[int(gt_labels[i])]

        if gt_class_name != obj_cur:
            continue

        # alpha = 0.7
        # beta = (1.0 - alpha)
        # img = cv2.addWeighted(img.astype(np.int32), 1.0, rgb_mask.astype(np.int32), 0.5, 0.0)
        # where_mask = rgb_mask>0
        # img[where_mask] = rgb_mask[where_mask]
        cv2.rectangle(img, (int(gt_boxes[i][0]), int(gt_boxes[i][1])), (int(gt_boxes[i][2]), int(gt_boxes[i][3])),color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,gt_class_name, (int(gt_boxes[i][0:1]), int(gt_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    img_torch3 = torch.from_numpy(img).cuda().permute(2,0,1) / 255. - 0.5
    img_torch3 = F.interpolate(img_torch3.unsqueeze(0), scale_factor=0.5, mode='bilinear')

    

    # img = img_.copy()
    # for i in range(len(gt_boxes)):
    #     # rgb_mask = self.get_coloured_mask(gt_masks[i])
    #     if class_agnostic: 
    #         gt_class_name = 'Object'
    #     else:
    #         gt_class_name = id_to_name[int(gt_labels[i])]

    #     # if gt_class_name != obj_cur:
    #     #     continue

    #     # alpha = 0.7
    #     # beta = (1.0 - alpha)
    #     # img = cv2.addWeighted(img.astype(np.int32), 1.0, rgb_mask.astype(np.int32), 0.5, 0.0)
    #     # where_mask = rgb_mask>0
    #     # img[where_mask] = rgb_mask[where_mask]
    #     cv2.rectangle(img, (int(gt_boxes[i][0]), int(gt_boxes[i][1])), (int(gt_boxes[i][2]), int(gt_boxes[i][3])),color=(0, 255, 0), thickness=rect_th)
    #     cv2.putText(img,gt_class_name, (int(gt_boxes[i][0:1]), int(gt_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    # img_torch3 = torch.from_numpy(img).cuda().permute(2,0,1).unsqueeze(0) / 255. - 0.5

    img_torch = torch.cat([img_torch1, img_torch2, img_torch3], dim=3)
    summ_writer.summ_rgb(vis_name, img_torch)

@torch.no_grad()
def check_for_detections(rgb, ddetr, W, H, score_labels_name, score_boxes_name, score_threshold_ddetr=0.0, do_nms=True, return_features=False, target_object=None, target_object_score_threshold=None, solq=False, return_masks=False, nms_threshold=0.2, id_to_mapped_id=None):
    '''
    rgb: single rgb image
    NOTE: currently this only handles a single RGB
    '''
    
    rgb_PIL = Image.fromarray(rgb)
    rgb_norm = rgb.astype(np.float32) * 1./255
    rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1).cuda()
    
    # self.ddetr.cuda()
    if solq:
        outputs = ddetr(rgb_torch.unsqueeze(0), do_loss=False, return_features=return_features)
    else:
        outputs = ddetr([rgb_torch], do_loss=False, return_features=return_features)
    # self.ddetr.cpu()
    
    out = outputs['outputs']
    if return_features:
        features = out['features']
    else:
        features = None
    postprocessors = outputs['postprocessors']

    # if return_features:
    #     predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=True)
    #     features = features[0]
    # else:
    if return_features:
        predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
        features = features[0]
        # if hyp.do_segmentation:
        #     predictions = postprocessors['segm'](predictions, out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda())  
    else:
        predictions = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
    
    pred_boxes = predictions[score_labels_name][0]['boxes']
    pred_labels = predictions[score_labels_name][0]['labels']
    pred_scores_boxes = predictions[score_boxes_name][0]['scores']
    pred_scores_labels = predictions[score_labels_name][0]['scores']
    if return_masks:
        pred_masks = predictions[score_labels_name][0]['masks']
        
    if pred_boxes.shape[0]>1:
        
        if do_nms:
            keep, count = nms(pred_boxes, pred_scores_boxes, nms_threshold, top_k=100)
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            # pred_scores = pred_scores[keep]
            pred_scores_boxes = pred_scores_boxes[keep]
            pred_scores_labels = pred_scores_labels[keep]
            if return_masks:
                pred_masks = pred_masks[keep]
            if return_features:
                features = features[keep]

        pred_boxes = pred_boxes.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores_boxes = pred_scores_boxes.cpu().numpy() 
        pred_scores_labels = pred_scores_labels.cpu().numpy() 
        if return_masks:
            pred_masks = pred_masks.squeeze(1).cpu().numpy() 

        # pred_boxes = pred_boxes.cpu().numpy()
        # pred_labels = pred_labels.cpu().numpy()
        # pred_scores = pred_scores.cpu().numpy()

        # above score threshold
        if target_object is not None:
            if type(target_object)==int:
                keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
            elif type(target_object)==list:
                # keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
                keep_target = []
                for lab_i in range(len(pred_labels)):
                    check1 = pred_scores_labels[lab_i]>target_object_score_threshold
                    check2 = pred_labels[lab_i] in target_object
                    # check if above score and label in target_object list
                    if check1 and check2:
                        keep_target.append(True)
                    else:
                        keep_target.append(False)
                keep_target = np.array(keep_target)
            else:
                assert(False) # target_object should be int or list of ints
            keep = keep_target #np.logical_and(keep_target, keep)
        else:
            keep = pred_scores_labels>score_threshold_ddetr
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores_labels = pred_scores_labels[keep]
        if return_features:
            features = features[keep]
        if return_masks:
            pred_masks = pred_masks[keep]

        if id_to_mapped_id is not None:
            # map desired labels to another label
            for idx in range(len(pred_labels)):
                if pred_labels[idx] in id_to_mapped_id.keys():
                    pred_labels[idx] = id_to_mapped_id[pred_labels[idx]]

        # labels_in_view = [self.id_to_name[pred_labels_] for pred_labels_ in list(pred_labels)]
        # print("labels in view:")
        # print(labels_in_view)

    out = {}
    out["pred_labels"] = pred_labels
    out["pred_boxes"] = pred_boxes
    out["pred_scores"] = pred_scores_labels
    if return_masks:
        out["pred_masks"] = pred_masks
    if return_features:
        out["features"] = features

    return out
    # if return_features:
    #     return pred_scores_labels, pred_labels, pred_boxes, features
    # else:
    #     return pred_scores_labels, pred_labels, pred_boxes


@torch.no_grad()
def check_for_detections_two_head(
    rgb, 
    ddetr,
     W, H, 
     score_labels_name, 
     score_labels_name2, 
     score_threshold_head1=0.0, 
     score_threshold_head2=0.0, 
     do_nms=True, 
     return_features=False, 
     target_object=None, 
     target_object_score_threshold=None, 
     solq=False, 
     return_masks=False, 
     nms_threshold=0.2, 
     id_to_mapped_id=None
     ):
    '''
    rgb: single rgb image
    NOTE: currently this only handles a single RGB
    '''
    
    rgb_PIL = Image.fromarray(rgb)
    rgb_norm = rgb.astype(np.float32) * 1./255
    rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1).cuda()
    
    # self.ddetr.cuda()
    if solq:
        outputs = ddetr(rgb_torch.unsqueeze(0), do_loss=False, return_features=return_features)
    else:
        outputs = ddetr([rgb_torch], do_loss=False, return_features=return_features)
    # self.ddetr.cpu()
    
    out = outputs['outputs']
    if return_features:
        features = out['features']
    else:
        features = None
    postprocessors = outputs['postprocessors']

    # if return_features:
    #     predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=True)
    #     features = features[0]
    # else:
    if return_features:
        predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
        features = features[0]
        # if hyp.do_segmentation:
        #     predictions = postprocessors['segm'](predictions, out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda())  
    else:
        predictions = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
    
    count = 0

    pred_boxes = predictions[score_labels_name][0]['boxes']
    pred_labels = predictions[score_labels_name][0]['labels']
    pred_scores_boxes = predictions[score_labels_name][0]['scores']
    pred_scores_labels = predictions[score_labels_name][0]['scores']
    if return_masks:
        pred_masks = predictions[score_labels_name][0]['masks']

    pred_boxes2 = predictions[score_labels_name2][0]['boxes']
    pred_labels2 = predictions[score_labels_name2][0]['labels']
    pred_scores_boxes2 = predictions[score_labels_name2][0]['scores']
    pred_scores_labels2 = predictions[score_labels_name2][0]['scores']
    if return_masks:
        pred_masks2 = predictions[score_labels_name2][0]['masks']
        
        
    if pred_boxes.shape[0]>1:
        
        if do_nms:
            keep, count = nms(pred_boxes, pred_scores_boxes, nms_threshold, top_k=100)
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            # pred_scores = pred_scores[keep]
            pred_scores_boxes = pred_scores_boxes[keep]
            pred_scores_labels = pred_scores_labels[keep]
            if return_masks:
                pred_masks = pred_masks[keep]
            if return_features:
                features = features[keep]

            pred_boxes2 = pred_boxes2[keep]
            pred_labels2 = pred_labels2[keep]
            pred_scores_boxes2 = pred_scores_boxes2[keep]
            pred_scores_labels2 = pred_scores_labels2[keep]
            if return_masks:
                pred_masks2 = pred_masks2[keep]

        pred_boxes = pred_boxes.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores_boxes = pred_scores_boxes.cpu().numpy() 
        pred_scores_labels = pred_scores_labels.cpu().numpy() 
        if return_masks:
            pred_masks = pred_masks.squeeze(1).cpu().numpy() 

        pred_boxes2 = pred_boxes2.cpu().numpy()
        pred_labels2 = pred_labels2.cpu().numpy()
        pred_scores_boxes2 = pred_scores_boxes2.cpu().numpy()
        pred_scores_labels2 = pred_scores_labels2.cpu().numpy()
        if return_masks:
            pred_masks2 = pred_masks2.cpu().numpy()

        # pred_boxes = pred_boxes.cpu().numpy()
        # pred_labels = pred_labels.cpu().numpy()
        # pred_scores = pred_scores.cpu().numpy()

        # above score threshold
        keep = pred_scores_labels>score_threshold_head1
        if target_object is not None:
            keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
            keep = np.logical_or(keep_target, keep)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores_labels = pred_scores_labels[keep]
        if return_features:
            features = features[keep]
        if return_masks:
            pred_masks = pred_masks[keep]

        pred_boxes2 = pred_boxes2[keep]
        pred_labels2 = pred_labels2[keep]
        pred_scores_boxes2 = pred_scores_boxes2[keep]
        pred_scores_labels2 = pred_scores_labels2[keep]
        if return_masks:
            pred_masks2 = pred_masks2[keep]


        # above score threshold head 2
        keep = pred_scores_labels2>score_threshold_head2
        if target_object is not None:
            keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
            keep = np.logical_or(keep_target, keep)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores_labels = pred_scores_labels[keep]
        if return_features:
            features = features[keep]
        if return_masks:
            pred_masks = pred_masks[keep]

        pred_boxes2 = pred_boxes2[keep]
        pred_labels2 = pred_labels2[keep]
        pred_scores_boxes2 = pred_scores_boxes2[keep]
        pred_scores_labels2 = pred_scores_labels2[keep]
        if return_masks:
            pred_masks2 = pred_masks2[keep]

        if id_to_mapped_id is not None:
            # map desired labels to another label
            for idx in range(len(pred_labels)):
                if pred_labels[idx] in id_to_mapped_id.keys():
                    pred_labels[idx] = id_to_mapped_id[pred_labels[idx]]

        # labels_in_view = [self.id_to_name[pred_labels_] for pred_labels_ in list(pred_labels)]
        # print("labels in view:")
        # print(labels_in_view)

        out = {}
        out["pred_labels"] = pred_labels
        out["pred_boxes"] = pred_boxes
        out["pred_scores"] = pred_scores_labels
        if return_masks:
            out["pred_masks"] = pred_masks
        if return_features:
            out["features"] = features
        out["pred_labels2"] = pred_labels2
        out["pred_boxes2"] = pred_boxes2
        out["pred_scores2"] = pred_scores_labels2
        if return_masks:
            out["pred_masks2"] = pred_masks2

    return out


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    # keep = scores.new(scores.size(0)).zero_().long()
    keep = []
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        # keep[count] = i
        keep.append(i)
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        # with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", category=UserWarning)
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter.float() / union.float()  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    keep = torch.tensor(keep)
    return keep, count