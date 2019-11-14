import torch
import numpy as np


# PyTroch version
SMOOTH = 1e-6

def iou_pytorch(logits: torch.Tensor, labels: torch.Tensor):
    if len(logits.size())==4:
        outputs = (logits > 0).byte().squeeze(1)
    else:
        outputs = (logits > 0).byte()
    if len(labels.size())==4:
        labels = labels.byte().squeeze(1)
    else:
        labels = labels.byte()
    intersection = (outputs & labels).float().sum(dim=1).sum(dim=1)  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(dim=1).sum(dim=1)         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # average across the batch


def dice(logit:torch.Tensor, truth:torch.Tensor, iou:bool=False, eps:float=1e-8, 
         nonempty_only=False, logit_clf=None):#->Rank0Tensor
    """
    A slight modification of the default dice metric to make it comparable with the competition metric: 
    dice is computed for each image independently, and dice of empty image with zero prediction is 1. 
    Also I use noise removal and similar threshold as in my prediction pipline.
    """
    if nonempty_only:
        n, c = truth.shape[0], truth.shape[1]
        logit = torch.sigmoid(logit).view(n*c, -1)#reformat to: sample1,channel1,2,3,4; sample2,channel1,2,3,4; ......
        truth = truth.view(n*c, -1).long()
        ## select nonempty channels
        is_nonempty = truth.sum(dim=[1])>0
        logit = logit[is_nonempty]
        truth = truth[is_nonempty]
        
        #MASK_THRESHOLD = 0.5 #softmax>threshold, predict a mask=1
        best_score = 0
        for MASK_THRESHOLD in np.arange(0.1, 0.9, 0.02):
            pred = (logit>MASK_THRESHOLD).long()

            intersect = (pred * truth).sum(dim=1).float()
            union = (pred + truth).sum(dim=1).float()
            if not iou:
                score = (2.0*intersect / union).mean()
            else:
                score = (intersect / (union - intersect)).mean()
            
            if score>best_score:
                best_score = score
        return score

    h,w = truth.shape[2], truth.shape[3]#256, 1600
    if logit_clf is None:
        EMPTY_THRESHOLD = int(20000 * (h/512) * (w/768)) #count of predicted mask pixles<threshold, predict as empty mask image
        MASK_THRESHOLD = 0.7 #>threshold, predict a mask=1
    else:
        EMPTY_THRESHOLD = 1
        MASK_THRESHOLD = 0.3
        CLF_THRESHOLD = 0.7 #<threshold, predict empty-mask
    
    n, c = truth.shape[0], truth.shape[1]
    logit = torch.sigmoid(logit).view(n*c, -1)#reformat to: sample1,channel1,2,3,4; sample2,channel1,2,3,4; ......
    if logit_clf is not None:
        logit_clf = torch.sigmoid(logit_clf).view(n*c, -1)
    truth = truth.view(n*c, -1).long()
    
    pred = (logit>MASK_THRESHOLD).long()
    pred[pred.sum(dim=1) < EMPTY_THRESHOLD, ] = 0
    if logit_clf is not None:
        pred[logit_clf.squeeze()<CLF_THRESHOLD, ] = 0
    
    ## the correct LB metric: if both GT and pred empty mask image, then dice score=1
    is_empty = (truth.sum(dim=1)==0) * (pred.sum(dim=1)==0)
    truth_pos = truth[1-is_empty]#check is it correct slicing in torch??
    pred_pos = pred[1-is_empty]
    intersect_pos = (pred_pos * truth_pos).sum(dim=1).float()
    union_pos = (pred_pos + truth_pos).sum(dim=1).float()
    if not iou:
        dice_pos = ((2.0*intersect_pos + eps) / (union_pos+eps)).sum()
    else:
        dice_pos = ((intersect_pos + eps) / (union_pos - intersect_pos + eps)).sum()
    return (dice_pos + is_empty.sum()) / truth.size()[0]

    #my raw implementation
#     intersect = (pred * truth).sum(dim=1).float()
#     union = (pred + truth).sum(dim=1).float()
#     if not iou:
#         return ((2.0*intersect + eps) / (union+eps)).mean()
#     else:
#         return ((intersect + eps) / (union - intersect + eps)).mean()

# def dice_optimal(logit:torch.Tensor, truth:torch.Tensor, iou:bool=False, eps:float=1e-8):#->Rank0Tensor
#     """
#     A slight modification of the default dice metric to make it comparable with the competition metric: 
#     dice is computed for each image independently, and dice of empty image with zero prediction is 1. 
#     Also I use noise removal and similar threshold as in my prediction pipline.
#     """
#     h,w = truth.shape[2], truth.shape[3]#256, 1600
#     EMPTY_THRESHOLD = 10000*(h/1400)*(w/2100) #count of predicted mask pixles<threshold, predict as empty mask image
#     MASK_THRESHOLD = 0.9 #0.5 #softmax>threshold, predict a mask=1
    
#     n, c = truth.shape[0], truth.shape[1]
#     logit = torch.sigmoid(logit).view(n*c, -1)#reformat to: sample1,channel1,2,3,4; sample2,channel1,2,3,4; ......
#     truth = truth.view(n*c, -1).long()
    
#     pred = (logit>MASK_THRESHOLD).long()
#     pred[pred.sum(dim=1) < EMPTY_THRESHOLD, ] = 0
    
#     intersect = (pred * truth).sum(dim=1).float()
#     union = (pred + truth).sum(dim=1).float()
#     if not iou:
#         return ((2.0*intersect + eps) / (union+eps)).mean()
#     else:
#         return ((intersect + eps) / (union - intersect + eps)).mean()

def dice_deep_supervision(logit:torch.Tensor, truth:torch.Tensor, iou:bool=False, eps:float=1e-8):#->Rank0Tensor
    IMG_SIZE = truth.shape[-1]#256
    MASK_THRESHOLD = 0.5 #softmax>threshold, predict a mask=1
    
    n = truth.shape[0]
    if len(logit.size())==4:
        logit = logit.squeeze(1)
    if len(truth.size())==4:
        truth = truth.squeeze(1)
    logit = torch.sigmoid(logit).view(n, -1)
    pred = (logit>MASK_THRESHOLD).long()
    truth = truth.view(n, -1).long()

    intersect = (pred * truth).sum(dim=1).float()
    union = (pred + truth).sum(dim=1).float()
    if not iou:
        return ((2.0*intersect + eps) / (union+eps)).mean()
    else:
        return ((intersect + eps) / (union - intersect + eps)).mean()

def dice_multitask(logit_mask, truth_mask, logit_clf, truth_clf, iou=False, eps=1e-8):
    """
    1. use logit_clf>0.75 to split empty-mask/nonempty-mask samples
    2. replace empty-mask samples' logit_mask with zeros
    3. calculate dice score on all samples
    TODO: show classification accuracy
    """
    IMG_SIZE = truth_mask.shape[-1]#256
    MASK_THRESHOLD = 0.22 #>threshold, predict a mask=1
    CLF_THRESHOLD = 0.75 #>threshold, predict nonempty-mask image
    
    n = truth_mask.shape[0]
    logit_mask = torch.sigmoid(logit_mask).view(n, -1)
    logit_clf = torch.sigmoid(logit_clf).view(n, -1)
    is_nonempty = (logit_clf>CLF_THRESHOLD).float()
    logit_mask = logit_mask * is_nonempty
    pred_mask = (logit_mask>MASK_THRESHOLD).long()
    truth_mask = truth_mask.view(n, -1).long()
    
    intersect = (pred_mask * truth_mask).sum(dim=1).float()
    union = (pred_mask + truth_mask).sum(dim=1).float()
    if not iou:
        return ((2.0*intersect + eps) / (union+eps)).mean()
    else:
        return ((intersect + eps) / (union - intersect + eps)).mean()

# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()


##=============================================================================================================##
#Score the model and do a threshold optimization by the best IoU.

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in


    true_objects = 2
    pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
#     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    #print(temp1)
    intersection = temp1[0]
    #print("temp2 = ",temp1[1])
    #print(intersection.shape)
   # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    #print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    #print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)