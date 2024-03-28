from importlib import import_module
import numpy as np
import torch
import torchmetrics
import pytorch_lightning as pl
from metrics.skeletal_similarity import SkeletalSimilarity
from metrics.cal_score import CalScore
class MeInterface(pl.LightningModule):
    def __init__(self, metric_names, **kwargs):
        super(MeInterface, self).__init__()
        self.metric_names = metric_names
        self.kwargs = kwargs
        self.metric_functions = {}
        for metric_name in self.metric_names:
            if metric_name.lower() == 'f1':
                f1_score = torchmetrics.F1Score(task='binary').to(self.device)
                self.metric_functions['f1'] = f1_score
            if metric_name.lower() == 'iou':
                iou = torchmetrics.JaccardIndex(task='binary').to(self.device)
                self.metric_functions['iou'] = iou
            if metric_name.lower() == 'auroc':
                auroc = torchmetrics.AUROC(task='binary').to(self.device)
                self.metric_functions['auroc'] = auroc
            if metric_name.lower() == 'accuracy':
                accuracy = torchmetrics.Accuracy(task='binary').to(self.device)
                self.metric_functions['accuracy'] = accuracy
            if metric_name.lower() == 'precision':
                precision = torchmetrics.Precision(task='binary').to(self.device)
                self.metric_functions['precision'] = precision
            if metric_name.lower() == 'specificity':
                specificity = torchmetrics.Specificity(task='binary').to(self.device)
                self.metric_functions['specificity'] = specificity
            if metric_name.lower() == 'recall':
                sensitivity = torchmetrics.Recall(task='binary').to(self.device)
                self.metric_functions['recall'] = sensitivity
            if metric_name.lower() == 'mcc':
                mcc = torchmetrics.MatthewsCorrCoef(task='binary').to(self.device)
                self.metric_functions['mcc'] = mcc
            if metric_name.lower() == 'calscore':
                self.metric_functions['calscore'] = CalScore()

    @torch.no_grad()
    def calc_metrics(self, preds, labels, roi=None):
        """
        Calculate metrics
        :param preds: N, C, H, W
        :param labels: N, H, W
        :param roi:
        :return:
        """
        for name, metric in self.metric_functions.items():
            if name != 'skeletal_similarity' and name != 'calscore':
                metric.to(self.device)
        if len(preds.shape) == 4:
            _, preds = torch.max(preds, dim=1) # N, C, H, W -> N, H, W
        if roi is not None:
            y_pred_masked = torch.masked_select(preds, roi)
        else:
            y_pred_masked = preds
        if len(labels.shape) == 4:
            labels = labels.squeeze(1)


        results = {}
        for metric_name in self.metric_names:
            if metric_name.lower() == 'f1':
                dsc = self.metric_functions['f1'](preds, labels.long())
                results['f1'] = dsc
            elif metric_name.lower() == 'iou':
                iou = self.metric_functions['iou'](preds, labels.long())
                results['iou'] = iou
            elif metric_name.lower() == 'auroc':
                aucroc = self.metric_functions['auroc'](preds, labels.long())
                results['auroc'] = aucroc
            elif metric_name.lower() == 'accuracy':
                acc = self.metric_functions['accuracy'](preds, labels.long())
                results['accuracy'] = acc
            elif metric_name.lower() == 'precision':
                precision = self.metric_functions['precision'](preds, labels.long())
                results['precision'] = precision
            elif metric_name.lower() == 'specificity':
                specificity = self.metric_functions['specificity'](preds, labels.long())
                results['specificity'] = specificity
            elif metric_name.lower() == 'recall':
                sensitivity = self.metric_functions['recall'](preds, labels.long())
                results['recall'] = sensitivity
            elif metric_name.lower() == 'mcc':
                mcc = self.metric_functions['mcc'](preds, labels.long())
                results['mcc'] = mcc
            elif metric_name.lower() == 'skeletal_similarity':
                rSe, rSp, rAcc, SS= SkeletalSimilarity(preds.cpu().numpy().astype(np.uint8), labels.cpu().numpy().astype(np.uint8))
                results['rSe'] = rSe
                results['rSp'] = rSp
                results['rAcc'] = rAcc
                results['SS'] = SS
            elif metric_name.lower() == 'calscore':
                connects, area, length, calscore = self.metric_functions['calscore'].cal_metric_batch(preds, labels)
                results['connects'] = connects
                results['area'] = area
                results['length'] = length
                results['calscore'] = calscore
        del preds, labels
        torch.cuda.empty_cache()
        return results