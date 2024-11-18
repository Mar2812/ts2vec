import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def get_ks(y_true,y_pred):
    fpr,tpr,thresholds=roc_curve(y_true,y_pred)
    ks=max(tpr-fpr)
    return ks

def get_auc(y_true,y_pred):
    auc=roc_auc_score(y_true,y_pred)
    return auc

def eval_classification(score, labels):
    pred = score.argmax(axis=1)
    acc = (pred == labels).mean()

    # 计算AUPRC（平均精确率）分数
    num_classes = labels.max() + 1  # 类别总数
    test_labels_onehot = np.eye(int(num_classes))[labels.astype(int)]
    auprc = average_precision_score(test_labels_onehot, score)
    
    # 计算AUC（曲线下面积）
    auc = get_auc(labels, score[:,1])  # 取类别1的概率值来计算AUC

    # 计算KS（Kolmogorov-Smirnov统计量）
    ks = get_ks(labels, score[:, 1])  # 取类别1的概率值来计算KS

    return {'acc': acc, 'auprc': auprc, 'auc': auc, 'ks': ks}
