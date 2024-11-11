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

def eval_classification(train_repr, train_labels, test_repr, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    # 选择分类器训练函数
    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        raise ValueError('unknown evaluation protocol')

    # 合并第一、第二个维度（仅适用于多维标签情况）
    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    # 训练分类器
    clf = fit_clf(train_repr, train_labels)

    # 计算准确率
    acc = clf.score(test_repr, test_labels)

    # 根据分类器类型获取得分
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)

    # 计算AUPRC（平均精确率）分数
    num_classes = train_labels.max() + 1  # 类别总数
    test_labels_onehot = np.eye(int(num_classes))[test_labels.astype(int)]
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    # 计算AUC（曲线下面积）
    auc = get_auc(test_labels, y_score[:, 1])  # 取类别1的概率值来计算AUC

    # 计算KS（Kolmogorov-Smirnov统计量）
    ks = get_ks(test_labels, y_score[:, 1])  # 取类别1的概率值来计算KS

    return y_score, {'acc': acc, 'auprc': auprc, 'auc': auc, 'ks': ks}
