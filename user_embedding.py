from ts2vec import TS2Vec
import datautils
import torch
from tasks.classification import eval_classification
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re
import numpy as np
from tasks.classification import eval_classification
from sklearn.utils import resample


def preprocess_data(df, continue_col, categorical_col):
    # 提取时间戳的最大值，用于后续的3D张量转换
    time_suffixes = sorted(set(int(col.split('_')[-1]) for col in df.columns if '_' in col and col.split('_')[-1].isdigit()))
    max_timestamp = max(time_suffixes)

    # 分离连续特征和离散特征
    continuous_cols = [col for col in df.columns if any(tag in col for tag in continue_col)]
    categorical_cols = [col for col in df.columns if any(tag in col for tag in categorical_col)]

    # 处理连续特征 - 标准化
    scaler = StandardScaler()
    df[continuous_cols] = df[continuous_cols].astype(float)
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols].fillna(0))
    print("ok")
    # 处理离散特征 - One-Hot 编码
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_data = encoder.fit_transform(df[categorical_cols].fillna('unknown'))

    # 将One-Hot编码结果转换为DataFrame，方便后续操作
    categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    # 合并处理后的连续和离散特征
    df_processed = df[continuous_cols].join(categorical_df)

    # 提取所有列的时间戳
    feature_timestamp_map = {}
    for col in df_processed.columns:
        match = re.search(r'(\d+)', col)
        if match:
            timestamp = int(match.group(0))
            feature_timestamp_map.setdefault(timestamp, []).append(col)
    
    # 对每个时间戳下的特征列表进行排序，以确保顺序一致
    for timestamp in feature_timestamp_map:
        feature_timestamp_map[timestamp] = sorted(feature_timestamp_map[timestamp])

    # 确定三维数组的维度
    rows = len(df_processed)
    timestamps = len(feature_timestamp_map)
    features_per_timestamp = max(len(cols) for cols in feature_timestamp_map.values())
    result_array = np.full((rows, timestamps, features_per_timestamp), 0)

    # 填充三维数组
    for t_idx, (timestamp, cols) in enumerate(sorted(feature_timestamp_map.items())):
        for f_idx, col in enumerate(cols):
            result_array[:, t_idx, f_idx] = df_processed[col].values
    
    return np.nan_to_num(result_array, nan=0.0)

def load_data(dataset=None):
    if not dataset:
        raise ValueError("未输入数据源路径")
    # 读取数据
    data = pd.read_csv(dataset)

    # 平衡样本
    def balanced(data):
        # 分离标签为1和标签为0的样本
        data_minority = data[data['target'] == 1]
        data_majority = data[data['target'] == 0]

        # 设置倍数x，例如我们选择少数类的5倍数量
        x = 3

        # 从多数类样本中随机选择少数类样本数量的x倍
        data_majority_oversampled = resample(data_majority, 
                                            replace=False,    # 不放回采样
                                            n_samples=len(data_minority) * x,  # 使多数类样本数为少数类样本的x倍
                                            random_state=42)  # 保证结果可复现

        # 合并平衡后的数据集
        data_balanced = pd.concat([data_majority_oversampled, data_minority])
        return data_balanced
    
    # 分割数据集
    data_train = data[data['dataset'] == 'trainSet']
    data_train = balanced(data_train)
    print("训练样本分布")
    print(data_train['target'].value_counts())
    labels_train = data_train["target"].values
    data_test = data[data['dataset'] == 'valSet']
    labels_test = data_test["target"].values
    data_oot = data[data['dataset'] == 'ootSet']
    labels_oot = data_oot["target"].values

    # 定义不同维度的特征前缀（不带数字后缀）
    deduction_prefix = "daikou"
    payment_prefix = "daifu"
    invocation_prefix = "diaoyong"

    # 使用前缀匹配来提取不同特征
    deduction_columns = [col for col in data_train.columns if deduction_prefix in col]
    payment_columns = [col for col in data_train.columns if payment_prefix in col]
    invocation_columns = [col for col in data_train.columns if invocation_prefix in col]

    # 为了确保预处理函数的一致性，保证每个特征维度都有相同的列
    for df in [data_train, data_test, data_oot]:
        df["daikou_datediff_nextdate_1"] = 0
        df["daifu_datediff_nextdate_1"] = 0
        df["diaoyong_datediff_nextdate_1"] = 0

    # 对训练集、验证集和OOT集分别进行特征预处理
    # diaoyong 特征预处理
    diaoyong_features_train = preprocess_data(
        data_train[invocation_columns + ["diaoyong_datediff_nextdate_1"]],
        continue_col=["datediff_loandate", "member_times", "datediff_member", "datediff_nextdate"],
        categorical_col=[]
    )
    diaoyong_features_test = preprocess_data(
        data_test[invocation_columns + ["diaoyong_datediff_nextdate_1"]],
        continue_col=["datediff_loandate", "member_times", "datediff_member", "datediff_nextdate"],
        categorical_col=[]
    )
    diaoyong_features_oot = preprocess_data(
        data_oot[invocation_columns + ["diaoyong_datediff_nextdate_1"]],
        continue_col=["datediff_loandate", "member_times", "datediff_member", "datediff_nextdate"],
        categorical_col=[]
    )

    # daikou 特征预处理
    daikou_features_train = preprocess_data(
        data_train[deduction_columns + ["daikou_datediff_nextdate_1"]],
        continue_col=["order_money", "datediff_loandate", "member_times", "flag_times", "memberflag_times", "datediff_member", "datediff_flag", "datediff_memberflag", "datediff_nextdate"],
        categorical_col=["succ_flag", "member_name"]
    )
    daikou_features_test = preprocess_data(
        data_test[deduction_columns + ["daikou_datediff_nextdate_1"]],
        continue_col=["order_money", "datediff_loandate", "member_times", "flag_times", "memberflag_times", "datediff_member", "datediff_flag", "datediff_memberflag", "datediff_nextdate"],
        categorical_col=["succ_flag", "member_name"]
    )
    daikou_features_oot = preprocess_data(
        data_oot[deduction_columns + ["daikou_datediff_nextdate_1"]],
        continue_col=["order_money", "datediff_loandate", "member_times", "flag_times", "memberflag_times", "datediff_member", "datediff_flag", "datediff_memberflag", "datediff_nextdate"],
        categorical_col=["succ_flag", "member_name"]
    )

    # daifu 特征预处理
    daifu_features_train = preprocess_data(
        data_train[payment_columns + ["daifu_datediff_nextdate_1"]],
        continue_col=["transfer_money", "datediff_loandate", "member_times", "datediff_member", "datediff_nextdate"],
        categorical_col=["member_name"]
    )
    daifu_features_test = preprocess_data(
        data_test[payment_columns + ["daifu_datediff_nextdate_1"]],
        continue_col=["transfer_money", "datediff_loandate", "member_times", "datediff_member", "datediff_nextdate"],
        categorical_col=["member_name"]
    )
    daifu_features_oot = preprocess_data(
        data_oot[payment_columns + ["daifu_datediff_nextdate_1"]],
        continue_col=["transfer_money", "datediff_loandate", "member_times", "datediff_member", "datediff_nextdate"],
        categorical_col=["member_name"]
    )

    # 返回独立的特征和标签
    return (
        diaoyong_features_train, daikou_features_train, daifu_features_train, labels_train,
        diaoyong_features_test, daikou_features_test, daifu_features_test, labels_test,
        diaoyong_features_oot, daikou_features_oot, daifu_features_oot, labels_oot
    )

# 使用示例
import time

time_begin = time.time()
data_path = r"/home/mc/ts2vec/ts2vec/datasets/wfplus_application_1718475236931_781952_timefeattopresult_spark_tfmid.csv"
# data_path = r"C:\Users\chao_ma02\Desktop\work\ts2vec\datasets\wfplus_application_1718475236931_781952_timefeattopresult_spark_tfmid.csv"
(
    diaoyong_features_train, daikou_features_train, daifu_features_train, labels_train,
    diaoyong_features_test, daikou_features_test, daifu_features_test, labels_test,
    diaoyong_features_oot, daikou_features_oot, daifu_features_oot, labels_oot
) = load_data(data_path)
time_end = time.time()
print("data_load time:", time_end - time_begin)
print(diaoyong_features_train.shape)

# 使用TS2Vec模型对daikou, daifu, diaoyong特征分别进行编码
def encode_features(model, train_features, test_features, encoding_window='full_series'):
    train_repr = model.encode(train_features, encoding_window=encoding_window)
    test_repr = model.encode(test_features, encoding_window=encoding_window)
    return train_repr, test_repr

# 创建TS2Vec模型并对每个特征集进行编码
device = 0
# device = torch.device('cpu')
output_dims = 320

# daikou特征编码
model_daikou = TS2Vec(input_dims=daikou_features_train.shape[-1], device=device, output_dims=output_dims)
model_daikou.fit(daikou_features_train, verbose=True)
train_repr_daikou, test_repr_daikou = encode_features(model_daikou, daikou_features_train, daikou_features_test)

# daifu特征编码
model_daifu = TS2Vec(input_dims=daifu_features_train.shape[-1], device=device, output_dims=output_dims)
model_daifu.fit(daifu_features_train, verbose=True)
train_repr_daifu, test_repr_daifu = encode_features(model_daifu, daifu_features_train, daifu_features_test)

# diaoyong特征编码
model_diaoyong = TS2Vec(input_dims=diaoyong_features_train.shape[-1], device=device, output_dims=output_dims)
model_diaoyong.fit(diaoyong_features_train, verbose=True)
train_repr_diaoyong, test_repr_diaoyong = encode_features(model_diaoyong, diaoyong_features_train, diaoyong_features_test)

# 将编码后的特征拼接
train_repr = np.concatenate((train_repr_daikou, train_repr_daifu, train_repr_diaoyong), axis=-1)
test_repr = np.concatenate((test_repr_daikou, test_repr_daifu, test_repr_diaoyong), axis=-1)

# 调用eval_classification进行分类评估
y_score, eval_metrics = eval_classification(
    train_repr, labels_train,
    test_repr, labels_test,
    eval_protocol='linear'  # 可以调整为'linear'、'svm'或'knn'
)

# 打印评估结果
print("Evaluation Metrics:", eval_metrics)
print("Accuracy:", eval_metrics['acc'])
print("AUPRC:", eval_metrics['auprc'])