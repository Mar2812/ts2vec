from ts2vec import TS2Vec
import datautils
import torch
from tasks.classification import eval_classification
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re


def preprocess_data(df, continue_col, categorical_col):
    # 提取时间戳的最大值，用于后续的3D张量转换
    time_suffixes = sorted(set(int(col.split('_')[-1]) for col in df.columns if '_' in col and col.split('_')[-1].isdigit()))
    max_timestamp = max(time_suffixes)

    # 分离连续特征和离散特征
    continuous_cols = [col for col in df.columns if any(tag in col for tag in continue_col)]
    categorical_cols = [col for col in df.columns if any(tag in col for tag in categorical_col)]

    # 处理连续特征 - 标准化
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols].fillna(0))

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
    data = pd.read_csv(r"C:\Users\chao_ma02\Desktop\work\ts2vec\datasets\wfplus_application_1718475236931_781952_timefeattopresult_spark_tfmid.csv")
    
    data_train = data[data['dataset'] == 'trainSet']
    labels_train = data_train["target"].values
    data_test = data[data['dataset'] == 'valSet']
    labels_test = data_test["target"].values
    data_oot = data[data['dataset'] == 'ootSet']
    labels_oot = data_oot["target"].values
    # 定义不同维度的特征前缀（不带数字后缀）
    deduction_prefix = "daikou"
    payment_prefix = "daifu"
    invocation_prefix = "diaoyong"

    # 使用前缀匹配来提取代扣特征
    deduction_columns = [col for col in data_train.columns if deduction_prefix in col]
    df_deduction = data_train[deduction_columns].copy()
    df_deduction["daikou_datediff_nextdate_1"] = 0

    # 使用前缀匹配来提取代付特征
    payment_columns = [col for col in data_train.columns if payment_prefix in col]
    df_payment = data_train[payment_columns].copy()
    df_payment["daifu_datediff_nextdate_1"] = 0

    # 使用前缀匹配来提取调用特征
    invocation_columns = [col for col in data_train.columns if invocation_prefix in col]
    df_invocation = data_train[invocation_columns].copy()
    df_invocation["diaoyong_datediff_nextdate_1"] = 0

    # 输出结果，检查每个特征维度是否提取成功
    print("代扣特征:")
    print(df_deduction.head())
    print("代付特征:")
    print(df_payment.head())
    print("调用特征:")
    print(df_invocation.head())
    diaoyong_features = preprocess_data(
        df_invocation,
        continue_col=[
            "datediff_loandate",
            "member_times",
            "datediff_member",
            "datediff_nextdate"
        ],
        categorical_col=[])
    daikou_features = preprocess_data(
        df_deduction,
        continue_col=[
            "order_money",
            "datediff_loandate",
            "member_times",
            "flag_times",
            "memberflag_times",
            "datediff_member",
            "datediff_flag",
            "datediff_memberflag",
            "datediff_nextdate"
        ],
        categorical_col=["succ_flag", "member_name"])
    daifu_features = preprocess_data(
        df_payment,
        continue_col=[
            "transfer_money",
            "datediff_loandate",
            "member_times",
            "datediff_member",
            "datediff_nextdate"
        ],
        categorical_col=["member_name"])
    
load_data()