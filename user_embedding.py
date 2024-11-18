import logging
from ts2vec import TS2Vec
import datautils
import torch
from tasks.classification import eval_classification
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re
from sklearn.utils import resample
import time
from datetime import datetime

# # # 参数计算
# model = TS2Vec(input_dims=1, output_dims=320, hidden_dims=64, depth=5, device=torch.device('cpu'))
# print(model.compute_model_params())
# exit()

# 设置 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设为DEBUG级别，以捕获详细信息

# 配置 StreamHandler 用于终端输出
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# 确保目录存在
log_dir = "results/terminal_info"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + "_loginfo.txt")
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

def preprocess_data(df, continue_col, categorical_col):
    # 分离连续特征和离散特征
    continuous_cols = [col for col in df.columns if any(tag in col for tag in continue_col)]
    categorical_cols = [col for col in df.columns if any(tag in col for tag in categorical_col)]

    # 处理连续特征 - 标准化
    scaler = StandardScaler()
    df[continuous_cols] = df[continuous_cols].astype(float)
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols].fillna(0))
    logger.info("连续特征已标准化")
    
    # 处理离散特征 - One-Hot 编码
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_data = encoder.fit_transform(df[categorical_cols].fillna('unknown'))
    categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
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
    
    logger.info(f"预处理后的 result_array 形状: {result_array.shape}")

    return np.nan_to_num(result_array, nan=0.0)

def load_data(dataset=None):
    if not dataset:
        raise ValueError("未输入数据源路径")
    # 读取数据
    data = pd.read_csv(dataset)#[:5000]

    # 平衡样本
    def balanced(data):
        # 分离标签为1和标签为0的样本
        data_minority = data[data['target'] == 1]
        data_majority = data[data['target'] == 0]

        # 从多数类样本中随机选择与少数类样本数量相同的样本
        data_majority_downsampled = resample(data_majority,
                                            replace=False,    # 不放回采样
                                            n_samples=len(data_minority)*2,
                                            random_state=42)

        # 合并平衡后的数据集
        data_balanced = pd.concat([data_majority_downsampled, data_minority])
        return data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 分割数据集
    data_train = data[data['dataset'] == 'trainSet']
    data_train = balanced(data_train)
    logger.info(f"训练样本分布: {data_train['target'].value_counts()}")
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

time_begin = time.time()
data_path = r"/home/mc/ts2vec/ts2vec/datasets/wfplus_application_1718475236931_781952_timefeattopresult_spark_tfmid.csv"
# data_path = r"C:\Users\chao_ma02\Desktop\work\ts2vec\datasets\wfplus_application_1718475236931_781952_timefeattopresult_spark_tfmid.csv"
(
    diaoyong_features_train, daikou_features_train, daifu_features_train, labels_train,
    diaoyong_features_test, daikou_features_test, daifu_features_test, labels_test,
    diaoyong_features_oot, daikou_features_oot, daifu_features_oot, labels_oot
) = load_data(data_path)
time_end = time.time()
logger.info(f"数据加载时间: {time_end - time_begin}秒")
logger.info(f"训练数据形状: {diaoyong_features_train.shape}")

# 使用TS2Vec模型对daikou, daifu, diaoyong特征分别进行编码
def encode_features(model, train_features, test_features, oot_features, encoding_window='full_series', return_cls=False):
    train_repr = model.encode(train_features,  encoding_window=encoding_window, return_cls=return_cls)
    test_repr = model.encode(test_features, encoding_window=encoding_window, return_cls=return_cls)
    oot_repr = model.encode(oot_features, encoding_window=encoding_window, return_cls=return_cls)
    return train_repr, test_repr, oot_repr

# 获取当前时间，用于生成结果文件名
result_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_metric.txt"
result_filename = f"results/metric/{result_filename}"
directory = os.path.dirname(result_filename)
# 如果目录不存在，则创建目录
if not os.path.exists(directory):
    os.makedirs(directory)

def save_results_to_file(content, filename):
    """将内容写入指定文件"""
    with open(filename, "a") as file:
        file.write(content + "\n")

# 使用TS2Vec模型对daikou, daifu, diaoyong特征分别进行编码并进行独立测试
def encode_and_evaluate(model, train_features, test_features, oot_features, labels_train, labels_test, labels_oot, feature_name):
    # 对训练集、测试集和OOT集进行编码
    train_score, test_score, oot_score = encode_features(model, train_features, test_features, oot_features, return_cls=True)

    # 在训练集上评估
    eval_metrics_train = eval_classification(train_score, labels_train)

    # 在测试集上评估
    eval_metrics_test = eval_classification(test_score, labels_test)
    
    # 在OOT集上评估
    eval_metrics_oot = eval_classification(oot_score, labels_oot)

    # 组织评估结果的文本
    result_text = f"【{feature_name}】特征 - 训练集评估指标:\n" \
                  f"训练集准确率 (Accuracy): {eval_metrics_train['acc']}\n" \
                  f"训练集AUPRC: {eval_metrics_train['auprc']}\n" \
                  f"训练集AUC: {eval_metrics_train['auc']}\n" \
                  f"训练集KS: {eval_metrics_train['ks']}\n" \
                  f"【{feature_name}】特征 - 测试集评估指标:\n" \
                  f"测试集准确率 (Accuracy): {eval_metrics_test['acc']}\n" \
                  f"测试集AUPRC: {eval_metrics_test['auprc']}\n" \
                  f"测试集AUC: {eval_metrics_test['auc']}\n" \
                  f"测试集KS: {eval_metrics_test['ks']}\n" \
                  f"【{feature_name}】特征 - OOT集评估指标:\n" \
                  f"OOT集准确率 (Accuracy): {eval_metrics_oot['acc']}\n" \
                  f"OOT集AUPRC: {eval_metrics_oot['auprc']}\n" \
                  f"OOT集AUC: {eval_metrics_oot['auc']}\n" \
                  f"OOT集KS: {eval_metrics_oot['ks']}\n"

    # 打印并保存结果
    logger.info(result_text)
    save_results_to_file(result_text, result_filename)

# 设置TS2Vec模型的公共参数
device = 0
# device = torch.device('cpu')
output_dims = 320
batch_size = 1024
lr = 0.001
depth = 5
epochs = 30


try:
    # # daikou特征编码和评估
    # model_daikou = TS2Vec(lr=lr, depth=depth, input_dims=daikou_features_train.shape[-1], device=device, output_dims=output_dims, batch_size=batch_size)
    # model_daikou.fit(daikou_features_train, verbose=True, n_epochs=epochs)
    # encode_and_evaluate(model_daikou, daikou_features_train, daikou_features_test, daikou_features_oot, labels_train, labels_test, labels_oot, "daikou")

    # # daifu特征编码和评估
    # model_daifu = TS2Vec(lr=lr, depth=depth, input_dims=daifu_features_train.shape[-1], device=device, output_dims=output_dims, batch_size=batch_size)
    # model_daifu.fit(daifu_features_train, verbose=True, n_epochs=epochs)
    # encode_and_evaluate(model_daifu, daifu_features_train, daifu_features_test, daifu_features_oot, labels_train, labels_test, labels_oot, "daifu")

    # diaoyong特征编码和评估
    model_diaoyong = TS2Vec(lr=lr, depth=depth, input_dims=diaoyong_features_train.shape[-1], device=device, output_dims=output_dims, batch_size=batch_size)
    model_diaoyong.fit(diaoyong_features_train, train_labels=labels_train, verbose=True, n_epochs=epochs)
    encode_and_evaluate(model_diaoyong, diaoyong_features_train, diaoyong_features_test, diaoyong_features_oot, labels_train, labels_test, labels_oot, "diaoyong")
except Exception as e:
    logger.exception(str(e))