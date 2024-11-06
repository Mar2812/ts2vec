from ts2vec import TS2Vec
import datautils
import torch
from tasks.classification import eval_classification

# Load the ECG200 dataset from UCR archive
train_data, train_labels, test_data, test_labels = datautils.load_UCR('ECG200')
# (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)
# 将 train_data 和 test_data 转换为 tensor，并扩展最后的特征维度
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)

# 使用 repeat 扩展最后一维特征的数量，从 1 扩展到 10
train_data = train_data.repeat(1, 1, 5).numpy()
test_data = test_data.repeat(1, 1, 5).numpy()

# Train a TS2Vec model
model = TS2Vec(
    input_dims=train_data.shape[-1],
    device=torch.device('cpu'),
    output_dims=320
)
loss_log = model.fit(
    train_data,
    verbose=True
)

# Compute timestamp-level representations for test set
test_repr = model.encode(test_data)  # n_instances x n_timestamps x output_dims

# 最终的特征没有了时间维度，提取的是每个实例的特征，适合画像特征构建
# Compute instance-level representations for test set
test_repr = model.encode(test_data, encoding_window='full_series')  # n_instances x output_dims

# Sliding inference for test set
test_repr = model.encode(
    test_data,
    causal=True,
    sliding_length=1,
    sliding_padding=50
)  # n_instances x n_timestamps x output_dims
# (The timestamp t's representation vector is computed using the observations located in [t-50, t])

out, eval_res = eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear')
print(out, eval_res)