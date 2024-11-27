import torch
from torch import nn
import torch.nn.functional as F

def supervised_hierarchical_contrastive_loss(z1, z2, labels, alpha=0.5, temporal_unit=0):
    """
    有监督的层次对比损失
    Args:
        z1, z2: 特征表示张量，形状为 [B, T, C]
        labels: 样本标签，形状为 [B]
        alpha: 实例对比损失与时间对比损失的权重
        temporal_unit: 从哪个层次开始计算时间对比损失
    Returns:
        loss: 总的对比损失
    """
    device = z1.device
    loss = torch.tensor(0., device=device)
    d = 0  # 层数计数器

    # 生成标签掩码，相同标签为1，不同标签为0
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # 逐层计算
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * supervised_instance_contrastive_loss(z1, z2, mask)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        # 时间池化，减少时间维度
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * supervised_instance_contrastive_loss(z1, z2, mask)
        d += 1
    return loss / d

# 有监督实例对比损失
def supervised_instance_contrastive_loss(z1, z2, mask):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = sim / 0.07  # 使用温度参数 τ=0.07
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]  # 数值稳定性

    # 构造正负样本掩码
    mask = mask.repeat(2, 2)  # 扩展掩码到 [2B, 2B]
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask

    # 计算对比损失
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=-1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)

    # 返回平均损失
    return -mean_log_prob_pos.mean()

# 有监督时间对比损失
def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
