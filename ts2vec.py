import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.losses import supervised_hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math
import logging  # 添加日志库
import os
from datetime import datetime
from torch import nn
from losses import SupConLoss

class TS2Vec:
    '''The TS2Vec model'''

    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        nums_cls=2,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.fc = nn.Sequential(
            nn.Linear(output_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dims, nums_cls)
        ).to(self.device)
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
        
        # 设置logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)  # 设置日志等级为INFO
    
    def compute_model_params(self):
        '''计算并记录模型的总参数量'''
        total_params = sum(p.numel() for p in self._net.parameters() if p.requires_grad)
        self.logger.info(f"模型的总参数量为: {total_params}")
        return total_params

    def fit(self, train_data, train_labels, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model with supervised contrastive learning.

        Args:
            train_data (numpy.ndarray): The training data. Shape: (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            train_labels (numpy.ndarray): Ground truth labels for supervised learning. Shape: (n_instance,).
            n_epochs (Union[int, NoneType]): Number of epochs. Training stops after this if specified.
            n_iters (Union[int, NoneType]): Number of iterations. Training stops after this if specified. If neither is specified, defaults are used.
            verbose (bool): Whether to print training loss after each epoch.

        Returns:
            loss_log: A list containing the training losses for each epoch.
        '''
        assert train_data.ndim == 3

        os.makedirs("results/loss", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        loss_file_path = f"results/loss/{timestamp}_loss_result.txt"

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), torch.from_numpy(train_labels).to(torch.long))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(
            list(self._net.parameters()) + list(self.fc.parameters()),
            lr=self.lr
        )
        
        iters_per_epoch = train_data.shape[0]//train_loader.batch_size + 1
        T_max = iters_per_epoch * n_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=1e-7
        )

        loss_log = []

        self.fc.train()
        self._net.train()
        
        with open(loss_file_path, "w") as f:
            while True:
                if n_epochs is not None and self.n_epochs >= n_epochs:
                    break

                cum_loss = 0
                n_epoch_iters = 0

                interrupted = False
                for batch in train_loader:
                    if n_iters is not None and self.n_iters >= n_iters:
                        interrupted = True
                        break

                    x = batch[0]
                    y = batch[1]
                    if self.max_train_length is not None and x.size(1) > self.max_train_length:
                        window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                        x = x[:, window_offset: window_offset + self.max_train_length]
                    x = x.to(self.device)
                    y = y.to(self.device)

                    optimizer.zero_grad()

                    # 创建两种增强方式
                    ts_l = x.size(1)
                    crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                    crop_left = np.random.randint(ts_l - crop_l + 1)
                    crop_right = crop_left + crop_l

                    crop_eleft = np.random.randint(crop_left + 1)
                    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

                    # z1 和 z2 分别是两种增强方式下的特征表示
                    z1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                    z1 = z1[:, -crop_l:]

                    z2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                    z2 = z2[:, :crop_l]
                    
                    z1_pool = F.max_pool1d(
                        z1.transpose(1, 2),
                        kernel_size = z1.size(1),
                    ).transpose(1, 2)
                    y1_score = self.fc(z1_pool.squeeze(1))
                    
                    z2_pool = F.max_pool1d(
                        z2.transpose(1, 2),
                        kernel_size = z2.size(1),
                    ).transpose(1, 2)
                    y2_score = self.fc(z2_pool.squeeze(1))
                    
                    # 有监督对比学习损失
                    # z = torch.cat([z1_pool.unsqueeze(1), z2_pool.unsqueeze(1)], dim=1)  # [batch_size, 2, feature_dim]
                    supervised_contrastive_loss = supervised_hierarchical_contrastive_loss(z1, z2, labels=y)
                    x_pool = F.max_pool1d(
                        self._net(x).transpose(1, 2),
                        kernel_size = x.size(1),
                    ).transpose(1, 2)
                    y_score_all = self.fc(x_pool.squeeze(1))
                    classification_loss_all = self.cross_entropy_loss_fn(y_score_all, y)
                    classification_loss1 = self.cross_entropy_loss_fn(y1_score, y)
                    classification_loss2 = self.cross_entropy_loss_fn(y2_score, y)
                    classification_loss = (classification_loss1 + classification_loss2 + classification_loss_all) / 3.0
                    # 综合损失（对比学习损失 + 分类损失）
                    lambda_contrastive = 0.4
                    lambda_classification = 0.6
                    loss = lambda_contrastive * supervised_contrastive_loss + lambda_classification * classification_loss
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.net.update_parameters(self._net)

                    cum_loss += loss.item()
                    n_epoch_iters += 1

                    self.n_iters += 1

                    if self.after_iter_callback is not None:
                        self.after_iter_callback(self, loss.item())

                if interrupted:
                    break

                cum_loss /= n_epoch_iters
                loss_log.append(cum_loss)

                f.write(f"Epoch #{self.n_epochs}: loss={cum_loss}\n")

                if verbose:
                    self.logger.info(f"Epoch #{self.n_epochs}: loss={cum_loss}")
                self.n_epochs += 1

                if self.after_epoch_callback is not None:
                    self.after_epoch_callback(self, cum_loss)

        return loss_log

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None, return_cls=False):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        self.fc.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                        if return_cls:
                            out = self.fc(out.to(self.device, non_blocking=True)).cpu()
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        if return_cls:
                            self.fc.eval()
                            out = self.fc(out.to(self.device, non_blocking=True)).cpu()
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
