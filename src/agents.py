import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer
import torch.nn.functional as F

print("PyTorch 版本:", torch.__version__)
print("編譯時使用的 CUDA 版本:", torch.version.cuda)
print("目前是否可用 CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("當前 GPU:", torch.cuda.get_device_name(0))
    print("當前 GPU 支援的 CUDA 計算能力:", torch.cuda.get_device_capability(0))
# ──────────────────────────────────────────────────────────────
# Backbone Models
# ──────────────────────────────────────────────────────────────

class MLPBackbone(nn.Module):
    """
    MLP backbone：將 (batch, lookback, n_assets) flatten 後通過多層 MLP
    Output: (batch, hidden_dim // 4)
    """
    def __init__(self, n_assets: int, lookback_days: int = 60,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        self.n_assets = n_assets
        self.lookback_days = lookback_days
        input_dim = lookback_days * n_assets
        self.output_dim = hidden_dim // 4
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, self.output_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_assets)
        batch_size = x.shape[0]
        return self.network(x.view(batch_size, -1))  # (batch, output_dim)


class LSTMBackbone(nn.Module):
    """
    LSTM backbone：輸入 (batch, lookback, n_assets) 的時間序列
    Output: (batch, hidden_dim)，取最後一個時間步的 hidden state
    """
    def __init__(self, n_assets: int, lookback_days: int = 60,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.output_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size  = n_assets,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_assets)
        out, _ = self.lstm(x)       # (batch, lookback, hidden_dim)
        out    = out[:, -1, :]      # (batch, hidden_dim) 取最後一步
        return self.dropout(out)


class TransformerBackbone(nn.Module):
    """
    Transformer backbone：輸入 (batch, lookback, n_assets) 的時間序列
    Output: (batch, hidden_dim)，取最後一個時間步的輸出
    """
    def __init__(self, n_assets: int, lookback_days: int = 60,
                 hidden_dim: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        assert hidden_dim % nhead == 0, \
            f"hidden_dim ({hidden_dim}) 必須能被 nhead ({nhead}) 整除"

        self.output_dim  = hidden_dim
        self.input_proj  = nn.Linear(n_assets, hidden_dim)
        self.pos_enc     = PositionalEncoding(hidden_dim, max_len=lookback_days)  # 預留一些額外位置

        encoder_layer    = nn.TransformerEncoderLayer(
            d_model         = hidden_dim,
            nhead           = nhead,
            dim_feedforward = hidden_dim * 2,
            dropout         = dropout,
            batch_first     = True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_assets)
        out = self.input_proj(x)    # (batch, lookback, hidden_dim)
        out = self.pos_enc(out)     # (batch, lookback, hidden_dim)
        out = self.encoder(out)     # (batch, lookback, hidden_dim)
        out = out[:, -1, :]         # (batch, hidden_dim) 取最後一步
        return self.dropout(out)

    # def forward(self, x):
    #     seq_len = x.size(1)
    #     out = self.input_proj(x)
    #     out = self.pos_enc(out)
    #     mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
    #     out = self.encoder(out, mask=mask)
    #     out = out[:, -1, :]
    #     return self.dropout(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(1), :]
# ──────────────────────────────────────────────────────────────
# Factory Function
# ──────────────────────────────────────────────────────────────

def build_backbone(model_type: str, n_assets: int,
                   lookback_days: int) -> nn.Module:
    """
    根據 model_type 建立對應的 backbone

    Args:
        model_type:   'mlp' | 'lstm' | 'transformer'
        n_assets:     資產數量
        lookback_days: lookback 天數
        config:       包含各 backbone 超參數的字典

    Returns:
        backbone: nn.Module，具有 .output_dim 屬性
    """
    if model_type == 'mlp':
        return MLPBackbone(
            n_assets      = n_assets,
            lookback_days = lookback_days
        )
    elif model_type == 'lstm':
        return LSTMBackbone(
            n_assets      = n_assets,
            lookback_days = lookback_days
        )
    elif model_type == 'transformer':
        return TransformerBackbone(
            n_assets      = n_assets,
            lookback_days = lookback_days
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Choose from ['mlp', 'lstm', 'transformer']"
        )

class ReturnPredictionModel(nn.Module):
    """
    A unified reward prediction model:

    The backbone is responsible for feature extraction, and the head is responsible for outputting n_assets of predicted rewards.

    Input:  (batch, lookback_days, n_assets)
    Output: (batch, n_assets)
    """
    def __init__(self, n_assets: int, lookback_days: int,
                 model_type: str = 'mlp'):
        super().__init__()
        self.backbone = build_backbone(
            model_type    = model_type,
            n_assets      = n_assets,
            lookback_days = lookback_days
        )

        # head：把 backbone 的輸出映射到 n_assets
        self.head = nn.Linear(self.backbone.output_dim, n_assets)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):
                # Xavier 適合 ReLU/LeakyReLU 前後的 Linear
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        # input-hidden weight：Kaiming
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        # hidden-hidden weight：Orthogonal
                        # 正交初始化讓 hidden state 的信息流更穩定 
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        # LSTM forget gate bias 設為 1
                        # 讓模型初始時傾向於記住信息而不是遺忘
                        n = param.shape[0]
                        param.data[n // 4 : n // 2].fill_(1.0)

            elif isinstance(m, nn.TransformerEncoderLayer):
                # Transformer 的 self-attention 和 feedforward 層
                for name, param in m.named_parameters():
                    if 'weight' in name and param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        # print(f"backbone output mean: {feat.abs().mean():.6f}, std: {feat.std():.6f}")

        mu_hat = self.head(feat)    # (batch, n_assets)
        # print(f"head output mean: {mu_hat.abs().mean():.6f}, std: {mu_hat.std():.6f}")

        return mu_hat


class MVOOptimizationLayer(nn.Module):
    """
    Mean-Variance Optimization layer that solves:
    w* = argmax_w (w^T * mu - lambda * w^T * Sigma * w)
    s.t. sum(w) = 1, 0 <= w_i <= 1/k
    
    Uses CVXPY for optimization with gradient computation capability.
    """
    def __init__(self, n_assets: int, lambda_risk: float = 1.0,
                 max_weight: float = 1.0, device: str = 'cpu',
                 transaction_cost: float = 0.002,
                 group_indices: Optional[List[int]] = None,
                 group_min_weight: Optional[float] = None):
        """
        group_indices    : 要設下限的標的索引（對應 n_assets 的欄位順序）。
        group_min_weight : group_indices 這幾檔「合計權重」的硬性下限，直接寫進
                            MVO 的限制式，由優化器依當天 mu/Sigma 自行決定組內怎麼
                            分配（而不是像 w_sum 那樣事後平分剩餘預算）。
                            None 時不加這條限制（預設，等同原本做法）。
        """
        super(MVOOptimizationLayer, self).__init__()
        self.n_assets = n_assets
        self.lambda_risk = lambda_risk
        self.max_weight = max_weight  # 1/k constraint
        self.device = device
        self.transaction_cost = transaction_cost
        self.group_indices = group_indices
        self.group_min_weight = group_min_weight

        mu_param = cp.Parameter(n_assets)
        # sigma_param = cp.Parameter((n_assets, n_assets), PSD=True)
        L_param = cp.Parameter((n_assets, n_assets))
        w_prev_param = cp.Parameter(n_assets)

        w = cp.Variable(n_assets)

        # risk = cp.quad_form(w, L_param)
        risk = cp.sum_squares(L_param.T @ w)
        returns = mu_param @ w
        turnover = cp.norm1(w - w_prev_param)
        # objective = cp.Minimize(self.lambda_risk * risk - returns)
        objective = cp.Maximize(returns - self.lambda_risk * risk - self.transaction_cost * turnover)

        constraints = [
            cp.sum(w) == 1.0, 
            w >= 0,              # long-only
            w <= self.max_weight # 1/k constraint
        ]
        if group_indices is not None and group_min_weight is not None:
            constraints.append(cp.sum(w[group_indices]) >= group_min_weight)
        problem = cp.Problem(objective, constraints)

        print(f"Is DCP? {problem.is_dcp()}") # 必須是 True
        print(f"Is DPP? {problem.is_dpp()}") # 必須是 True
        print(f"DPP Validation: {problem.is_dpp()}") 

        self.cvx_layer = CvxpyLayer(
            cp.Problem(objective, constraints), 
            parameters=[mu_param, L_param, w_prev_param], 
            variables=[w]
        )

    def solve(self, mu_hat: torch.Tensor, sigma: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
        """
        Solve MVO problem for a batch of predictions.
        
        Args:
            mu_hat: (batch_size, n_assets) - predicted returns
            sigma: (batch_size, n_assets, n_assets) - covariance matrices
            w_prev: (batch_size, n_assets) - previous portfolio weights
        Returns:
            w_star: (batch_size, n_assets) - optimal portfolio weights
        """
        # mu_hat = mu_hat.detach().cpu().numpy()
        # sigma_np = sigma.detach().cpu().numpy()

        mu_hat = mu_hat.to(self.device)
        sigma = sigma.to(self.device)
        w_prev = w_prev.to(self.device)
        eps = 1e-6 * torch.eye(self.n_assets, device=self.device)
        # L = torch.linalg.cholesky(sigma + eps)
        # sigma_stable = sigma + eps

        # try:

        #     w_star, = self.cvx_layer(mu_hat, L)
        #     return w_star
            
        # except Exception as e:

        #     print(f"Warning: Optimization failed ({e}), fallback to equal weights.")
        #     return torch.ones_like(mu_hat) / self.n_assets

        try:
            L = torch.linalg.cholesky(sigma + eps)
        
        # 3. 核心修正：
        # 如果 self.device 是 'cuda'，有些 solver 會報錯。
        # 建議在這裡確保 cvx_layer 接收的是正確類型的數據。
            w_star, = self.cvx_layer(mu_hat, L, w_prev)
        
            return w_star.to(dtype=torch.float32, device=self.device)
        
        except Exception as e:
        # 這裡會捕捉到你看到的那個 "can't convert cuda:0..." 錯誤
            if "cuda" in str(e):
            # 強制將數據搬回 CPU 計算 (這是最後的保險方案)
                mu_cpu = mu_hat.cpu()
                w_prev_cpu = w_prev.cpu()
                L_cpu = torch.linalg.cholesky((sigma + eps).cpu())
                w_star_cpu, = self.cvx_layer(mu_cpu, L_cpu, w_prev_cpu)
                return w_star_cpu.to(dtype=torch.float32, device=self.device)
            
            print(f"Warning: Optimization failed ({e}), fallback to equal weights.")
            return torch.ones_like(mu_hat) / self.n_assets
    
    def compute_utility(self, w: torch.Tensor, mu: torch.Tensor, 
                       sigma: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute portfolio utility: U = w^T*mu - lambda*w^T*Sigma*w
        
        Args:
            w: (batch_size, n_assets) - portfolio weights
            mu: (batch_size, n_assets) - expected returns
            sigma: (batch_size, n_assets, n_assets) - covariance matrices
            w_prev: (batch_size, n_assets) - previous portfolio weights
        Returns:
            utility: (batch_size,) - portfolio utilities
        """
        # Returns: w^T * mu
        returns = torch.sum(w * mu, dim=1)

        # Risk: w^T * Sigma * w
        # For batch computation: (batch, 1, n) @ (batch, n, n) @ (batch, n, 1)
        w_expanded = w.unsqueeze(1) # (batch, 1, n_assets)
        risk = torch.bmm(torch.bmm(w_expanded, sigma), w_expanded.transpose(1, 2))
        risk = risk.squeeze()  # (batch,)
        
        tc = self.transaction_cost * torch.abs(w - w_prev).sum(dim=1)
        utility = returns - self.lambda_risk * risk - tc
        return utility


class CombinedLoss(nn.Module):
    """
    Combined loss: L = alpha * L_MVO + (1-alpha) * L_MSE
    
    L_MVO = Regret = U(w*(mu_hat), mu*) - U(w*(mu*), mu*)
    L_MSE = mean((mu* - mu_hat)^2)
    """
    def __init__(self, alpha: float = 0.5, mse_scale: float = 1000.0, 
                 transaction_cost: float = 0.003):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_scale = mse_scale  # Scale MSE to match MVO loss magnitude
        self.mse_loss = nn.MSELoss()
        self.transaction_cost = transaction_cost
    
    def forward(self, mu_hat: torch.Tensor, mu_star: torch.Tensor,
                w_hat: torch.Tensor, w_star: torch.Tensor,
                sigma: torch.Tensor, lambda_risk: float, 
                w_prev_hat: torch.Tensor, w_prev_star: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            mu_hat: (batch_size, n_assets) - predicted returns
            mu_star: (batch_size, n_assets) - ground truth returns
            w_hat: (batch_size, n_assets) - portfolio weights from mu_hat
            w_star: (batch_size, n_assets) - optimal portfolio weights from mu_star
            sigma: (batch_size, n_assets, n_assets) - covariance matrices
            lambda_risk: risk aversion parameter
            w_prev: (batch_size, n_assets) - previous portfolio weights (for transaction cost)
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary with loss components
        """

        # MSE Loss
        mse_loss = self.mse_loss(mu_hat, mu_star) * self.mse_scale

        # MVO Loss (Regret)
        # U(w_hat, mu_star)
        returns_hat = torch.sum(w_hat * mu_star, dim=1)
        w_hat_expanded = w_hat.unsqueeze(1)
        risk_hat = torch.bmm(torch.bmm(w_hat_expanded, sigma), 
                            w_hat_expanded.transpose(1, 2)).squeeze(-1).squeeze(-1)
        tc_hat = self.transaction_cost * torch.abs(w_hat - w_prev_hat).sum(dim=1)
        utility_hat = returns_hat - lambda_risk * risk_hat - tc_hat

        # U(w_star, mu_star)
        
        returns_star = torch.sum(w_star * mu_star, dim=1)
        w_star_expanded = w_star.unsqueeze(1)
        risk_star = torch.bmm(torch.bmm(w_star_expanded, sigma), 
                             w_star_expanded.transpose(1, 2)).squeeze(-1).squeeze(-1)
        tc_star = self.transaction_cost * torch.abs(w_star - w_prev_star).sum(dim=1)    
        utility_star = returns_star - lambda_risk * risk_star - tc_star
        
        # if torch.equal(w_hat, w_star):
        #     print("weight相同")

        # Regret = U(w_star, mu_star) - U(w_hat, mu_star)

        # We want to minimize this, so loss = -utility_hat + utility_star
        # print(torch.equal(utility_hat, utility_star))

        regret = torch.clamp(utility_star - utility_hat, min=0.0)
        # regret = F.softplus(utility_star - utility_hat, beta=10)

        
        # print(f"regret : {regret}")
        mvo_loss = torch.mean(regret)

        # policy gradient loss (without regret)
        # mvo_loss = -torch.mean(utility_hat)

        # Combined loss
        total_loss = self.alpha * mvo_loss + (1 - self.alpha) * mse_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'mvo_loss': mvo_loss.item(),
            'mse_loss': mse_loss.item() / self.mse_scale,  # Unscaled for reporting
            'regret': mvo_loss.item()
        }
        
        return total_loss, loss_dict


class DFLMVOModel(nn.Module):
    """
    Complete DFL-MVO model combining prediction and optimization.
    """
    def __init__(self, n_assets: int, lookback_days: int = 60,
                 lambda_risk: float = 1.0, max_weight: float = 0.2,
                 alpha: float = 0.75, device: str = 'cpu',
                 transaction_cost: float = 0.002,
                 model_type: str = 'mlp',
                 backbone_config: dict = None,
                 group_indices: Optional[List[int]] = None,
                 group_min_weight: Optional[float] = None):
        """
        group_indices    : 要設下限的標的索引（對應 n_assets 的欄位順序），w_sum 機制與
                            group_min_weight 機制共用同一份索引。
        group_min_weight : group_indices 這幾檔「合計權重」的硬性下限，直接寫進 MVO 的
                            限制式，由優化器依當天 mu/Sigma 自行決定組內怎麼分配。
                            跟 w_sum<1 的平分做法不同、也可以獨立使用（w_sum 保持 1.0）。
        """

        super(DFLMVOModel, self).__init__()

        self.n_assets = n_assets
        self.lookback_days = lookback_days
        self.lambda_risk = lambda_risk
        self.max_weight = max_weight
        self.device = device
        self.transaction_cost = transaction_cost
        self.model_type = model_type
        self.group_indices = group_indices
        self.group_min_weight = group_min_weight

        # Prediction model
        self.prediction_model = ReturnPredictionModel(
            n_assets=n_assets,
            lookback_days=lookback_days,
            model_type=model_type
        ).to(device)

        # Optimization layer
        self.optimizer_layer = MVOOptimizationLayer(
            n_assets=n_assets,
            lambda_risk=lambda_risk,
            max_weight=max_weight,
            device=device,
            transaction_cost=transaction_cost,
            group_indices=group_indices,
            group_min_weight=group_min_weight,
        )

        # Loss function
        self.loss_fn = CombinedLoss(alpha=alpha, transaction_cost=transaction_cost).to(device)
        self.register_buffer(
            'w_prev_hat',
            torch.zeros(1, n_assets, device=device) #initialization zero weight
        )
        self.register_buffer(
            'w_prev_star',
            torch.zeros(1, n_assets, device=device) #initialization zero weight
        )
    
    def reset_w_prev(self):
        """每個 epoch 開始時重置，避免跨 epoch 污染"""
        self.w_prev_hat  = torch.zeros(1, self.n_assets, device=self.device)
        self.w_prev_star = torch.zeros(1, self.n_assets, device=self.device)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor,
                mu_star: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass through the model.
        
        Args:
            x: (batch_size, lookback_days, n_assets) - historical returns
            sigma: (batch_size, n_assets, n_assets) - covariance matrices
            mu_star: (batch_size, n_assets) - ground truth returns (for training)
        Returns:
            output_dict: dictionary containing predictions and weights
        """
        batch_size = x.shape[0]
        # Predict expected returns
        
        # Solve optimization problem
        w_prev_hat_raw  = self.w_prev_hat.expand(batch_size, -1)
        w_prev_star_raw = self.w_prev_star.expand(batch_size, -1)

        mu_hat = self.prediction_model(x)
        # DFLMVOModel.forward 裡


        w_hat = self.optimizer_layer.solve(mu_hat, sigma, w_prev_hat_raw)
        self.w_prev_hat = w_hat[-1:].detach()

        output_dict = {
            'mu_hat': mu_hat,
            'w_hat': w_hat,
        }

        # Compute loss if ground truth is provided
        if mu_star is not None:
            with torch.no_grad():
                w_star = self.optimizer_layer.solve(mu_star.detach(), sigma.detach(), w_prev_star_raw)
            self.w_prev_star = w_star[-1:].detach()

            loss, loss_dict = self.loss_fn(
                mu_hat, mu_star, w_hat, w_star,
                sigma, self.lambda_risk, w_prev_hat_raw, w_prev_star_raw
            )
            output_dict['loss'] = loss
            output_dict['loss_dict'] = loss_dict
            output_dict['w_star'] = w_star

        return output_dict