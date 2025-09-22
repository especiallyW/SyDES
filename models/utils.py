from models import *


class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)


class MLPLayer(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, middle_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(middle_dim, output_dim)
        )

    def forward(self, input):
        return self.mlp(input)


# GateKV module (per-head gating)
class GateKV(nn.Module):
    def __init__(self, n_heads: int, dec_dim: int, init_bias: float = -3.0):
        super().__init__()
        self.n_heads = n_heads
        self.dec_dim = dec_dim
        # per-head bias
        self.gate_bias = nn.Parameter(torch.ones(n_heads) * init_bias)  # sigmoid(-3) ~ 0.047

    def forward(self, img_kv, text_kv):
        # img_kv, text_kv: [BV, L, D]
        BV, L, D = img_kv.shape
        assert D == self.dec_dim
        # reshape into heads
        head_dim = D // self.n_heads
        img_h = img_kv.view(BV, L, self.n_heads, head_dim)
        txt_h = text_kv.view(BV, L, self.n_heads, head_dim)
        alpha = torch.sigmoid(self.gate_bias).view(1, 1, self.n_heads, 1).to(img_kv.device)  # [1,1,H,1]
        combined = alpha * txt_h + (1.0 - alpha) * img_h
        return combined.view(BV, L, D), alpha.squeeze()  # return combined and alpha per head


class CombinedAggregator(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_views=4):
        super().__init__()
        # learnable weights for each view
        self.view_weights = nn.Parameter(torch.ones(num_views))

        # lightweight MLP: (D) -> hidden -> (D)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        # LayerNorm before any attention
        self.norm = nn.LayerNorm(feature_dim)

        # projection layers for tiny self-attention
        self.attn_q = nn.Linear(feature_dim, feature_dim)
        self.attn_k = nn.Linear(feature_dim, feature_dim)
        self.attn_v = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** -0.5

    def forward(self, views):
        """
        views: tensor of shape (B, V, D)
        returns: tensor of shape (B, D)
        """
        # ---- 1) Weighted sum pooling ----
        weights = F.softmax(self.view_weights, dim=0)  # (V,)
        weighted = (views * weights.view(1, -1, 1)).sum(1)  # (B, D)

        # ---- 2) MLP projection ----
        mlp_out = self.mlp(weighted)  # (B, D)

        # ---- 3) Tiny self-attention pooling ----
        # Normalize before attention to bound values
        mlp_norm = self.norm(mlp_out)
        # compute Q, K, V from mlp output expanded across views dimension
        # here we treat mlp_out as a single “query” to attend to itself for simplicity:
        q = self.attn_q(mlp_norm).unsqueeze(1)  # (B, 1, D)
        k = self.attn_k(self.norm(views))  # (B, V, D)
        v = self.attn_v(self.norm(views))  # (B, V, D)
        # compute attention scores (B,1,V) then normalize
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)  # (B,1,V)
        attn_out = torch.matmul(attn, v).squeeze(1)  # (B, D)

        # final aggregated feature
        return attn_out
