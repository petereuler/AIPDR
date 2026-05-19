import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_tokens, kv_tokens, attn_mask=None):
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        out, _ = self.attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        x = q_tokens + self.dropout(out)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class EncoderTransformerReasoner(nn.Module):
    def __init__(self, pose_dim, dist_dim, d_model=192, nhead=6, num_layers=4, dropout=0.1):
        super().__init__()
        self.pose_q = nn.Linear(pose_dim, d_model)
        self.dist_q = nn.Linear(dist_dim, d_model)
        self.pose_kv = nn.Linear(pose_dim, d_model)
        self.dist_kv = nn.Linear(dist_dim, d_model)
        self.type_pose = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.type_dist = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([CausalCrossAttentionBlock(d_model, nhead=nhead, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pose_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 4),
        )
        self.dp_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 3),
        )
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)
        nn.init.zeros_(self.dp_head[-1].weight)
        nn.init.zeros_(self.dp_head[-1].bias)

    def build_query_tokens(self, pose_feat, dist_feat):
        pose_tok = self.pose_q(pose_feat) + self.type_pose
        dist_tok = self.dist_q(dist_feat) + self.type_dist
        return pose_tok + dist_tok

    def build_memory_tokens(self, pose_feat, dist_feat):
        pose_tok = self.pose_kv(pose_feat) + self.type_pose
        dist_tok = self.dist_kv(dist_feat) + self.type_dist
        return pose_tok + dist_tok

    def forward(self, pose_feat, dist_feat, base_q_rel=None, base_dp_body=None):
        if pose_feat.ndim == 2:
            pose_feat = pose_feat.unsqueeze(0)
            dist_feat = dist_feat.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        q_tokens = self.build_query_tokens(pose_feat, dist_feat)
        kv_tokens = self.build_memory_tokens(pose_feat, dist_feat)
        seq_len = q_tokens.size(1)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=q_tokens.device)
        mask = torch.triu(mask, diagonal=1)
        x = q_tokens
        for block in self.blocks:
            x = block(x, kv_tokens, attn_mask=mask)
        x = self.norm(x)
        delta_q = self.pose_head(x)
        delta_dp = self.dp_head(x)
        if base_q_rel is not None:
            if base_q_rel.ndim == 2:
                base_q_rel = base_q_rel.unsqueeze(0)
            q_rel = F.normalize(base_q_rel + delta_q, dim=-1, eps=1e-8)
        else:
            q_rel = F.normalize(delta_q, dim=-1, eps=1e-8)
        if base_dp_body is not None:
            if base_dp_body.ndim == 2:
                base_dp_body = base_dp_body.unsqueeze(0)
            dp_body = base_dp_body + delta_dp
        else:
            dp_body = delta_dp
        if squeeze:
            return q_rel[0], dp_body[0]
        return q_rel, dp_body
