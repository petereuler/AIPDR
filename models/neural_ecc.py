import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gray_encode_int(x):
    return x ^ (x >> 1)


def int_to_bits(x, num_bits):
    bits = []
    for i in range(num_bits - 1, -1, -1):
        bits.append((x >> i) & 1)
    return np.array(bits, dtype=np.int32)


def bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hadamard_matrix(order):
    """
    Recursive Hadamard matrix construction (order must be power of 2).
    Returns +/-1 matrix of shape (order, order).
    """
    if order == 1:
        return torch.tensor([[1.0]])
    h = hadamard_matrix(order // 2)
    top = torch.cat([h, h], dim=1)
    bottom = torch.cat([h, -h], dim=1)
    return torch.cat([top, bottom], dim=0)


class HeadingQuantizer:
    """
    Angle -> 12-bit Gray -> 4x Hadamard(8,3) -> 32-bit codeword.
    """
    def __init__(self, num_bits=12):
        if num_bits != 12:
            raise ValueError("HeadingQuantizer expects num_bits=12.")
        self.num_bits = num_bits
        self.num_bins = 2 ** num_bits
        self.block_bits = 3
        self.block_code_bits = 8
        self.num_blocks = self.num_bits // self.block_bits
        self.code_bits = self.num_blocks * self.block_code_bits
        self.angle_range = 2 * np.pi
        self.bin_width = self.angle_range / self.num_bins
        self.fitted = True

        self.H8 = hadamard_matrix(8)  # (8,8) bipolar

        # Build full codebook (4096, 32) in bipolar {-1,+1}
        codebook_bits = []
        for q in range(self.num_bins):
            g = gray_encode_int(q)
            g_bits = int_to_bits(g, num_bits)
            blocks = [g_bits[i:i+self.block_bits] for i in range(0, num_bits, self.block_bits)]
            block_words = []
            for b in blocks:
                idx = bits_to_int(b)
                cw = self.H8[idx].numpy()
                cw_bits = ((1.0 - cw) / 2.0).astype(np.int32)
                block_words.append(cw_bits)
            codebook_bits.append(np.concatenate(block_words, axis=0))
        codebook_bits = np.stack(codebook_bits, axis=0).astype(np.float32)
        self.all_codewords = torch.from_numpy(codebook_bits)  # (4096, 32) in {0,1}
        # Map to bipolar for DCC
        codebook = 1.0 - 2.0 * codebook_bits
        self.codebook = torch.from_numpy(codebook)  # (4096, 32)

        self.bin_edges = np.linspace(-np.pi, np.pi, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def angle_to_codeword(self, theta):
        """
        theta: (B,) radians
        returns: (B, 32) bits in {0,1}
        """
        theta = torch.remainder(theta + np.pi, 2 * np.pi)
        q = torch.floor(theta / (2 * np.pi) * self.num_bins).long()
        q = torch.clamp(q, 0, self.num_bins - 1)
        g = q ^ (q >> 1)
        shifts = torch.arange(self.num_bits - 1, -1, -1, device=q.device, dtype=q.dtype)
        g_bits = ((g.unsqueeze(1) >> shifts) & 1).float()
        blocks = torch.split(g_bits, self.block_bits, dim=1)
        codewords = []
        for b in blocks:
            idx = (b[:, 0].long() << 2) | (b[:, 1].long() << 1) | b[:, 2].long()
            cw = self.H8.to(b.device)[idx]  # (B, 8) bipolar
            cw_bits = (1.0 - cw) / 2.0
            codewords.append(cw_bits)
        return torch.cat(codewords, dim=1)

    def decode_data_bits_from_codeword(self, code_bits):
        """
        Decode 32-bit codeword bits to 12 data bits using per-block Hadamard ML.
        code_bits: (B, 32) in {0,1}
        returns: (B, 12) in {0,1}
        """
        if code_bits.shape[1] != self.code_bits:
            raise ValueError("Hadamard code length mismatch.")
        bits = code_bits.reshape(-1, self.num_blocks, self.block_code_bits)
        bipolar = 1.0 - 2.0 * bits  # (B, 4, 8)
        H = self.H8.to(bipolar.device)
        scores = torch.matmul(bipolar, H.t())  # (B, 4, 8)
        idx = torch.argmax(scores, dim=2)  # (B, 4)
        b0 = (idx >> 2) & 1
        b1 = (idx >> 1) & 1
        b2 = idx & 1
        data = torch.stack([b0, b1, b2], dim=2).reshape(-1, self.num_bits)
        return data.float()

    def decode_soft_expectation(self, logits):
        """
        Soft decode using codebook likelihoods.
        logits: (B, 21)
        returns: (B,) radians
        """
        device = logits.device
        probs = torch.sigmoid(logits)  # (B, 21)

        codes = self.all_codewords.to(device).unsqueeze(0)  # (1, 4096, 21)
        log_probs_1 = torch.log(probs + 1e-8).unsqueeze(1)
        log_probs_0 = torch.log(1.0 - probs + 1e-8).unsqueeze(1)
        bin_log_probs = torch.sum(
            codes * log_probs_1 + (1.0 - codes) * log_probs_0,
            dim=2
        )  # (B, 4096)
        bin_probs = torch.softmax(bin_log_probs, dim=1)

        angles_map = torch.tensor(self.bin_centers, device=device, dtype=torch.float32)
        sin_sum = torch.sum(bin_probs * torch.sin(angles_map), dim=1)
        cos_sum = torch.sum(bin_probs * torch.cos(angles_map), dim=1)
        return torch.atan2(sin_sum, cos_sum)

    def decode_hard_from_logits(self, logits):
        """
        Hard decode via codebook ML (argmax over bins).
        logits: (B, 21)
        returns: (B,) radians
        """
        device = logits.device
        probs = torch.sigmoid(logits)
        codes = self.all_codewords.to(device).unsqueeze(0)
        log_probs_1 = torch.log(probs + 1e-8).unsqueeze(1)
        log_probs_0 = torch.log(1.0 - probs + 1e-8).unsqueeze(1)
        bin_log_probs = torch.sum(
            codes * log_probs_1 + (1.0 - codes) * log_probs_0,
            dim=2
        )  # (B, 4096)
        bin_idx = torch.argmax(bin_log_probs, dim=1)
        angles = torch.tensor(self.bin_centers, device=device, dtype=torch.float32)[bin_idx]
        return angles

    def codeword_from_logits(self, logits):
        """
        Return the ML codeword bits from logits.
        logits: (B, 21)
        returns: (B, 21) in {0,1}
        """
        device = logits.device
        probs = torch.sigmoid(logits)
        codes = self.all_codewords.to(device).unsqueeze(0)
        log_probs_1 = torch.log(probs + 1e-8).unsqueeze(1)
        log_probs_0 = torch.log(1.0 - probs + 1e-8).unsqueeze(1)
        bin_log_probs = torch.sum(
            codes * log_probs_1 + (1.0 - codes) * log_probs_0,
            dim=2
        )
        bin_idx = torch.argmax(bin_log_probs, dim=1)
        return self.all_codewords.to(device)[bin_idx]


class DirectCodebookCorrector(nn.Module):
    """
    Soft ML decoding via codebook attention (no trainable weights except beta/T).
    """
    def __init__(self, codebook, temperature=1.0):
        super().__init__()
        self.register_buffer("codebook", codebook)  # (4096, 21) in {-1,+1}
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # Pre-normalize codebook
        cb = self.codebook
        cb_norm = cb / (cb.norm(dim=1, keepdim=True) + 1e-6)
        self.register_buffer("codebook_norm", cb_norm)

    def forward(self, logits):
        # logits: (B, code_bits)
        T = torch.clamp(self.temperature, 0.1, 10.0)
        x = torch.tanh(logits / T)  # bipolar proxy
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-6)
        scores = x_norm @ self.codebook_norm.t()  # (B, 4096)
        weights = F.softmax(scores, dim=1)
        recon = weights @ self.codebook  # (B, 21) bipolar
        x_corr = x + self.beta * (recon - x)
        x_corr = torch.clamp(x_corr, -0.999, 0.999)
        logits_corr = 0.5 * torch.log((1 + x_corr) / (1 - x_corr)) * T
        return logits_corr


class HeadingBinaryHead(nn.Module):
    """
    Backbone -> code_bits logits -> DCC correction.
    """
    def __init__(self, feat_dim, quantizer, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, quantizer.code_bits),
        )
        self.dcc = DirectCodebookCorrector(quantizer.codebook, temperature=1.0)

    def forward(self, feat):
        logits = self.fc(feat)
        return self.dcc(logits)


class HadamardEnergyLoss(nn.Module):
    """
    Encourage logits to be close to bipolar {-1,+1} after tanh.
    """
    def __init__(self, T=1.0):
        super().__init__()
        self.T = T

    def forward(self, logits):
        x = torch.tanh(logits / self.T)
        return torch.mean((x.abs() - 1.0) ** 2)
