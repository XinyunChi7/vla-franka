"""
FEATS (feats-ai/feats) UNet model and helpers for markerless force estimation.

Architecture and normalization conventions taken verbatim from:
  https://github.com/feats-ai/feats

Default pretrained weights:  unet_09042025_124903_80.pt
Default normalization file:  normalization_08042025_122519.npy

The model predicts three spatial force-distribution maps (grid_x, grid_y, grid_z)
at 24×32 resolution from a raw 320×240 RGB GelSight image.
Summing each map gives total Fx / Fy / Fz in physical units (no calibration
needed for the latest model weights).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# UNet architecture (matches feats-ai/feats src/feats/src/models/unet.py)
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class _Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [_Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class _Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [_Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x: torch.Tensor, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc = self._crop(encoder_features[i], x)
            x = torch.cat([x, enc], dim=1)
            x = self.dec_blocks[i](x)
        return x

    @staticmethod
    def _crop(enc_ftrs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        _, _, eH, eW = enc_ftrs.shape
        h0 = (eH - H) // 2
        w0 = (eW - W) // 2
        return enc_ftrs[:, :, h0:h0 + H, w0:w0 + W]


class FEATSUNet(nn.Module):
    """U-Net that predicts (Fx, Fy, Fz) force distributions from a GelSight image."""

    def __init__(
        self,
        enc_chs=(3, 16, 32, 64, 128, 256),
        dec_chs=(256, 128, 64, 32, 16),
        out_sz=(24, 32),
    ):
        super().__init__()
        self.encoder = _Encoder(enc_chs)
        self.decoder = _Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], 3, 1)
        self.out_sz = out_sz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        return F.interpolate(out, self.out_sz)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_feats(
    model_path: str,
    norm_path: str,
    device: torch.device = None,
) -> tuple:
    """
    Load FEATS model weights and normalization stats.

    Returns (model, norm_dict, device).
    norm_dict keys: 'grid_x', 'grid_y', 'grid_z' — each {'min': float, 'max': float}
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    model = FEATSUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    norm = np.load(norm_path, allow_pickle=True).item()
    return model, norm, device


def preprocess(frame_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Prepare a uint8 RGB image (H×W×3) for FEATS inference.
    Returns float tensor of shape (1, 3, H, W) with values in [0, 1].
    """
    img = frame_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def postprocess(
    output: torch.Tensor,
    norm: dict,
) -> tuple:
    """
    Unnormalize model output (1, 3, H, W) → (grid_x, grid_y, grid_z).

    grid_x / grid_y were normalized to [-1, 1]; grid_z to [0, 1].
    Each returned array has shape (24, 32) in physical force units.
    """
    out = output.squeeze(0).cpu().detach().numpy()  # (3, H, W)

    def _unnorm_xy(v: np.ndarray, key: str) -> np.ndarray:
        mn, mx = norm[key]['min'], norm[key]['max']
        return (v + 1.0) / 2.0 * (mx - mn) + mn

    def _unnorm_z(v: np.ndarray, key: str) -> np.ndarray:
        mn, mx = norm[key]['min'], norm[key]['max']
        return v * (mx - mn) + mn

    return _unnorm_xy(out[0], 'grid_x'), _unnorm_xy(out[1], 'grid_y'), _unnorm_z(out[2], 'grid_z')
