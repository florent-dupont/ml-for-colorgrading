"""Content-invariant movie-look encoder trained with supervised contrastive learning."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def shuffle_patches(images: Tensor, grid: int = 4) -> Tensor:
    """Destroy composition while preserving the image's local colour/texture population."""
    b, c, h, w = images.shape
    ph, pw = h // grid, w // grid
    images = images[:, :, :ph * grid, :pw * grid]
    patches = images.reshape(b, c, grid, ph, grid, pw).permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(b, grid * grid, c, ph, pw)
    shuffled = []
    for item in patches:
        shuffled.append(item[torch.randperm(grid * grid, device=images.device)])
    patches = torch.stack(shuffled).reshape(b, grid, grid, c, ph, pw)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(b, c, ph * grid, pw * grid)


class MovieStyleEncoder(nn.Module):
    """Shallow pretrained features plus a learned movie-style projection.

    Only ResNet's stem/layer1/layer2 are used; high-level semantic layers are
    deliberately absent. Contrastive training with shuffled compositions makes
    same-movie, different-scene frames converge in this embedding.
    """

    embedding_size = 256

    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.projection = nn.Sequential(
            nn.Linear(2 * (64 + 128), 384), nn.GELU(),
            nn.Linear(384, self.embedding_size),
        )
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    @staticmethod
    def _moments(feature: Tensor) -> Tensor:
        return torch.cat((feature.mean((2, 3)),
                          feature.var((2, 3), unbiased=False).add(1e-6).sqrt()), 1)

    def forward(self, display: Tensor, destroy_content: bool = False) -> Tensor:
        if destroy_content:
            display = shuffle_patches(display)
        x = (display - self.mean) / self.std
        f1 = self.layer1(self.stem(x))
        f2 = self.layer2(f1)
        return F.normalize(self.projection(torch.cat((self._moments(f1),
                                                       self._moments(f2)), 1)), dim=1)


def supervised_contrastive_loss(embedding: Tensor, labels: Tensor,
                                temperature: float = 0.10) -> Tensor:
    """Pull same-movie/different-frame embeddings together; push movies apart."""
    similarity = embedding @ embedding.T / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    denominator = torch.logsumexp(similarity.masked_fill(self_mask, -torch.inf), dim=1)
    log_probability = similarity - denominator[:, None]
    valid = positive.any(1)
    return -(log_probability.masked_fill(~positive, 0).sum(1)[valid] /
             positive.sum(1)[valid]).mean()


def load_style_encoder(checkpoint: str, device: str) -> MovieStyleEncoder:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if blob.get("style_encoder_version") != 1:
        raise ValueError(f"unsupported style encoder checkpoint: {checkpoint}")
    model = MovieStyleEncoder()
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()
