from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class SimpleViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def create_model(config: dict, num_classes: int, image_size: int) -> nn.Module:
    model_config = config["model"]
    backend = model_config.get("backend", "simple").lower()

    if backend == "timm":
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm or set model.backend=simple in the config.") from exc

        return timm.create_model(
            model_config.get("name", "vit_tiny_patch16_224"),
            pretrained=bool(model_config.get("pretrained", False)),
            num_classes=num_classes,
            img_size=image_size,
        )

    if backend != "simple":
        raise ValueError(f"Unsupported model backend: {backend}")

    return SimpleViT(
        image_size=image_size,
        patch_size=int(model_config.get("patch_size", 16)),
        num_classes=num_classes,
        embed_dim=int(model_config.get("embed_dim", 192)),
        depth=int(model_config.get("depth", 6)),
        num_heads=int(model_config.get("num_heads", 3)),
        mlp_ratio=float(model_config.get("mlp_ratio", 4.0)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
