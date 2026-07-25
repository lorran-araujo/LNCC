import torch
import torch.nn as nn

from ddpm import TimeEmbedding

# ----------------------------------------------------------------------------
# Blocos da UNet
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# ----------------------------------------------------------------------------
# UNet 
# ----------------------------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim // 4)
        t_dim = time_emb_dim  # saída do MLP dentro de TimeEmbedding é dim*4

        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Encoder
        self.down1 = ResBlock(base_ch, base_ch, t_dim)
        self.pool1 = Down(base_ch)
        self.down2 = ResBlock(base_ch, base_ch * 2, t_dim)
        self.pool2 = Down(base_ch * 2)

        # Bottleneck
        self.mid = ResBlock(base_ch * 2, base_ch * 2, t_dim)

        # Decoder
        self.up2 = Up(base_ch * 2)
        self.dec2 = ResBlock(base_ch * 2 + base_ch * 2, base_ch, t_dim)
        self.up1 = Up(base_ch)
        self.dec1 = ResBlock(base_ch + base_ch, base_ch, t_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x0 = self.in_conv(x)

        d1 = self.down1(x0, t_emb)
        p1 = self.pool1(d1)

        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)

        m = self.mid(p2, t_emb)

        u2 = self.up2(m)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1, t_emb)

        return self.out_conv(u1)
