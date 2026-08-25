import torch
import torch.nn as nn

from ddpm import TimeEmbedding


def group_norm(channels):
    return nn.GroupNorm(8, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.block1 = nn.Sequential(
            group_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            group_norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x).view(batch, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        scale = channels ** -0.5
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2))
        h = self.proj(h).view(batch, channels, height, width)
        return x + h


class Down(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim // 4)
        t_dim = time_emb_dim

        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4
        ch4 = base_ch * 8

        self.in_conv = nn.Conv2d(in_channels, ch1, 3, padding=1)

        self.down1 = nn.ModuleList([
            ResBlock(ch1, ch1, t_dim),
            ResBlock(ch1, ch1, t_dim),
        ])
        self.pool1 = Down(ch1)

        self.down2 = nn.ModuleList([
            ResBlock(ch1, ch2, t_dim),
            ResBlock(ch2, ch2, t_dim),
        ])
        self.pool2 = Down(ch2)

        self.down3 = nn.ModuleList([
            ResBlock(ch2, ch3, t_dim),
            ResBlock(ch3, ch3, t_dim),
        ])
        self.pool3 = Down(ch3)

        self.down4 = nn.ModuleList([
            ResBlock(ch3, ch4, t_dim),
            ResBlock(ch4, ch4, t_dim),
            AttentionBlock(ch4),
        ])
        self.pool4 = Down(ch4)

        self.mid = nn.ModuleList([
            ResBlock(ch4, ch4, t_dim),
            AttentionBlock(ch4),
            ResBlock(ch4, ch4, t_dim),
        ])

        self.up4 = Up(ch4)
        self.dec4 = nn.ModuleList([
            ResBlock(ch4 + ch4, ch3, t_dim),
            ResBlock(ch3, ch3, t_dim),
        ])

        self.up3 = Up(ch3)
        self.dec3 = nn.ModuleList([
            ResBlock(ch3 + ch3, ch2, t_dim),
            ResBlock(ch2, ch2, t_dim),
        ])

        self.up2 = Up(ch2)
        self.dec2 = nn.ModuleList([
            ResBlock(ch2 + ch2, ch1, t_dim),
            ResBlock(ch1, ch1, t_dim),
        ])

        self.up1 = Up(ch1)
        self.dec1 = nn.ModuleList([
            ResBlock(ch1 + ch1, ch1, t_dim),
            ResBlock(ch1, ch1, t_dim),
        ])

        self.out_conv = nn.Sequential(
            group_norm(ch1),
            nn.SiLU(),
            nn.Conv2d(ch1, in_channels, 3, padding=1),
        )

    def apply_blocks(self, blocks, x, t_emb):
        for block in blocks:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
        return x

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x = self.in_conv(x)

        d1 = self.apply_blocks(self.down1, x, t_emb)
        x = self.pool1(d1)

        d2 = self.apply_blocks(self.down2, x, t_emb)
        x = self.pool2(d2)

        d3 = self.apply_blocks(self.down3, x, t_emb)
        x = self.pool3(d3)

        d4 = self.apply_blocks(self.down4, x, t_emb)
        x = self.pool4(d4)

        x = self.apply_blocks(self.mid, x, t_emb)

        x = self.up4(x)
        x = torch.cat([x, d4], dim=1)
        x = self.apply_blocks(self.dec4, x, t_emb)

        x = self.up3(x)
        x = torch.cat([x, d3], dim=1)
        x = self.apply_blocks(self.dec3, x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.apply_blocks(self.dec2, x, t_emb)

        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.apply_blocks(self.dec1, x, t_emb)

        return self.out_conv(x)
