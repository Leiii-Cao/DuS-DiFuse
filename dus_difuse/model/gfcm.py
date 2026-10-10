import torch as th
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.norm = nn.GroupNorm(min(32, out_channels//4), out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        x = th.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class GroupwiseFusionBlock(nn.Module):
    def __init__(self, channel, time_emb_dim, expansion=4):
        super().__init__()
        self.model_channels = channel * expansion
        self.time_emb_to_channel = nn.Sequential(
            nn.Linear(time_emb_dim, channel * 2),
            nn.SiLU(),
            nn.Linear(channel * 2, channel)
        )
        self.inBlock = nn.Sequential(
            DepthwiseSeparableConv(in_channels = channel * 2 + channel, out_channels = self.model_channels, kernel_size=3, padding=1),
        )
        self.RES = nn.Sequential(
            DepthwiseSeparableConv(in_channels = self.model_channels, out_channels = self.model_channels, kernel_size = 3, padding=1),
        )

        
        self.CBAM = CBAM(self.model_channels)

        self.outBlock = nn.Sequential(
            nn.Conv2d(self.model_channels, channel * 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2, time_emb):
        """Fuse one group of visible/infrared features at timestep ``time_emb``."""
        B, C, H, W = feat1.shape
        t = self.time_emb_to_channel(time_emb).view(B, C, 1, 1).expand(B, C, H, W)
        x = th.cat([feat1, feat2, t], dim=1)
        h = self.inBlock(x)
        h = self.RES(h) + h
        h = self.CBAM(h)
        weights = self.outBlock(h)
        visible_weight, infrared_weight, bias = weights.chunk(3, dim=1)
        return feat1 * visible_weight + feat2 * infrared_weight + bias


class GroupwiseFusionControlModule(nn.Module):
    def __init__(self, input_block_chans, time_emb_dim, enable_checkpointing):
        super().__init__()
        self.dual_skip_nets = nn.ModuleList()
        self._block_map = nn.ModuleDict()  
        self.enable_checkpointing = enable_checkpointing

        for ch in input_block_chans:
            key = str(ch)
            if key not in self._block_map:
                block = GroupwiseFusionBlock(channel=ch, time_emb_dim=time_emb_dim)
                self._block_map[key] = block
            self.dual_skip_nets.append(self._block_map[key])

        bottleneck_ch = input_block_chans[-1]
        key = str(bottleneck_ch)
        if key not in self._block_map:
            self._block_map[key] = GroupwiseFusionBlock(channel=bottleneck_ch, time_emb_dim=time_emb_dim)
        self.bottleneck_fusion = self._block_map[key]

    def _checkpoint_forward(self, block, *inputs):
        if self.enable_checkpointing and self.training:
            return checkpoint.checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    def forward(self, hs1, hs2, feats1, feats2, emb):
        fused_feats = []
        for f1, f2 in zip(feats1, feats2):
            ch = f1.shape[1]
            block = self._block_map[str(ch)]  
            fused_feats.append(self._checkpoint_forward(block, f1, f2, emb))

        bottleneck_ch = hs1.shape[1]
        bottleneck_block = self._block_map[str(bottleneck_ch)]
        fused_bottleneck = self._checkpoint_forward(bottleneck_block, hs1, hs2, emb)

        return fused_bottleneck, fused_feats
