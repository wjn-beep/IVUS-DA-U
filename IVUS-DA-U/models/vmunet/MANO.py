import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    """
    MLP block with pre-layernorm, GELU activation, and dropout.
    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """
    Global multi-head self-attention block with optional projection.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        Expected input shape: [B, L, C]
        """
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LocalAttention2D(nn.Module):
    """
    Windowed/local attention for 2D grids using unfold & fold.
    """
    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.Attention = AttentionBlock(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = rearrange(x, "B H W C -> B C H W")

        # unfold into local 2D patches
        patches = self.unfold(x)  # [B, C*K*K, L]
        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )
        patches = self.norm(patches)

        # Intra-Window attention
        out = self.Attention(patches)  # [B*L, K*K, C]

        # Reshape back to [B, C*K*K, L]
        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        # Fold back to [B, C, H, W]
        fold = nn.Fold(
            output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride
        )
        out = fold(out)

        # Normalize overlapping regions
        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))
        norm = fold(norm)
        out = out / norm

        # Reshape to [B, H, W, C]
        out = rearrange(out, "B C H W -> B H W C")
        return out


class Multipole_Attention(nn.Module):
    """
    多尺度层次化局部注意力机制（支持下采样和上采样）。
    输入输出格式：(B, C, H, W)
    """
    def __init__(
        self,
        in_channels,
        image_size,
        local_attention_kernel_size=2,
        local_attention_stride=2,
        downsampling="conv",
        upsampling="conv",
        sampling_rate=2,
        heads=4,
        dim_head=16,
        dropout=0.1,
        channel_scale=1,
    ):
        super().__init__()
        
        # 计算金字塔层级数
        self.levels = int(math.log(image_size, sampling_rate))
        
        # 验证image_size的合法性
        assert image_size == sampling_rate ** self.levels, \
            f"image_size ({image_size}) 必须是 sampling_rate ({sampling_rate}) 的整数次幂"
        
        # 定义每一层的通道数
        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        # 定义局部注意力模块
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=channels_conv[0],
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # 下采样模块
        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )
        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False
                ),
                Rearrange("B C H W -> B H W C"),
            )

        # 上采样模块
        if upsampling == "avg_pool":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )
        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        """
        输入: x 形状为 [B, C, H, W]
        输出: res 形状为 [B, C, H, W]
        """
        # 转换为 [B, H, W, C] 格式
        x = x.permute(0, 2, 3, 1)
        x_in = x

        x_out = []
        x_out.append(self.Attention(x_in))  # 第0层（原分辨率）

        # 多尺度下采样 + 注意力
        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.Attention(x_in)
            x_out.append(x_out_down)

        # 从最粗尺度开始融合
        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        # ===== 关键修复：正确的维度转换 =====
        # 从 [B, H, W, C] 转回 [B, C, H, W]
        return res.permute(0, 3, 1, 2)  # 修改这里！


class Multipole_Attention_BHWC(nn.Module):
    """
    多尺度层次化局部注意力机制
    输入输出格式：(B, H, W, C)
    """
    def __init__(
        self,
        in_channels,
        image_size,
        local_attention_kernel_size=2,
        local_attention_stride=2,
        downsampling="conv",
        upsampling="conv",
        sampling_rate=2,
        heads=4,
        dim_head=16,
        dropout=0.1,
        channel_scale=1,
    ):
        super().__init__()

        self.levels = int(math.log(image_size, sampling_rate))
        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=channels_conv[0],
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )
        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

        if upsampling == "avg_pool":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )
        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        """
        输入输出: x 形状为 [B, H, W, C]
        """
        x_in = x
        x_out = []
        x_out.append(self.Attention(x_in))

        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.Attention(x_in)
            x_out.append(x_out_down)

        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        return res


class MultipoleBlock(nn.Module):
    """
    Transformer block stacking multiple Multipole_Attention + FeedForward layers.
    输入输出格式：(B, C, H, W)
    """
    def __init__(
        self,
        in_channels,
        image_size,
        kernel_size=2,
        local_attention_stride=2,
        downsampling="conv",
        upsampling="conv",
        sampling_rate=2,
        depth=2,
        heads=4,
        dim_head=16,
        att_dropout=0.1,
        channel_scale=1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.layers = nn.ModuleList([])
        mlp_dim = int(4 * in_channels)
        
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Multipole_Attention_BHWC(
                        in_channels,
                        image_size,
                        kernel_size,
                        local_attention_stride,
                        downsampling,
                        upsampling,
                        sampling_rate,
                        heads,
                        dim_head,
                        att_dropout,
                        channel_scale,
                    ),
                    FeedForward(in_channels, mlp_dim),
                ])
            )

    def forward(self, x):
        """
        输入输出: x 形状为 [B, C, H, W]
        """
        # 转换为 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        x = self.norm(x)
        
        # 转换回 [B, C, H, W]
        return x.permute(0, 3, 1, 2)


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 Multipole_Attention 模块")
    print("=" * 60)
    
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(2, 32, 64, 64)
    
    # 创建 Multipole_Attention 模块
    MA = Multipole_Attention(
        in_channels=32,
        image_size=64,
        heads=8,
        dim_head=4,
    )
    
    # 前向传播
    output = MA(input)
    
    # 打印结果
    print(f'输入形状: {input.size()}')
    print(f'输出形状: {output.size()}')
    print(f'输入输出形状是否一致: {input.size() == output.size()}')
    
    print("\n" + "=" * 60)
    print("测试 MultipoleBlock 模块")
    print("=" * 60)
    
    # 创建 MultipoleBlock
    MABlock = MultipoleBlock(
        in_channels=32,
        image_size=64,
        depth=2,
        heads=8,
        dim_head=4,
    )
    
    # 前向传播
    output_block = MABlock(input)
    
    # 打印结果
    print(f'输入形状: {input.size()}')
    print(f'输出形状: {output_block.size()}')
    print(f'输入输出形状是否一致: {input.size() == output_block.size()}')
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
