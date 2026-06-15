import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionWeight(nn.Module):

    def __init__(self, channel, kernel_size=7):
        super(AttentionWeight, self).__init__()
        padding = (kernel_size - 1) // 2
        

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1) 
        

        self.conv2 = nn.Conv1d(
            channel, channel, 
            kernel_size, 
            padding=padding, 
            groups=channel, 
            bias=False
        )
        self.bn = nn.BatchNorm1d(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, W, C, H) 或 (B, H, C, W) 取决于处理方向
        Returns:
            加权后的特征
        """
        b, w, c, h = x.size()
        

        x_avg = torch.mean(x, 1).unsqueeze(1) 
        x_max = torch.max(x, 1)[0].unsqueeze(1)  
        x_weight = torch.cat((x_max, x_avg), dim=1)  
        

        x_weight = self.conv1(x_weight).view(b, c, h)  

        x_weight = self.sigmoid(self.bn(self.conv2(x_weight)))
        x_weight = x_weight.view(b, 1, c, h)

        return x * x_weight


class IIA(nn.Module):
    """
    Information Integration Attention (IIA) - 论文正确实现版本
    
    核心修改:
    1. 增加双输入接口(enc_feat, dec_feat)
    2. 添加特征拼接逻辑
    3. 添加通道调整层
    """
    def __init__(self, channel):
        super(IIA, self).__init__()

        self.channel_reduce = nn.Conv2d(
            channel * 2,  
            channel,     
            kernel_size=1,
            bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(channel)
        

        self.attention = AttentionWeight(channel)

    def forward(self, enc_feat, dec_feat):  
        """
        Args:
            enc_feat: 编码器特征 (B, C, H, W)
            dec_feat: 解码器特征 (B, C, H, W)
        Returns:
            融合后的特征 (B, C, H, W)
        """
  
        M = torch.cat([enc_feat, dec_feat], dim=1)  
        

        M = F.relu(self.bn_reduce(self.channel_reduce(M)))  
        


        x_h = M.permute(0, 3, 1, 2).contiguous()
        x_h = self.attention(x_h).permute(0, 2, 3, 1)  
        

        x_w = M.permute(0, 2, 1, 3).contiguous()
        x_w = self.attention(x_w).permute(0, 2, 1, 3) 

        return M + x_h + x_w



class ChannelAttention(nn.Module):
    def __init__(self, inp, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(inp, inp // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inp // ratio, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    net = IIA(512).to(device)
    
    enc_feat = torch.rand(8, 512, 16, 16).to(device)  
    dec_feat = torch.rand(8, 512, 16, 16).to(device) 
    
    print("=" * 60)
    print("IIA Module Test ")
    print("=" * 60)
    

    y = net(enc_feat, dec_feat)
    
    print(f"Encoder feature shape: {enc_feat.shape}")
    print(f"Decoder feature shape: {dec_feat.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    



    print("\n" + "=" * 60)
    print("=" * 60)
    
    class IIA_Compatible(IIA):
        def forward(self, x, y=None):
            if y is None:

                y = x.clone()
            return super().forward(x, y)
    
    net_compat = IIA_Compatible(512).to(device)
    x_single = torch.rand(8, 512, 16, 16).to(device)
    y_single = net_compat(x_single)  
    
    print(f"Single input shape: {x_single.shape}")
    print(f"Output shape: {y_single.shape}")

