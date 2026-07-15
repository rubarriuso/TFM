import torch
import torch.nn as nn
import torch.nn.functional as F

#Bloques
#======================================================================================
# -------- Basic Conv Block --------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dp=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Dropout3d(dp),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU())

    def forward(self, x):
        return self.block(x)

# -------- Decoder Block --------
class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dp=0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, (1, 2, 2), stride=(1, 2, 2))
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels, dp)
        
    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatches
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)

        x = self.conv_block(x)
        return x

# -------- Temporal Aggregation Block --------
class TemporalAggregationBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.temporal_attn = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=in_channels),
            nn.Sigmoid())
        self.temporal_score = nn.Conv3d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        attn = self.temporal_attn(x)
        feat = x * attn
        score = self.temporal_score(feat)
        weights = torch.softmax(score, dim=2) 
        feat = (feat * weights).sum(dim=2)    
        return feat

# -------- Global Attention Branch --------
class GlobalAttentionBlock(nn.Module):
    def __init__(self, channels, time_pooling=None):
        super().__init__()
        reduction = max(channels // 4, 8)
        self.branch = nn.Sequential(
            nn.AdaptiveAvgPool3d((time_pooling, 1, 1)),
            nn.Conv3d(channels, reduction, 1),
            nn.PReLU(reduction),
            nn.Conv3d(reduction, channels, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * (self.branch(x))
    
    
class GatedTemporalMixBlock(nn.Module):
    def __init__(self, channels, window_size=5):
        super().__init__()

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")

        pad_t = window_size // 2
        kernel = (window_size, 1, 1)
        padding = (pad_t, 0, 0)

        self.value = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel,
                padding=padding,
                bias=False),
            nn.BatchNorm3d(channels),
            nn.PReLU(channels),
        )

        self.gate = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel,
                padding=padding),
            nn.Sigmoid())

    def forward(self, x):
        mix = self.value(x)
        gate = self.gate(x)
        return x + (mix * gate)
    
class RegressionHead(nn.Module):

    def __init__(self, base_ch: int, out_channels: int, out_hw: tuple):
        
        super(RegressionHead, self).__init__()
        self.out_hw = out_hw
        self.refinement= nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.PReLU(base_ch))
        
        self.regressor = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.PReLU(base_ch),
            nn.Dropout2d(0.1),
            nn.Conv2d(base_ch, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.refinement(x)

        # Interpolate spatial dimensions to match out_hw
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        
        # Apply regression layers
        out = self.regressor(x)

        return out

class ClassificationHead(nn.Module):

    def __init__(self, base_ch: int, out_channels: int, out_hw: tuple):
        
        super(ClassificationHead, self).__init__()
        self.out_hw = out_hw
        self.refinement= nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.PReLU(base_ch),
        )
        
        self.regressor = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.PReLU(base_ch),
            nn.Conv2d(base_ch, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.refinement(x)

        # Interpolate spatial dimensions to match out_hw
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        
        # Apply regression layers
        out = self.regressor(x)
        return out

#Modelos
#======================================================================================
# -------- U-Net 3D classification temporal collapse after bottleneck --------
class UNet3DCla(nn.Module):
    def __init__(self, in_channels, out_hw, num_classes, base_ch=8):
        super().__init__()
        dp_lvl1 = 0.20
        dp_lvl2 = 0.3
        dp_lvl3 = 0.3
        self.out_hw = out_hw  # (H, W)
        self.num_classes = num_classes

        # -------- Encoder --------
        self.enc1 = ConvBlock(in_channels, base_ch, dp=dp_lvl1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))


        self.enc2 = ConvBlock(base_ch, base_ch * 2, dp=dp_lvl2)
        self.temp_mix2 = GatedTemporalMixBlock(base_ch * 2, window_size=7)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # -------- Bottleneck --------
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4, dp=dp_lvl3)
        self.temp_mix = GatedTemporalMixBlock(base_ch * 4, window_size=7)
        self.global_att = GlobalAttentionBlock(base_ch * 4)
        
        # -------- Decoder --------
        self.decoder2 = DecoderBlock3D(base_ch * 4, base_ch * 2, base_ch * 2, dp=dp_lvl2)
        self.decoder1 = DecoderBlock3D(base_ch * 2, base_ch, base_ch, dp=dp_lvl1)
    
        # -------- Temporal Aggregation --------
        self.temporal_agg = TemporalAggregationBlock(base_ch)

        # -------- Classifier Head --------
        self.output_head = ClassificationHead(base_ch=base_ch, out_channels=self.num_classes, out_hw=self.out_hw)

    def forward(self, x, mask=None):
        # INPUT COMES AS: (B, T, C, H, W) Convert to:(B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # -------- Encoder --------
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s2 = self.temp_mix2(s2)

        # -------- Bottleneck --------
        b = self.bottleneck(p2)
        b = self.temp_mix(b)     
        b = self.global_att(b)

        # -------- Decoder --------
        d2 = self.decoder2(b, s2)
        d1 = self.decoder1(d2, s1)

        # -------- Temporal Aggregation --------
        final_2d = self.temporal_agg(d1)
        
        # -------- Classifier Head --------
        logits = self.output_head(final_2d)

        if mask is not None:
            # Match out shape: (B, 1, H, W)
            logits = logits * mask.unsqueeze(1).float()
        return logits

# -------- U-Net 3D temporal collapse after bottleneck --------
class UNet3DReg(nn.Module):
    def __init__(self, in_channels, out_hw, out_channels=1, base_ch=10):
        super().__init__()
        dp_lvl1 = 0.25
        dp_lvl2 = 0.3
        dp_lvl3 = 0.3
        self.out_hw = out_hw  # (H, W)

        # -------- Encoder --------
        self.enc1 = ConvBlock(in_channels, base_ch, dp=dp_lvl1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))


        self.enc2 = ConvBlock(base_ch, base_ch * 2, dp=dp_lvl2)
        self.temp_mix2 = GatedTemporalMixBlock(base_ch * 2, window_size=7)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # -------- Bottleneck --------
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4, dp=dp_lvl3)
        self.temp_mix = GatedTemporalMixBlock(base_ch * 4, window_size=7)

        self.global_att = GlobalAttentionBlock(base_ch * 4)
        
        # -------- Decoder --------
        self.decoder2 = DecoderBlock3D(base_ch * 4, base_ch * 2, base_ch * 2, dp=dp_lvl2)
        self.decoder1 = DecoderBlock3D(base_ch * 2, base_ch, base_ch, dp=dp_lvl1)
    
        # -------- Temporal Aggregation --------
        self.temporal_agg = TemporalAggregationBlock(base_ch)

        # -------- Regression Head --------
        self.regression_head = RegressionHead(base_ch=base_ch, out_channels=out_channels, out_hw=self.out_hw)

    def forward(self, x, mask=None):
        # INPUT COMES AS: (B, T, C, H, W) Convert to:(B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # -------- Encoder --------
        s1 = self.enc1(x)
        p1 = self.pool1(s1)


        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s2 = self.temp_mix2(s2)

        # -------- Bottleneck --------
        b = self.bottleneck(p2)
        b = self.temp_mix(b)     
        b = self.global_att(b)

        # -------- Decoder --------
        d2 = self.decoder2(b, s2)
        d1 = self.decoder1(d2, s1)

        # -------- Temporal Aggregation --------
        final_2d = self.temporal_agg(d1)
        
        # -------- Regression Head --------
        out = self.regression_head(final_2d)
        
  


        if mask is not None:
            # Match out shape: (B, 1, H, W)
            out = out * mask.unsqueeze(1).float()
        return out