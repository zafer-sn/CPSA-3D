import torch
import math
import utils

class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(torch.nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ComplexFrequencyAttentionMask(torch.nn.Module):
    def __init__(self, in_channels):
        super(ComplexFrequencyAttentionMask, self).__init__()
        # Modulator network to predict complex modulation mask
        self.modulator = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, bias=False) # Output real and imag parts of mask
        )
        
        # Refine the output after IFFT to a single-channel mask
        self.refine_output = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 1. Compute FFT
        fft_x = torch.fft.fft2(x, norm='ortho')
        
        # 2. Prepare input for modulator (real and imaginary parts concatenated)
        fft_x_combined = torch.cat([fft_x.real, fft_x.imag], dim=1)
        
        # 3. Predict complex modulation mask
        modulation_output = self.modulator(fft_x_combined)
        modulation_real, modulation_imag = torch.chunk(modulation_output, 2, dim=1)
        
        modulation_mask = torch.complex(modulation_real, modulation_imag)
        
        # 4. Apply modulation in frequency domain
        new_fft_x = fft_x * modulation_mask
        
        # 5. Inverse FFT
        ifft_x = torch.fft.ifft2(new_fft_x, norm='ortho')
        
        # 6. Combine real and imaginary parts and refine to a single-channel mask
        ifft_x_combined_spatial = torch.cat([ifft_x.real, ifft_x.imag], dim=1)
        spatial_mask = self.refine_output(ifft_x_combined_spatial)
        
        return self.sigmoid(spatial_mask)

class CPSA(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CPSA, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio) # Use existing ChannelAttention
        self.sa = ComplexFrequencyAttentionMask(in_planes) # Use new complex frequency attention mask

    def forward(self, x):
        # This flow is identical to CBAM
        x_ca = self.ca(x) * x # Apply channel attention
        sa_mask = self.sa(x_ca) # Generate spatial mask from channel-attended features
        x_out = x_ca * sa_mask # Apply spatial mask multiplicatively
        return x_out

class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.cube_len = args.cube_len

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.args.z_size, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, self.args.z_size, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


class _D(torch.nn.Module):
    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args
        self.cube_len = args.cube_len

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=args.bias, padding=padd),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out

class _E_MultiView(torch.nn.Module):
    def output_features(self,size,kernel_size,stride,padding):

        out = (((size - kernel_size) + (2*padding)) // stride) + 1
        return out

    def __init__(self, args):
        super(_E_MultiView, self).__init__()
        self.args = args
        self.img_size = args.image_size
        self.combine_type = args.combine_type
        self.attention_type = getattr(args, 'attention_type', 'none')

        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 400, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(400),
            torch.nn.ReLU()
        )
        input = self.img_size
        for i in range(5):
            input = self.output_features(input,5,2,2)
        self.FC1 = torch.nn.Linear(400*input*input,200)
        self.FC2 = torch.nn.Linear(400*input*input, 200)

        if self.attention_type == 'cbam':
            self.att1 = CBAM(128)
            self.att2 = CBAM(256)
            self.att3 = CBAM(512)
        elif self.attention_type == 'se':
            self.att1 = SELayer(128)
            self.att2 = SELayer(256)
            self.att3 = SELayer(512)
        elif self.attention_type == 'cpsa':
            self.att1 = CPSA(128)
            self.att2 = CPSA(256)
            self.att3 = CPSA(512)
        else:
            self.att1 = torch.nn.Identity()
            self.att2 = torch.nn.Identity()
            self.att3 = torch.nn.Identity()


    def forward(self,images):
        # Batch boyutunu sabit tut (testte drop_last=True olduğundan güvenli)
        bs = self.args.batch_size
        means = utils.var_or_cuda(torch.zeros(self.args.num_views, bs, 200))
        vars = utils.var_or_cuda(torch.zeros(self.args.num_views, bs, 200))
        zs = utils.var_or_cuda(torch.zeros(self.args.num_views, bs, 200))
        for i, image in enumerate(images):
            image = utils.var_or_cuda(image)
            z_mean, z_log_var = self.single_image_forward(image)  # [B,200]
            z_sample = self.reparameterize(z_mean, z_log_var)     # [B,200]
            zs[i] = z_sample
            means[i] = z_mean
            vars[i] = z_log_var
        return self.combine(zs), means, vars

    def combine(self,input):        
        if self.combine_type == 'mean':
            output =  torch.mean(input,0)
        elif self.combine_type == 'max':
            output = torch.max(input, 0)[0]
        elif self.combine_type == 'concat':
            pass
        return output
    def single_image_forward(self, x):
        # x shape can be [B, 4, H, W] or [4, H, W]; normalize to [B, 4, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.att1(out)
        out = self.layer3(out)
        out = self.att2(out)
        out = self.layer4(out)
        out = self.att3(out)
        out = self.layer5(out)

        out = out.view(out.size(0), -1)  # flatten using actual batch size
        z_mean = self.FC1(out)
        z_log_var = self.FC2(out)

        return z_mean, z_log_var

    def reparameterize(self, mu, var):
        if self.training:
            std = var.mul(0.5).exp_()
            eps = utils.var_or_cuda((std.data.new(std.size()).normal_()))
            z =  eps.mul(std).add_(mu)
            return z
        else:
            return mu

