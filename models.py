import torch
import torch.nn as nn
import os


class Noise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        if self.training:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            return x + self.weight.view(1, -1, 1, 1) * noise
        return x 
    

class Conv2d_WS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        std = weight.std(dim=(1, 2, 3), keepdim=True) + 1e-5
        weight = (weight - mean) / std 
        return nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample=None):
        super().__init__()
        self.ic = in_channels
        self.oc = out_channels

        self.conv1 = Conv2d_WS(self.ic, self.oc, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d_WS(self.oc, self.oc, kernel_size=3, stride=1, padding=1)
        self.conv_skip = Conv2d_WS(self.ic, self.oc, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(32, self.oc)
    
        self.activation = nn.LeakyReLU(0.02, inplace=True)

        self.sample = sample
        if self.sample == 'down':
            self.sample_layer = nn.Sequential(
                Conv2d_WS(self.oc, self.oc, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, self.oc),
                nn.LeakyReLU(0.02, inplace=True)
            )
        elif self.sample == 'up':
            self.sample_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2d_WS(self.oc, self.oc, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, self.oc),
                nn.LeakyReLU(0.02, inplace=True)
            )

    def forward(self, x):
        if self.ic != self.oc:
            identity = self.conv_skip(x)
            identity = self.norm(identity)
        else:
            identity = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)
        x += identity
        x = self.activation(x)
        if self.sample:
            x = self.sample_layer(x)
        return x
    

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            Conv2d_WS(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32, F_int)
        )
        self.W_l = nn.Sequential(
            Conv2d_WS(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32, F_int)
        )
        self.psi = nn.Sequential(
            Conv2d_WS(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.LeakyReLU(0.02, inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    

class Sketch2Color(nn.Module):
    def __init__(self, input_channels, pretrained=False):
        super().__init__()
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ResidualBlock(input_channels, 64, sample='down')
                self.layer2 = ResidualBlock(64, 128, sample='down')
                self.layer3 = ResidualBlock(128, 256, sample='down')
                self.layer4 = ResidualBlock(256, 512, sample='down')
                self.layer5 = ResidualBlock(512, 512, sample='down')
                self.layer6 = ResidualBlock(512, 512, sample='down')
                self.layer7 = ResidualBlock(512, 512, sample='down')

            def forward(self, input):
                x1 = self.layer1(input)
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
                x5 = self.layer5(x4)
                x6 = self.layer6(x5)
                x7 = self.layer7(x6)
                return x1, x2, x3, x4, x5, x6, x7
            
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.noise7 = Noise(512)
                self.layer7_up = ResidualBlock(512, 512, sample='up')

                self.att6 = AttentionBlock(F_g=512, F_l=512, F_int=256)
                self.layer6 = ResidualBlock(1024, 512, sample=None)
                self.noise6 = Noise(512)
                self.layer6_up = ResidualBlock(512, 512, sample='up')

                self.att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
                self.layer5 = ResidualBlock(1024, 512, sample=None)
                self.noise5 = Noise(512)
                self.layer5_up = ResidualBlock(512, 512, sample='up')

                self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
                self.layer4 = ResidualBlock(1024, 512, sample=None)
                self.noise4 = Noise(512)
                self.layer4_up = ResidualBlock(512, 256, sample='up')

                self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
                self.layer3 = ResidualBlock(512, 256, sample=None)
                self.noise3 = Noise(256)
                self.layer3_up = ResidualBlock(256, 128, sample='up')

                self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
                self.layer2 = ResidualBlock(256, 128, sample=None)
                self.noise2 = Noise(128)
                self.layer2_up = ResidualBlock(128, 64, sample='up')

                self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
                self.layer1 = ResidualBlock(128, 64, sample=None)
                self.noise1 = Noise(64)
                self.layer1_up = ResidualBlock(64, 32, sample='up')

                self.noise0 = Noise(32)
                self.layer0 = Conv2d_WS(32, 3, kernel_size=3, stride=1, padding=1)
                self.tanh = nn.Tanh()

            def forward(self, mid_input):
                x1, x2, x3, x4, x5, x6, x7 = mid_input
                x = self.noise7(x7)
                x = self.layer7_up(x)

                x6 = self.att6(g=x, x=x6)
                x = torch.cat((x, x6), dim=1)
                x = self.layer6(x)
                x = self.noise6(x)
                x = self.layer6_up(x)

                x5 = self.att5(g=x, x=x5)
                x = torch.cat((x, x5), dim=1)
                x = self.layer5(x)
                x = self.noise5(x)
                x = self.layer5_up(x)

                x4 = self.att4(g=x, x=x4)
                x = torch.cat((x, x4), dim=1)
                x = self.layer4(x)
                x = self.noise4(x)
                x = self.layer4_up(x)

                x3 = self.att3(g=x, x=x3)
                x = torch.cat((x, x3), dim=1)
                x = self.layer3(x)
                x = self.noise3(x)
                x = self.layer3_up(x)

                x2 = self.att2(g=x, x=x2)
                x = torch.cat((x, x2), dim=1)
                x = self.layer2(x)
                x = self.noise2(x)
                x = self.layer2_up(x)

                x1 = self.att1(g=x, x=x1)
                x = torch.cat((x, x1), dim=1)
                x = self.layer1(x)
                x = self.noise1(x)
                x = self.layer1_up(x)

                x = self.noise0(x)
                x = self.layer0(x)
                output = self.tanh(x)
                return output
        
        self.encoder = Encoder()    
        self.decoder = Decoder()

        if pretrained:
            print('Loading pretrained Generator model...')
            assert os.path.exists('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.load_state_state(checkpoint['netG'], strict=True)
            print('Done.')
        else:
            self.apply(weights_init)
            print('Weights of Generator model are initialized.')
    
    def forward(self, input):
        mid_input = self.encoder(input)
        output = self.decoder(mid_input)
        return output
    

class Discriminator(nn.Module):
    def __init__(self, input_channels, pretrained=False):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.GroupNorm(32, 64)
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.GroupNorm(32,128)
        self.conv3 = torch.nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.GroupNorm(32, 256)
        self.conv4 = torch.nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1))
        self.bn4 = nn.GroupNorm(32, 512)
        self.conv5 = torch.nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))         
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        if pretrained:
            print('Loading pretrained Discriminator model...')
            assert os.path.exists('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.load_state_state(checkpoint['netD'], strict=True)
            print('Done.')
        else:
            self.apply(weights_init)
            print('Weights of Discriminator model are initialized.')

    def forward(self, base, unknown):
        input = torch.cat((base, unknown), dim=1)
        x = self.activation(self.conv1(input))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x.mean((2,3))
    

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('Conv2d_WS') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('GroupNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    else:
        pass
    

