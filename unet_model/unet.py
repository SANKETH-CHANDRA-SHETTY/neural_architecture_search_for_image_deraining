device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UNet(nn.Module):
    def __init__(self, encoded_list, n_channels, out_channels):
        super(UNet, self).__init__()
        self.encoding = encoded_list
        self.isOk = True
        self.model = None
        self.psnr = None
        self.normalized_psnr = None
        self.flops = None
        self.normalized_flops = None
        self.fitness_score = None

        if len(encoded_list) != 24:
            self.isOk = False
            return

        self.block_list = encoded_list[:4]
        self.param_list = encoded_list[4:]
        block_params = [self.param_list[i:i + 5] for i in range(0, len(self.param_list), 5)]

        blocks = {
            1: Restormerh,
            2: M3SNeth,
            3: NAFNeth,
            4: HINeth,
            5: MCWNeth
        }
        base_channels = n_channels

        # Encoder
        self.en1 = blocks[self.block_list[0]](base_channels, *block_params[0])
        self.down1 = nn.MaxPool2d(2)
        self.down_conv1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)

        self.en2 = blocks[self.block_list[1]](base_channels * 2, *block_params[1])
        self.down2 = nn.MaxPool2d(2)
        self.down_conv2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)

        self.en3 = blocks[self.block_list[2]](base_channels * 4, *block_params[2])
        self.down3 = nn.MaxPool2d(2)
        self.down_conv3 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)

        self.mid = blocks[self.block_list[3]](base_channels * 8, *block_params[3])

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv3 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.de3 = blocks[self.block_list[2]](base_channels * 4, *block_params[2])

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.de2 = blocks[self.block_list[1]](base_channels * 2, *block_params[1])

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.de1 = blocks[self.block_list[0]](base_channels, *block_params[0])

        self.conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self.isOk = all([self.en1.isOk, self.en2.isOk, self.en3.isOk, self.mid.isOk,
                         self.de1.isOk, self.de2.isOk, self.de3.isOk])

    def forward(self, x):
        if not self.isOk:
            return None

        try:
            x1 = self.en1(x)
            if not self.en1.isOk: return None

            x1_down = self.down_conv1(self.down1(x1))
            x2 = self.en2(x1_down)
            if not self.en2.isOk: return None

            x2_down = self.down_conv2(self.down2(x2))
            x3 = self.en3(x2_down)
            if not self.en3.isOk: return None

            x3_down = self.down_conv3(self.down3(x3))
            mid = self.mid(x3_down)
            if not self.mid.isOk: return None

            x3_up = self.up_conv3(self.up3(mid)) + x3
            x3_dec = self.de3(x3_up)
            if not self.de3.isOk: return None

            x2_up = self.up_conv2(self.up2(x3_dec)) + x2
            x2_dec = self.de2(x2_up)
            if not self.de2.isOk: return None

            x1_up = self.up_conv1(self.up1(x2_dec)) + x1
            x1_dec = self.de1(x1_up)
            if not self.de1.isOk: return None

            return self.conv(x1_dec)
        except Exception as e:
            return None
