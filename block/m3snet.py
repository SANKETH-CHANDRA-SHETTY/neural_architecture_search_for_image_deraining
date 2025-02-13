class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=6, FFN_Expand=6, drop_out_rate=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0)

        # Dropout layers
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.chunk(2, dim=1)  # SimpleGate operation
        x = x[0] * x[1]  # Element-wise multiplication
        x = self.conv3(x)
        x = self.dropout1(x)
        return x

class M3SNet(nn.Module):
    def __init__(self, inp_channels, DW_Expand,FFN_Expand,drop_out_rate):
        super().__init__()
        # Intro layer to project input channels
        self.intro = nn.Conv2d(in_channels=inp_channels, out_channels=inp_channels, kernel_size=3, padding=1)
        # One encoder block (NAFBlock)
        self.encoder = NAFBlock(inp_channels,DW_Expand,FFN_Expand,drop_out_rate)

    def forward(self, inp):
        x = self.intro(inp)  # Initial projection
        #print(f"After Intro Layer: {x.shape}")    # Debugging output

        x = self.encoder(x)  # Single encoder block
        #print(f"After Encoder Block: {x.shape}")  # Debugging output

        return x