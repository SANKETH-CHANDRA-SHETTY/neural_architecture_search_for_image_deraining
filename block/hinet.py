class HINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(HINBlock, self).__init__()
        # Initialize layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.instance_norm = nn.InstanceNorm2d((out_channels + 1) // 2, affine=True)  # Handle odd channels

        # Define activation function based on parameter
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation type. Use 'leaky_relu' or 'relu'.")

    def forward(self, x):
        out = self.conv1(x)
        # Split the output unevenly if out_channels is odd
        mid1, mid2 = torch.split(out, [(out.size(1) + 1) // 2, out.size(1) // 2], dim=1)
        mid1 = self.instance_norm(mid1)
        out = torch.cat([mid1, mid2], dim=1)  # Concatenate after normalization
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class HINet(nn.Module):
    def __init__(self,inp_channels, num_blocks=1, activation='leaky_relu'):

        super(HINet, self).__init__()

        self.num_blocks = num_blocks
        self.activation = activation

        # Create a sequence of HINBlocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(HINBlock(inp_channels, inp_channels, activation=activation))  # Fixed in/out channels for simplicity
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out