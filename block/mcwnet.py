class DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_in // 2, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=channel_in + channel_in // 2, out_channels=channel_in // 2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in + channel_in // 2 + channel_in // 2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x

        # First conv
        out1 = self.relu1(self.conv_1(x))  # [B, channel_in // 2, H, W]
        conc1 = torch.cat([x, out1], dim=1)  # [B, channel_in + channel_in // 2, H, W]

        # Second conv
        out2 = self.relu2(self.conv_2(conc1))  # [B, channel_in // 2, H, W]
        conc2 = torch.cat([conc1, out2], dim=1)  # [B, channel_in + channel_in // 2 + channel_in // 2, H, W]

        # Third conv
        out3 = self.relu3(self.conv_3(conc2))  # [B, channel_in, H, W]

        # Residual connection
        out = out3 + residual

        return out


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=False):

        super(NonLocalBlock2D, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0) # padding = 0

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0), # padding = 0
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0) # padding = 0
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0) # padding = 0
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0) # padding = 0

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class RegionNONLocalBlock(nn.Module):
    def __init__(self, in_channels, sub_sample=True, bn_layer=False, grid=[8, 8]):
        super(RegionNONLocalBlock, self).__init__()

        self.non_local_block = NonLocalBlock2D(
            in_channels=in_channels,
            sub_sample=sub_sample,
            bn_layer=bn_layer
        )
        self.grid = grid

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Split along height into grid rows
        input_row_list = x.chunk(self.grid[0], dim=2)

        output_row_list = []
        for row in input_row_list:
            # Split each row into grid columns
            input_grid_list_of_a_row = row.chunk(self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            # Apply NonLocalBlock2D on each grid
            for grid in input_grid_list_of_a_row:
                processed_grid = self.non_local_block(grid)
                output_grid_list_of_a_row.append(processed_grid)

            # Concatenate along width for each row
            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        # Concatenate along height
        output = torch.cat(output_row_list, dim=2)
        return output

class MCWNet(nn.Module):
    def __init__(self,inp_channels,sub_sample, bn_layer):
        super(MCWNet, self).__init__()
        ############################################# Encoder #############################################
          # Maintain the same number of channels as input
        reduction = 16

        # Level 1
        self.conv_i = nn.Conv2d(in_channels=inp_channels, out_channels=inp_channels, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.DCR_block11 = self.make_layer(DCR_block, inp_channels)
        self.DCR_block12 = self.make_layer(DCR_block, inp_channels)
        self.NonLocalBlock2D12 = RegionNONLocalBlock(inp_channels,sub_sample, bn_layer,grid=[16,4])

    def make_layer(self, block, channel_in, inter_channels=None):
        layers = []
        if isinstance(block, list):
            for i in range(len(block)):
                b = block[i]
                c = channel_in[i]
                layers.append(b(c))
        else:  # isinstance(block, nn.Module):
            layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        ############################################# Encoder #############################################

        # Level 1
        out = self.relu1(self.conv_i(x))  # Adjusts the input size to [1, 3, 256, 256]
        out = self.DCR_block11(out)
        out = self.DCR_block12(out)
        out1 = self.NonLocalBlock2D12(out)  # Channels stay 3 throughout

        return out1

