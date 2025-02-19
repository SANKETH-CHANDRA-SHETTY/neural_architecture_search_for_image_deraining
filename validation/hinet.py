class HINeth(nn.Module):
    def __init__(self, inp_channels, dummy1, dummy2, num_blocks, dummy3, activation):
        super(HINeth, self).__init__()

        self.isOk = True
        self.block = None

        is_valid, dummy1, dummy2, num_blocks, dummy3, activation = self.validate_and_adjust_params(
            dummy1, dummy2, num_blocks, dummy3, activation
        )

        if not is_valid:
            self.isOk = False
        else:
            try:
                self.block = HINet(inp_channels,num_blocks,activation)
            except Exception as e:
                self.isOk = False

    def validate_and_adjust_params(self, dummy1, dummy2, num_blocks, dummy3, activation):
        if not (2 <= num_blocks <= 6) or activation not in [0, 1]:
            return False, None, None, None, None, None

        activation = 'leaky_relu' if activation == 0 else 'relu'

        return True, dummy1, dummy2, num_blocks, dummy3, activation

    def forward(self, x):
            return self.block(x)
