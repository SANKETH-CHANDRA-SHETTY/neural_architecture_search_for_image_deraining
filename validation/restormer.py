class Restormerh(nn.Module):
    def __init__(self, inp_channels, dim, heads, num_blocks, ffn_expansion_factor, LayerNorm_type):

        super(Restormerh, self).__init__()

        self.isOk = True
        self.block = None

        is_valid, dim, heads, num_blocks, ffn_expansion_factor, LayerNorm_type = self.validate_and_adjust_params(
            dim, heads, num_blocks, ffn_expansion_factor, LayerNorm_type
        )

        if not is_valid:
            self.isOk = False
        else:
            try:
                self.block = Restormer(inp_channels, dim, num_blocks, heads, ffn_expansion_factor, LayerNorm_type)
            except Exception as e:
                self.isOk = False

    def validate_and_adjust_params(self, dim, heads, num_blocks, ffn_expansion_factor, LayerNorm_type):
        if not (4 <= dim <= 128):
            return False, None, None, None, None, None

        if not (1 <= heads <= 8):
            return False, None, None, None, None, None

        if not (2 <= num_blocks <= 6):
            return False, None, None, None, None, None

        if not (2 <= ffn_expansion_factor <= 6):
            return False, None, None, None, None, None

        if LayerNorm_type not in [0, 1]:
            return False, None, None, None, None, None

        LayerNorm_type = 'BiasFree' if LayerNorm_type == 0 else 'WithBias'

        if dim % heads != 0:
          new_dim = (dim // heads) * heads
          dim = max(new_dim, heads)

        return True, dim, heads, num_blocks, ffn_expansion_factor, LayerNorm_type

    def forward(self, x):
        return self.block(x)