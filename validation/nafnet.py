class NAFNeth(nn.Module):
    def __init__(self, inp_channels, dummy1, dummy2, DW_Expand, FFN_Expand, drop_out_rate):

        super(NAFNeth, self).__init__()

        self.isOk = True
        self.block = None

        is_valid, dummy1, dummy2, DW_Expand, FFN_Expand, drop_out_rate = self.validate_and_adjust_params(dummy1, dummy2, DW_Expand, FFN_Expand, drop_out_rate)

        if not is_valid:
            self.isOk = False
        else:
            try:
                self.block = NAFNet(inp_channels, DW_Expand,FFN_Expand, drop_out_rate)
            except Exception as e:
                self.isOk = False

    def validate_and_adjust_params(self,dummy1, dummy2, DW_Expand, FFN_Expand, drop_out_rate):
      if not(2 <= DW_Expand <= 6) or not(2 <= FFN_Expand <= 6) or not(0 <= drop_out_rate <= 1):
        return False, None, None, None, None, None

      return True, dummy1, dummy2, DW_Expand, FFN_Expand, drop_out_rate

    def forward(self, x):
        return self.block(x)
