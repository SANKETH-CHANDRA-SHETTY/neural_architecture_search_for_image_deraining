class MCWNeth(nn.Module):
    def __init__(self, inp_channels, d1,d2,d3,sub_sample, bn_layer):

        super(MCWNeth, self).__init__()

        self.isOk = True
        self.block = None

        is_valid, d1,d2,d3,sub_sample, bn_layer = self.validate_and_adjust_params(d1,d2,d3,sub_sample, bn_layer)

        if not is_valid:
            self.isOk = False
        else:
            try:
                self.block = MCWNet(inp_channels, sub_sample, bn_layer)
            except Exception as e:
                self.isOk = False

    def validate_and_adjust_params(self,d1,d2,d3,sub_sample, bn_layer):
      if not(0 <= sub_sample <= 1) or not(0 <= bn_layer <= 1):
        return False, None, None, None, None, None

      sub_sample=True if sub_sample==1 else False
      bn_layer=True if bn_layer==1 else False

      return True, d1,d2,d3,sub_sample, bn_layer

    def forward(self, x):
        return self.block(x)
