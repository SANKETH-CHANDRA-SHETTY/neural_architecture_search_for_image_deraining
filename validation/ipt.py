class ipth(nn.Module):
    def __init__(self, num_queries, patch_dim, num_heads, num_layers, dropout_rate):
        super(ipth, self).__init__()

        self.isOk = True
        self.model = None
        self.patch_size = 8  

        is_valid, num_queries, patch_dim, num_heads, num_layers, dropout_rate = self.validate_and_adjust_params(
            num_queries, patch_dim, num_heads, num_layers, dropout_rate
        )

        if not is_valid:
            self.isOk = False
        else:
            try:
                self.model = ipt(num_queries, patch_dim, num_heads, num_layers, dropout_rate)  # Fixed: Sending only 5 arguments
            except Exception as e:
                self.isOk = False

    def validate_and_adjust_params(self, num_queries, patch_dim, num_heads, num_layers, dropout_rate):
        if not (1 <= num_queries <= 128) or not (1 <= patch_dim <= 8) or not (1 <= num_heads <= 8) or not (1 <= num_layers <= 12) or not (0.0 <= dropout_rate <= 1):
            return False, None, None, None, None, None

        valid_patch_dims = [1, 2, 4, 8]  
        
        if patch_dim not in valid_patch_dims:
            closest_dim = None
            min_diff = float('inf') 

            for dim in valid_patch_dims:
                diff = abs(dim - patch_dim) 
                if diff < min_diff:  
                    min_diff = diff
                    closest_dim = dim  

            patch_dim = closest_dim  

        return True, num_queries, patch_dim, num_heads, num_layers, dropout_rate  # Removed 'dim' from return

    def forward(self, x):
        if self.isOk and self.model is not None:
            return self.model(x)
        else:
            raise ValueError("Invalid parameters: Model was not initialized")
