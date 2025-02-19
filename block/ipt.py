import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Common layers and utilities
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)

# Vision Transformer class (simplified)
class VisionTransformer(nn.Module):
    def __init__(self, img_dim, patch_dim, num_channels, embedding_dim, num_heads, num_layers, hidden_dim, num_queries, dropout_rate=0, no_norm=False, mlp=False, pos_every=False, no_pos=False):
        super(VisionTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = (img_dim // patch_dim) * (img_dim // patch_dim)  # Total number of patches
        self.flatten_dim = patch_dim * patch_dim * num_channels  # Each patch will have this many values
        self.out_dim = patch_dim * patch_dim * num_channels
        self.no_pos = no_pos
        self.mlp = mlp

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
            nn.Dropout(dropout_rate)
        )

        self.query_embed = nn.Embedding(num_queries, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding = nn.Embedding(self.num_patches, embedding_dim)  # Positional encoding with matching size

    def forward(self, x, query_idx=None):
        B, C, H, W = x.size()
        # Reshape into patches
        x = x.view(B, C, H // self.patch_dim, self.patch_dim, W // self.patch_dim, self.patch_dim)
        # Reorder the dimensions and flatten the patches
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Flatten each patch
        self.num_patches = (H // self.patch_dim) * (W // self.patch_dim)
        x = x.view(B, C, H // self.patch_dim, self.patch_dim, W // self.patch_dim, self.patch_dim)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, self.num_patches, self.flatten_dim)

        # Linear projection of flattened patches to embedding_dim
        x = self.linear_encoding(x)

        # Add positional encoding
        if self.pos_every and not self.no_pos:
            if query_idx is None:
                query_idx = torch.arange(self.num_patches, device=x.device)

            if isinstance(query_idx, int):
                query_idx = torch.tensor(query_idx, dtype=torch.long, device=x.device)
            elif not isinstance(query_idx, torch.LongTensor):
                query_idx = query_idx.long()

            pos = self.position_encoding(query_idx).unsqueeze(0).expand(x.size(0), x.size(1), -1)
            x = x + pos

        # Pass through the transformer encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x, x)

        return x



# Main model definition (from the provided code)
class ipt(nn.Module):
    def __init__(self,num_queries,patch_dim,num_heads, num_layers, dropout_rate, conv=default_conv):
        super(ipt, self).__init__()

        self.scale_idx = 0

        # Define default args
        self.args = {
            'n_feats': 32,           # Reduced feature size
            'n_colors': 3,           # RGB channels remain the same
            'rgb_range': 255,        # Color range stays as it is
            'scale': [2],            # Scaling remains the same
            'patch_size': 8,         # Smaller patch size
            'patch_dim': patch_dim,  # Patch dimension from input
            'num_heads': num_heads,  # Number of attention heads from input
            'num_layers': num_layers,# Number of layers from input
            'num_queries': num_queries, # Number of queries from input
            'dropout_rate': dropout_rate, # Dropout rate from input
            'no_mlp': False,         # MLP usage remains unchanged
            'pos_every': True,       # Keep positional encoding
            'no_pos': False,         # Keep positional encoding enabled
            'no_norm': False         # Keep normalization enabled
        }

        n_feats = self.args['n_feats']
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(self.args['rgb_range'])
        self.add_mean = MeanShift(self.args['rgb_range'], sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(self.args['n_colors'], n_feats, kernel_size),
                ResBlock(conv, n_feats, 5, act=act),
                ResBlock(conv, n_feats, 5, act=act)
            ) for _ in self.args['scale']
        ])

        self.body = VisionTransformer(
            img_dim=self.args['patch_size'],
            patch_dim=self.args['patch_dim'],
            num_channels=n_feats,
            embedding_dim=n_feats * self.args['patch_dim'] * self.args['patch_dim'],
            num_heads=self.args['num_heads'],
            num_layers=self.args['num_layers'],
            hidden_dim=n_feats * self.args['patch_dim'] * self.args['patch_dim'] * 4,
            num_queries=self.args['num_queries'],
            dropout_rate=self.args['dropout_rate'],
            mlp=self.args['no_mlp'],
            pos_every=self.args['pos_every'],
            no_pos=self.args['no_pos']
        )

        self.tail = nn.ModuleList([
            nn.Sequential(
                Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, self.args['n_colors'], kernel_size)
            ) for s in self.args['scale']
        ])

    def forward(self, x):
        original_size = x.size()

        x = self.sub_mean(x)
        x = self.head[self.scale_idx](x)

        res = self.body(x, self.scale_idx)

        B, _, H, W = x.shape
        res = res.view(B, x.shape[1], H, W)
        res += x

        x = self.tail[self.scale_idx](res)
        x = self.add_mean(x)

        if x.size() != original_size:
            x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=False)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
