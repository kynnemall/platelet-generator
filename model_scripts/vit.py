import torch
import torchvision
from torch import nn
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput


class TransformerEncoder(BaseEncoder):
    def __init__(self, args=None):
        img_width = 160
        img_channels = 1
        patch_size = 32
        d_model = 64
        num_heads = 4
        num_layers = 3
        num_latents = 100
        ff_dim = 1024
        # taken from tutorial at https://comsci.blog/posts/vit
        BaseEncoder.__init__(self)
        self.name = 'ViT_v1'

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(img_channels * patch_size * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # + 1 because we add cls tokens
        self.position_embedding = nn.Parameter(
            torch.rand(1, (img_width // patch_size) * (img_width // patch_size) + 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_latents)

    def forward(self, x):
        N, C, H, W = x.shape

        # we divide the image into patches then flatten and embed them
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(N, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, C * self.patch_size * self.patch_size)
        x = self.patch_embedding(x)

        # cls tokens concatenated after repeating it for the batch
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.transformer_encoder(x)

        # only taking the transformed output of the cls token
        x = x[:, 0]

        # compress to latent space
        x = self.fc(x)
        output = ModelOutput(reconstruction=x)
        return output