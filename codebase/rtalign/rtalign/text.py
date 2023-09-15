import torch
from torch import nn
from torch.nn import functional as F

from transformers import CanineModel

class TextEncoder(nn.Module):
    def __init__(self, pretrained='google/canine-c'):
        # canine-c: pre-trained with autoregressive character loss
        # canine-s: pre-trained with subword masking loss
        super().__init__()
        self.net = CanineModel.from_pretrained(pretrained)  

    def text_bytes(self, text):
        if len(text)==0:
            return torch.empty((1,0), dtype=torch.long)
        return torch.tensor([[ord(char) for char in text]])
    
    def pad(self, tokens):
        pad = 4 - tokens.shape[1]
        if pad > 0:
            tokens = torch.cat((tokens, tokens.new_zeros(tokens.shape[0], pad)), 1)
        return tokens

    def forward(self, text_t):
        return self.net(text_t, output_hidden_states=True)

    def encode(self, text, layer=-1):
        if isinstance(text, str):
            text = self.text_bytes(text)
        return self(text).hidden_states[layer]