import torch
import torch.nn as nn

class GPTMini(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos = nn.Embedding(config.block_size, config.n_embd)

        layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layer
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.block_size = config.block_size

    def forward(self, x):
        B, T = x.shape

        tok = self.token(x)
        pos = self.pos(torch.arange(T, device=x.device))

        x = tok + pos
        x = self.transformer(x)
        return self.lm_head(x)