import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class GPTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size=400, context=312, n_heads=8, n_layers=8):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.positions = nn.Embedding(context, embed_size)
        self.blocks = nn.Sequential(*[self._block(embed_size, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_size)
        self.classifier = nn.Linear(embed_size, 2)  

    def _block(self, embed_size, n_heads):
        head_size = embed_size // n_heads
        return nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_heads),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
        )

    def forward(self, inp):
        BS, SL = inp.shape
        emb = self.embeddings(inp)
        pos = self.positions(torch.arange(SL, device=device))
        x = emb + pos
        for block in self.blocks:
            attn_out, _ = block[1](x, x, x)
            x = x + attn_out
            x = block[3](block[2](x))
            x = block[5](x)
        x = self.ln(x)
        logits = self.classifier(x[:, -1, :])  
        return logits
