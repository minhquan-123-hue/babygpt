"""Small GPT-style causal language model."""

from pathlib import Path
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = Path(__file__).parent / "data" / "data.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_vocab():
    text = DATA_PATH.read_text(encoding="utf-8").lower()
    words = re.findall(r"\S+", text)
    vocab = ["<unk>", "<bos>", "<eos>"] + sorted(set(words))
    return vocab, {w: i for i, w in enumerate(vocab)}

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, max_seq=64, dim=64, heads=4, layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq = max_seq

    def forward(self, ids):
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

class TextGenerator:
    def __init__(self):
        self.vocab, self.stoi = build_vocab()
        self.itos = {i: w for i, w in enumerate(self.vocab)}
        self.model = TinyGPT(len(self.vocab)).to(DEVICE)
        self._train()

    def _train(self):
        text = DATA_PATH.read_text(encoding="utf-8").lower()
        tokens = [self.stoi.get(w, 0) for w in re.findall(r"\S+", text)]
        if len(tokens) < 3:
            return

        x = torch.tensor(tokens[:-1], dtype=torch.long, device=DEVICE)[None, :]
        y = torch.tensor(tokens[1:], dtype=torch.long, device=DEVICE)[None, :]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)

        self.model.train()
        for _ in range(120):
            logits = self.model(x)
            loss = F.cross_entropy(logits.reshape(-1, len(self.vocab)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_words=200):
        words = re.findall(r"\S+", prompt.lower())
        ids = [self.stoi.get(w, 0) for w in words]
        if not ids:
            ids = [self.stoi["<bos>"]]

        generated = ids[:]
        for _ in range(min(max_words, 200)):
            context = generated[-self.model.max_seq:]
            x = torch.tensor(context, dtype=torch.long, device=DEVICE)[None, :]
            logits = self.model(x)[0, -1]
            probs = F.softmax(logits / 0.9, dim=-1)
            generated.append(torch.multinomial(probs, 1).item())

        output_words = [
            self.itos[i] for i in generated
            if i not in (self.stoi["<bos>"], self.stoi["<eos>"])
        ][:max_words]

        return {
            "text": " ".join(output_words),
            "tokens": generated[:max_words],
            "vocab_size": len(self.vocab),
            "device": DEVICE,
        }
