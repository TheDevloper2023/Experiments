import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

class EmotionClassifierHybrid(nn.Module):
    def __init__(self, n_emotions, input_dim, dropout=0.15, minilm="", n_keep=3):
        super().__init__()
        # MiniLM
        self.encoder = AutoModel.from_pretrained(minilm)

        h = self.encoder.config.hidden_size

        self.freeze_layers(n_keep)

        # Linear path — preserves per-class independence, calibrated
        self.linear = nn.Linear(input_dim, n_emotions)

        # Nonlinear path — adds capacity for shared/rare signal
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mlp_head = nn.Linear(h, n_emotions)

        # CRITICAL: zero-init the MLP head so training starts at fc behavior.
        # At step 0, output == fc(x) exactly. Nonlinearity only "turns on"
        # if it reduces loss.
        nn.init.zeros_(self.mlp_head.weight)
        nn.init.zeros_(self.mlp_head.bias)


    def freeze_layers(self, n_unfreeze):
        total = self.encoder.config.num_hidden_layers
        freeze_until = total - n_unfreeze

        n_frozen, n_trainable = 0, 0

        for name, p in self.encoder.named_parameters():
            p.requires_grad = False
            if name.startswith('encoder.layer.'):
                if int(name.split('.')[2]) >= freeze_until:
                    p.requires_grad = True


    def forward(self, input_ids, attn_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
        m = attn_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)

        return self.linear(pooled) + self.mlp_head(self.trunk(pooled))



# Vibecode the test
if __name__ == "__main__":
    import hparam as hp

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = EmotionClassifierHybrid(28, 384, minilm=MODEL, n_keep=3)
    model.eval()

    # ── 1. Parameter count check ──
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_tr:,} / {n_tot:,}  ({100*n_tr/n_tot:.1f}%)")
    # Expected: ~10-12M / 22M for L6 + head, i.e. ~45-55%

    # ── 2. Which encoder blocks are trainable? ──
    print("\nEncoder block trainability:")
    for i in range(model.encoder.config.num_hidden_layers):
        p = next(model.encoder.parameters())  # just need one per block
        # Find any param in layer i
        for name, param in model.encoder.named_parameters():
            if name.startswith(f"encoder.layer.{i}."):
                print(f"  block {i}: {'TRAIN' if param.requires_grad else 'frozen'}")
                break

    # ── 3. Forward-pass shape check ──
    texts = ["I am so happy!", "I am furious at you."]
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=hp.MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\nForward: input {enc['input_ids'].shape} → output {out.shape}")
    assert out.shape == (2, 28), f"Bad output shape: {out.shape}"

    # ── 4. Gradient-flow check (encoder actually gets gradients?) ──
    model.train()
    out = model(enc["input_ids"], enc["attention_mask"])
    loss = out.sum()
    loss.backward()

    print("\nGradient flow (should be non-None and non-zero for blocks 3-5):")
    for name, p in model.encoder.named_parameters():
        if p.requires_grad and p.grad is not None:
            gnorm = p.grad.norm().item()
            if gnorm > 0:
                print(f"  {name}: grad={gnorm:.4f}")
                break
    for i in range(model.encoder.config.num_hidden_layers):
        for name, p in model.encoder.named_parameters():
            if name.startswith(f"encoder.layer.{i}.") and p.requires_grad:
                if p.grad is not None:
                    gnorm = p.grad.norm().item()
                    print(f"  block {i}: grad_norm={gnorm:.6f}")
                    break

    # ── 5. Frozen blocks should NOT have gradients ──
    print("\nFrozen blocks (should have grad=None):")
    for i in range(model.encoder.config.num_hidden_layers):
        for name, p in model.encoder.named_parameters():
            if name.startswith(f"encoder.layer.{i}.") and not p.requires_grad:
                grad_is_none = p.grad is None
                print(f"  block {i}: grad is None = {grad_is_none}")
                break