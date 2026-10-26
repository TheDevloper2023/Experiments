import torch
import torch.nn as nn
import torch.nn.functional as F

# load model
from _1_Emotion_Predictor.Finetuned_MiniLM.model import EmotionClassifierHybrid
from torch.utils.data import TensorDataset, DataLoader, random_split
from _1_Emotion_Predictor.Finetuned_MiniLM.dataset import TextDataset, collate
import _1_Emotion_Predictor.Finetuned_MiniLM.hparam as hp
from _1_Emotion_Predictor.Finetuned_MiniLM.loss_fn import BinaryFocalLoss, compute_cooccurrence_matrix, cooccurrence_penalty
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(hp.MODEL)

device = "cuda"


## Load dataset
blob = torch.load('minilm_eqn.pt')
#X, Y = blob['X'], blob['Y']          # (N, 384), (N, 28)
X, Y = blob['texts'], blob['Y'] 
EMOTIONS = blob['emotions']          

#print(X.shape, Y.shape, X.dtype)     # torch.Size([N, 384]) ... float32

dataset = TextDataset(X, Y, tokenizer, hp.MAX_LEN) #TensorDataset(X, Y)

# make val and train dataloader(s)
n = len(dataset)
n_val = max(1, int(0.1 * n))           # 10% val
n_train = n - n_val
gen = torch.Generator().manual_seed(42)  
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

train_labels = Y[train_ds.indices].to(device)
cooc_matrix = compute_cooccurrence_matrix(train_labels)
print(f'Co-occurrence matrix: {cooc_matrix.shape}, '
      f'range [{cooc_matrix[cooc_matrix > 0].min():.3f}, {cooc_matrix.max():.3f}]')

# make dataloaders
train_dataloader =  DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, collate_fn=collate)
valid_dataloader =  DataLoader(val_ds, batch_size=32, shuffle=False,  num_workers=0, collate_fn=collate)


## Load Model
#model = EmotionClassifierV1(hp.NUM_EMOTIONS, hp.inputc_dim)
#model = EmotionClassifierV2(hp.NUM_EMOTIONS, hp.input_dim, hidden=hp.hdn_dims, dropout=0.25)
model = EmotionClassifierHybrid(hp.NUM_EMOTIONS, hp.input_dim, dropout=0.25,minilm=hp.MODEL,n_keep=3)
model = model.to(device)

#citration = nn.MSELoss() # TODO
#citration = nn.BCEWithLogitsLoss()
citration = BinaryFocalLoss(gamma=1.0)

## load optimizer

#optimizer = torch.optim.AdamW(
#    model.parameters() ,lr=hp.lr
#)


enc_params  = [p for p in model.encoder.parameters() if p.requires_grad]
head_params = list(model.linear.parameters()) + list(model.trunk.parameters())
mlp_params  = list(model.mlp_head.parameters())

optimizer = torch.optim.AdamW([
    {'params': enc_params,  'lr': 1.5e-5, 'weight_decay': 0.01},
    {'params': head_params, 'lr': 3e-4,   'weight_decay': 1e-3},
    {'params': mlp_params,  'lr': 1e-3,   'weight_decay': 1e-3},
])

########### Main training loop ####################
lambda_cooc = 0.1   # ← new hyperparameter, start here
best_val = float('inf')
for i in range(hp.epochs):
    print("Train Epoch: ", i)

    # train stuf
    model.train()
    train_loss = 0.0

    for step, (ids, masks, yb) in enumerate(train_dataloader):
        ids, masks, yb = ids.to(device), masks.to(device), yb.to(device)

        optimizer.zero_grad()

        pred = model(ids, masks)
        loss_focal = citration(pred, yb)
        loss_cooc = cooccurrence_penalty(pred, cooc_matrix) * lambda_cooc
        

        loss = loss_focal + loss_cooc


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


        # logging
        if step % 50 == 0:
            print(f"  step {step:4d} | loss: {loss.item():.4f} | loss_cooc: {loss_cooc.item():.4f}, loss_focus: {loss_focal:.4f}")

        train_loss += loss.item() * ids.size(0)

    train_loss /= len(train_ds)

    print("Valid Epoch: ", i)
    # valid stuf
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): 
        for ids, masks, yb in valid_dataloader:
            ids, masks, yb = ids.to(device), masks.to(device), yb.to(device)
            
            pred = model(ids, masks)
            val_loss += citration(pred, yb).item() * ids.size(0)

    val_loss /= len(val_ds)

    print(f'epoch {i:3d}  train {train_loss:.4f}  val {val_loss:.4f}')

    # save checkpoint
    if val_loss < best_val:
        print("Saving best val loss!")
        best_val = val_loss
        torch.save(model.state_dict(), f'saved/emotion_head_best_{i}_new_loss.pt')
    else:
            print('✗ val diverged — stopping')
            pass
        



