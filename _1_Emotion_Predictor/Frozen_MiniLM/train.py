import torch
import torch.nn as nn
import torch.nn.functional as F

# load model
from _1_Emotion_Predictor.Frozen_MiniLM.model import EmotionClassifierV1, EmotionClassifierV2, EmotionClassifierHybrid
from torch.utils.data import TensorDataset, DataLoader, random_split
import _1_Emotion_Predictor.Frozen_MiniLM.hparam as hp
from _1_Emotion_Predictor.Frozen_MiniLM.loss_fn import BinaryFocalLoss, BCEFocalMixed

device = "cuda"


## Load dataset
blob = torch.load('minilm_eqn.pt')
X, Y = blob['X'], blob['Y']          # (N, 384), (N, 28)
EMOTIONS = blob['emotions']          

print(X.shape, Y.shape, X.dtype)     # torch.Size([N, 384]) ... float32

dataset = TensorDataset(X, Y)

# make val and train dataloader(s)
n = len(dataset)
n_val = max(1, int(0.1 * n))           # 10% val
n_train = n - n_val
gen = torch.Generator().manual_seed(42)  
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

# make dataloaders
train_dataloader =  DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
valid_dataloader =  DataLoader(val_ds, batch_size=256, shuffle=False,  num_workers=0)


## Load Model
#model = EmotionClassifierV1(hp.NUM_EMOTIONS, hp.inputc_dim)
#model = EmotionClassifierV2(hp.NUM_EMOTIONS, hp.input_dim, hidden=hp.hdn_dims, dropout=0.25)
model = EmotionClassifierHybrid(hp.NUM_EMOTIONS, hp.input_dim, hidden=hp.hdn_dims, dropout=0.25)
model = model.to(device)

#citration = nn.MSELoss() # TODO
#citration = nn.BCEWithLogitsLoss()
citration = BinaryFocalLoss(gamma=1.0)

## load optimizer
optimizer = torch.optim.AdamW(
    model.parameters() ,lr=hp.lr
)

########### Main training loop ####################
best_val = float('inf')
for i in range(hp.epochs):
    print("Train Epoch: ", i)

    # train stuf
    model.train()
    train_loss = 0.0

    for step, (xb, yb) in enumerate(train_dataloader):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)
        loss = citration(pred, yb)
        loss.backward()
        optimizer.step()


        # logging
        if step % 50 == 0:
            print(f"  step {step:4d} | loss: {loss.item():.4f}")

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_ds)

    print("Valid Epoch: ", i)
    # valid stuf
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): 
        for xb, yb in valid_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            
            pred = model(xb)
            val_loss += citration(pred, yb).item() * xb.size(0)

    val_loss /= len(val_ds)

    print(f'epoch {i:3d}  train {train_loss:.4f}  val {val_loss:.4f}')

    # save checkpoint
    if val_loss < best_val:
        print("Saving best val loss!")
        best_val = val_loss
        torch.save(model.state_dict(), 'emotion_head_best.pt')
        



