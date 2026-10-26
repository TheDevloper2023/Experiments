from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_len)
        return {
            'input_ids': torch.tensor(enc['input_ids']),
            'attention_mask': torch.tensor(enc['attention_mask']),
            'labels': self.labels[i],
        }

def collate(batch):
    ids = torch.nn.utils.rnn.pad_sequence(
        [b['input_ids'] for b in batch], batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(
        [b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b['labels'] for b in batch])
    return ids, masks, labels