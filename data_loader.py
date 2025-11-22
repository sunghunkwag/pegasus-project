import torch
from torch.utils.data import DataLoader, Dataset
import random

class SimpleTokenizer:
    def __init__(self):
        # A very simple char-level tokenizer + a few words
        self.chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'"
        self.vocab = {char: i for i, char in enumerate(self.chars)}
        self.idx_to_char = {i: char for char, i in self.vocab.items()}
        self.vocab_size = len(self.chars)
        self.pad_token_id = 0 # Space as pad for simplicity here

    def encode(self, text):
        return [self.vocab.get(c, 0) for c in text]

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return "".join([self.idx_to_char.get(t, '') for t in tokens])

class MockPhilosophyDataset(Dataset):
    def __init__(self, seq_len, size=100):
        self.seq_len = seq_len
        self.size = size
        self.tokenizer = SimpleTokenizer()
        # Some dummy data
        self.data = [
            "The unexamined life is not worth living.",
            "I think, therefore I am.",
            "God is dead.",
            "Man is born free, and everywhere he is in chains.",
            "To be is to be perceived."
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return random sample from data
        text = random.choice(self.data)
        tokens = self.tokenizer.encode(text)

        # Pad or truncate
        if len(tokens) < self.seq_len:
            tokens = tokens + [0] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[:self.seq_len]

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(0) # Dummy label

def load_philosophy_data(seq_len=32, batch_size=32):
    dataset = MockPhilosophyDataset(seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    tokenizer = dataset.tokenizer
    return loader, tokenizer
