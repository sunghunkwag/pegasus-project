import torch
import torch.nn as nn
import torch.optim as optim
import requests
import os
import time
import torch.nn.functional as F
# --- 1. Model (HRM) ---
class InputNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    def forward(self, x):
        return self.embedding(x)
class OutputNetwork(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        return self.fc(x)
class LowLevelModule(nn.Module):
    def __init__(self, hidden_dim, h_dim):
        super().__init__()
        self.cell = nn.GRUCell(hidden_dim + h_dim, hidden_dim)
    def forward(self, x, h_prev, h_high):
        combined_input = torch.cat([x, h_high], dim=-1)
        return self.cell(combined_input, h_prev)
class HighLevelModule(nn.Module):
    def __init__(self, hidden_dim, l_dim):
        super().__init__()
        self.cell = nn.GRUCell(l_dim, hidden_dim)
    def forward(self, l_summary, h_prev):
        return self.cell(l_summary, h_prev)
class HRM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, T=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.T = T
        self.input_net = InputNetwork(vocab_size, hidden_dim)
        self.l_module = LowLevelModule(hidden_dim, hidden_dim)
        self.h_module = HighLevelModule(hidden_dim, hidden_dim)
        self.output_net = OutputNetwork(hidden_dim, vocab_size)

    def forward(self, x, h_l_init=None, h_h_init=None):
        batch_size, seq_len = x.size()
        device = x.device
        x_emb = self.input_net(x)
        if h_l_init is None: h_l = torch.zeros(batch_size, self.hidden_dim).to(device)
        else: h_l = h_l_init
        if h_h_init is None: h_h = torch.zeros(batch_size, self.hidden_dim).to(device)
        else: h_h = h_h_init
        outputs = []
        pad_len = (self.T - (seq_len % self.T)) % self.T
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, self.hidden_dim).to(device)
            x_emb = torch.cat([x_emb, padding], dim=1)
        total_steps = x_emb.size(1)
        curr_idx = 0
        for n in range(total_steps // self.T):
            for t in range(self.T):
                if curr_idx < total_steps:
                    x_t = x_emb[:, curr_idx, :]
                    h_l = self.l_module(x_t, h_l, h_h)
                    outputs.append(self.output_net(h_l))
                    curr_idx += 1
            h_h = self.h_module(h_l, h_h)
        outputs = torch.stack(outputs, dim=1)
        return outputs[:, :seq_len, :], (h_l, h_h)
# --- 2. Data Loader ---
class CharTokenizer:
    def __init__(self, text_corpus):
        chars = sorted(list(set(text_corpus)))
        self.vocab_size = len(chars) + 1
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}
        self.itos = {i+1: ch for i, ch in enumerate(chars)}
        self.stoi['<PAD>'] = 0
        self.itos[0] = '<PAD>'
    def encode(self, text): return [self.stoi[c] for c in text if c in self.stoi]
    def decode(self, indices): return ''.join([self.itos[i] for i in indices if i in self.itos])
from torch.utils.data import Dataset, DataLoader
class PhilosophyDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=64):
        self.text = text
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
    def __len__(self): return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.seq_len+1]
        return chunk[:-1], chunk[1:]
def load_data():
    url = "https://www.gutenberg.org/cache/epub/1497/pg1497.txt" # Republic
    print("Downloading text...")
    try:
        text = requests.get(url).text
        print(f"Downloaded {len(text)} chars.")
        return text
    except:
        print("Download failed, using mock data.")
        return "Mock philosophy data. " * 1000
# --- 3. Training Loop ---
def train():
    print("Initializing Training...")
    text = load_data()
    tokenizer = CharTokenizer(text)
    dataset = PhilosophyDataset(text, tokenizer, seq_len=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HRM(tokenizer.vocab_size, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Starting Epoch 1...")
    model.train()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs, _ = model(x)
        loss = criterion(outputs.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")
        if i > 50: break # Limit for demo
    print("Training Finished.")
if __name__ == "__main__":
    train()
