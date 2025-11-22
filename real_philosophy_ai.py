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
    def __init__(self, vocab_size, hidden_dim, skill_dim, T=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.skill_dim = skill_dim
        self.T = T

        self.input_net = InputNetwork(vocab_size, hidden_dim)
        self.l_module = LowLevelModule(hidden_dim, hidden_dim)
        self.h_module = HighLevelModule(hidden_dim, hidden_dim)
        self.output_net = OutputNetwork(hidden_dim, vocab_size)
        self.z_proj = nn.Linear(skill_dim, hidden_dim)

    def forward(self, x, z_vec, h_l_init=None, h_h_init=None):
        # x: (batch, seq_len)
        # z_vec: (batch, skill_dim)
        batch_size, seq_len = x.size()
        device = x.device

        x_emb = self.input_net(x)

        if h_l_init is None: h_l = torch.zeros(batch_size, self.hidden_dim).to(device)
        else: h_l = h_l_init

        if h_h_init is None: h_h = torch.zeros(batch_size, self.hidden_dim).to(device)
        else: h_h = h_h_init

        # Modulate Initial High-Level State with Skill
        z_proj = self.z_proj(z_vec)
        h_h = h_h + z_proj

        outputs = []

        # Padding logic to ensure T steps
        pad_len = (self.T - (seq_len % self.T)) % self.T
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, self.hidden_dim).to(device)
            x_emb = torch.cat([x_emb, padding], dim=1)

        total_steps = x_emb.size(1)
        curr_idx = 0

        for n in range(total_steps // self.T):
            # Update High Level at macro steps
            # Recurrent update from previous h_h and last h_l
            if n > 0:
                h_h = self.h_module(h_l, h_h)
                # Re-inject skill modulation at every high-level update to maintain persona
                h_h = h_h + z_proj

            for t in range(self.T):
                if curr_idx < total_steps:
                    x_t = x_emb[:, curr_idx, :]
                    h_l = self.l_module(x_t, h_l, h_h)
                    outputs.append(self.output_net(h_l))
                    curr_idx += 1

        outputs = torch.stack(outputs, dim=1)
        return outputs[:, :seq_len, :], (h_l, h_h)

# --- 2. Discriminator ---
class SkillDiscriminator(nn.Module):
    def __init__(self, vocab_size, hidden_dim, skill_dim):
        super().__init__()
        # Linear embedding for soft tokens (Gumbel Softmax support)
        self.embed_linear = nn.Linear(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, skill_dim)

    def forward(self, x_soft):
        # x_soft: (batch, seq_len, vocab_size)
        embedded = self.embed_linear(x_soft)
        _, (h_n, _) = self.lstm(embedded)
        logits = self.classifier(h_n.squeeze(0))
        return logits

# --- 3. Data Loader ---
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
        # Basic cleanup
        start = text.find("THE REPUBLIC")
        end = text.find("End of Project Gutenberg")
        if start != -1 and end != -1:
            text = text[start:end]
        print(f"Downloaded {len(text)} chars.")
        return text
    except:
        print("Download failed, using mock data.")
        return "Mock philosophy data. " * 1000

# --- 4. Training Loop (DIAYN) ---
def train():
    print("Initializing Unsupervised Meta-RL (DIAYN) Training...")

    # Hyperparameters
    SEQ_LEN = 64
    BATCH_SIZE = 32
    HIDDEN_DIM = 128
    SKILL_DIM = 4
    EPOCHS = 3 # Train for longer

    text = load_data()
    tokenizer = CharTokenizer(text)
    dataset = PhilosophyDataset(text, tokenizer, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hrm = HRM(tokenizer.vocab_size, HIDDEN_DIM, skill_dim=SKILL_DIM).to(device)
    discriminator = SkillDiscriminator(tokenizer.vocab_size, HIDDEN_DIM, SKILL_DIM).to(device)

    optimizer = optim.Adam(list(hrm.parameters()) + list(discriminator.parameters()), lr=0.001)
    criterion_lm = nn.CrossEntropyLoss()
    criterion_d = nn.CrossEntropyLoss()

    print("Starting Training (Maximizing Mutual Information)...")

    step_count = 0
    total_steps_limit = 1000 # "As much as possible" within session limits

    hrm.train()
    discriminator.train()

    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            # 1. Sample Skill Z
            z_idx = torch.randint(0, SKILL_DIM, (batch_size,)).to(device)
            z_vec = F.one_hot(z_idx, num_classes=SKILL_DIM).float().to(device)

            # 2. Generator Forward (Modulated by Z)
            logits, _ = hrm(x, z_vec) # (batch, seq, vocab)

            # 3. Gumbel Softmax (Differentiable discrete approximation)
            # Allows gradients to flow from Discriminator to Generator
            soft_tokens = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)

            # 4. Discriminator Forward
            d_logits = discriminator(soft_tokens)

            # 5. Losses
            # LM Loss: Maintain ability to speak English
            lm_loss = criterion_lm(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))

            # Discriminator Loss: D tries to identify Z
            d_loss = criterion_d(d_logits, z_idx)

            # Total Loss
            # G minimizes LM loss AND minimizes D's error (Cooperative/Mutual Information maximization)
            # Note: We are maximizing MI(Z; Text). Maximizing D's accuracy maximizes the lower bound on MI.
            # So G wants D to succeed.
            total_loss = lm_loss + d_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            step_count += 1

            if step_count % 50 == 0:
                print(f"Step {step_count} | Total Loss: {total_loss.item():.4f} | LM: {lm_loss.item():.4f} | Disc: {d_loss.item():.4f}")

            if step_count >= total_steps_limit:
                print("Reached step limit.")
                return

    print("Training Finished.")

if __name__ == "__main__":
    train()
