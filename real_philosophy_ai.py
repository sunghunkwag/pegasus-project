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

def download_wikitext():
    print("Downloading WikiText-2...")
    try:
        url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
        text = requests.get(url).text
        print(f"WikiText: {len(text)} chars")
        return text
    except:
        print("WikiText download failed.")
        return ""

def download_arxiv():
    print("Downloading ArXiv CS.AI Abstracts...")
    try:
        import xml.etree.ElementTree as ET
        url = 'http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=500'
        data = requests.get(url).content
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        text = ""
        for entry in root.findall('atom:entry', ns):
            summary = entry.find('atom:summary', ns).text
            text += summary + "\n"
        print(f"ArXiv: {len(text)} chars")
        return text
    except Exception as e:
        print(f"ArXiv download failed: {e}")
        return ""

def load_data():
    # 1. Philosophy (Plato)
    url = "https://www.gutenberg.org/cache/epub/1497/pg1497.txt"
    print("Downloading Plato...")
    try:
        plato_text = requests.get(url).text
        start = plato_text.find("THE REPUBLIC")
        end = plato_text.find("End of Project Gutenberg")
        if start != -1 and end != -1:
            plato_text = plato_text[start:end]
    except:
        plato_text = ""

    # 2. WikiText (General Knowledge)
    wiki_text = download_wikitext()

    # 3. ArXiv (Technical Knowledge)
    arxiv_text = download_arxiv()

    combined_text = plato_text + "\n" + wiki_text + "\n" + arxiv_text
    print(f"Total Combined Corpus: {len(combined_text)} chars.")

    if len(combined_text) < 1000:
        return "Mock data for fallback. " * 1000

    return combined_text

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
    # total_steps_limit = 100000 # REMOVED: Infinite Training

    # --- Recursive Improvement: Load Checkpoint ---
    if os.path.exists("hrm_checkpoint.pth"):
        print("Loading existing HRM checkpoint for recursive improvement...")
        hrm.load_state_dict(torch.load("hrm_checkpoint.pth", map_location=device))
    if os.path.exists("disc_checkpoint.pth"):
        print("Loading existing Discriminator checkpoint...")
        discriminator.load_state_dict(torch.load("disc_checkpoint.pth", map_location=device))

    hrm.train()
    discriminator.train()

    print("Entering Infinite Training Loop for Continuous Recursive Improvement...")
    while True: # Infinite Epochs
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

            # Log frequently at start to prove it's working, then periodically
            if step_count < 10 or step_count % 1000 == 0:
                print(f"Step {step_count} | Total Loss: {total_loss.item():.4f} | LM: {lm_loss.item():.4f} | Disc: {d_loss.item():.4f}")

            if step_count % 100000 == 0:
                # Save Checkpoints for Recursion (Every 100k steps)
                torch.save(hrm.state_dict(), "hrm_checkpoint.pth")
                torch.save(discriminator.state_dict(), "disc_checkpoint.pth")
                print("Checkpoints saved. Recursion state updated.")

            # Infinite loop, no break condition

def interact():
    print("Loading model for interaction...")
    # Hyperparameters (Must match training)
    SEQ_LEN = 64
    HIDDEN_DIM = 128
    SKILL_DIM = 4

    text = load_data()
    tokenizer = CharTokenizer(text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hrm = HRM(tokenizer.vocab_size, HIDDEN_DIM, skill_dim=SKILL_DIM).to(device)

    if os.path.exists("hrm_checkpoint.pth"):
        hrm.load_state_dict(torch.load("hrm_checkpoint.pth", map_location=device))
        print("Loaded trained model.")
    else:
        print("No checkpoint found. Running with untrained weights (expect noise).")

    hrm.eval()

    prompt = "What is the nature of reality?"
    print(f"\nQuestion: {prompt}")

    # Encode prompt
    prompt_idxs = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_idxs, dtype=torch.long).unsqueeze(0).to(device) # (1, seq)

    print("\n--- AI Responses by Persona ---")

    for i in range(SKILL_DIM):
        z_vec = F.one_hot(torch.tensor([i]), num_classes=SKILL_DIM).float().to(device)

        # Autoregressive Generation
        curr_seq = prompt_tensor
        generated_text = prompt

        with torch.no_grad():
            for _ in range(100): # Generate 100 chars
                logits, _ = hrm(curr_seq, z_vec)
                last_logits = logits[:, -1, :]
                # Temperature sampling
                probs = F.softmax(last_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)

                curr_seq = torch.cat([curr_seq, next_token], dim=1)
                generated_text += tokenizer.decode(next_token[0].tolist())

                if tokenizer.decode(next_token[0].tolist()) == '<PAD>': break

        print(f"\n[Persona {i}]: {generated_text}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interact":
        interact()
    else:
        train()
