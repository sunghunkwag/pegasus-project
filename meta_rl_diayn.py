import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import HRM
from discriminator import SkillDiscriminator
from data_loader import load_philosophy_data
import numpy as np

def train_diayn():
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 32
    HIDDEN_DIM = 256
    SKILL_DIM = 4 # Number of distinct personas to discover
    LR_G = 1e-4 # Generator (HRM) LR
    LR_D = 1e-3 # Discriminator LR
    EPOCHS = 5
    ENTROPY_COEF = 0.1 # Encourage diversity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DIAYN on {device}")

    dataloader, tokenizer = load_philosophy_data(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    vocab_size = tokenizer.vocab_size

    hrm = HRM(vocab_size, HIDDEN_DIM, skill_dim=SKILL_DIM, T=4).to(device)
    discriminator = SkillDiscriminator(vocab_size, HIDDEN_DIM, SKILL_DIM).to(device)

    opt_g = optim.Adam(hrm.parameters(), lr=LR_G)
    opt_d = optim.Adam(discriminator.parameters(), lr=LR_D)
    criterion_d = nn.CrossEntropyLoss()

    print("Starting DIAYN Training...")

    for epoch in range(EPOCHS):
        total_reward = 0
        total_d_loss = 0

        for i, (real_text, _) in enumerate(dataloader):
            real_text = real_text.to(device)
            batch_size = real_text.size(0)

            z_idx = torch.randint(0, SKILL_DIM, (batch_size,)).to(device)
            z_vec = F.one_hot(z_idx, num_classes=SKILL_DIM).float().to(device)

            outputs, _ = hrm(real_text, skill_vec=z_vec)

            with torch.no_grad():
                gen_tokens = torch.argmax(outputs, dim=-1)

            d_logits = discriminator(gen_tokens)
            d_loss = criterion_d(d_logits, z_idx)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            with torch.no_grad():
                d_logits_new = discriminator(gen_tokens)
                log_probs = F.log_softmax(d_logits_new, dim=-1)
                rewards = log_probs.gather(1, z_idx.unsqueeze(1)).squeeze()

            lm_loss = nn.CrossEntropyLoss(reduction='none')(outputs.reshape(-1, vocab_size), real_text.reshape(-1))
            lm_loss = lm_loss.view(batch_size, -1).mean(dim=1)

            loss_g = (lm_loss - ENTROPY_COEF * rewards).mean()

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            total_reward += rewards.mean().item()
            total_d_loss += d_loss.item()

            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Step {i} | D_Loss: {d_loss.item():.4f} | Reward: {rewards.mean().item():.4f}")

        print(f"Epoch {epoch+1} Done. Avg Reward: {total_reward / len(dataloader):.4f}")

if __name__ == "__main__":
    train_diayn()
