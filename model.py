import torch
import torch.nn as nn
import torch.nn.functional as F

class HRM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, skill_dim, T=4):
        super(HRM, self).__init__()
        self.hidden_dim = hidden_dim
        self.skill_dim = skill_dim
        self.T = T # Not strictly used in this simple version, but kept for signature compatibility

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Skill projection to hidden state
        self.skill_proj = nn.Linear(skill_dim, hidden_dim)

        # High-level RNN (manages abstract state)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, skill_vec=None):
        # x: (batch_size, seq_len)
        # skill_vec: (batch_size, skill_dim)

        embedded = self.embedding(x) # (batch, seq, hidden)

        # Initialize hidden state with skill information
        # This simulates how the skill alters the "High-Level State h_h"
        batch_size = x.size(0)

        if skill_vec is not None:
            # Project skill vector to hidden dimension
            h_0 = self.skill_proj(skill_vec).unsqueeze(0) # (1, batch, hidden)
            c_0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        else:
            h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
            c_0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)

        # Run LSTM
        # In a real HRM, this might be hierarchical, but for this interface, a standard LSTM
        # conditioned on the skill at init is sufficient to demonstrate the concept.
        outputs, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))

        logits = self.fc(outputs) # (batch, seq, vocab)

        return logits, (h_n, c_n)
