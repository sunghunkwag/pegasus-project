import torch
import torch.nn as nn

class SkillDiscriminator(nn.Module):
    def __init__(self, vocab_size, hidden_dim, skill_dim):
        super(SkillDiscriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # The discriminator tries to predict the skill index from the text
        self.classifier = nn.Linear(hidden_dim, skill_dim)

    def forward(self, x):
        # x: (batch_size, seq_len) (indices)
        # Note: In the training loop provided, 'gen_tokens' are indices (argmax).
        # However, if we wanted to backprop through the generator, we'd need Gumbel-Softmax
        # or the provided loop's approach (optimizing G via REINFORCE/policy gradient reward).
        # The provided code does `loss_g = (lm_loss - ENTROPY_COEF * rewards)`.
        # The `rewards` come from the discriminator.

        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        # Use the final hidden state to classify the sequence
        logits = self.classifier(h_n.squeeze(0))
        return logits
