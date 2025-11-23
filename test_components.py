import torch
import unittest
from model import HRM
from discriminator import SkillDiscriminator
from data_loader import load_philosophy_data, SimpleTokenizer

class TestComponents(unittest.TestCase):
    def test_data_loader(self):
        loader, tokenizer = load_philosophy_data(seq_len=10, batch_size=5)
        self.assertIsInstance(tokenizer, SimpleTokenizer)
        batch = next(iter(loader))
        real_text, labels = batch
        self.assertEqual(real_text.shape, (5, 10))

    def test_hrm_forward(self):
        vocab_size = 100
        hidden_dim = 32
        skill_dim = 4
        model = HRM(vocab_size, hidden_dim, skill_dim)

        batch_size = 2
        seq_len = 10
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        skill_vec = torch.zeros(batch_size, skill_dim)
        skill_vec[:, 0] = 1 # Set first skill active

        outputs, (h_n, c_n) = model(x, skill_vec)

        self.assertEqual(outputs.shape, (batch_size, seq_len, vocab_size))
        self.assertEqual(h_n.shape, (1, batch_size, hidden_dim))

    def test_discriminator_forward(self):
        vocab_size = 100
        hidden_dim = 32
        skill_dim = 4
        disc = SkillDiscriminator(vocab_size, hidden_dim, skill_dim)

        batch_size = 2
        seq_len = 10
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = disc(x)
        self.assertEqual(logits.shape, (batch_size, skill_dim))

if __name__ == '__main__':
    unittest.main()
