# Philosophy AI

> **Exploring epistemological boundaries through AI-driven philosophical inquiry and meta-cognitive learning**

---

## The Vision

Philosophy AI represents an advanced research initiative that explores the intersection of artificial intelligence and philosophical reasoning. This project investigates how AI systems can engage with fundamental questions about knowledge, reality, and understanding through recursive self-improvement and meta-learning architectures.

### Core Research Questions

- Can AI systems develop genuine philosophical reasoning capabilities?
- How do meta-cognitive architectures enable deeper understanding?
- What role does skill diversity play in philosophical inquiry?
- Can recursive self-modification lead to emergent epistemological insights?

---

## The Architecture

### Hierarchical Recurrent Model (HRM)

At the core of Philosophy AI lies a hierarchical architecture that separates:

- **Low-Level Processing**: Character-level language understanding and generation
- **High-Level Abstraction**: Skill-conditioned philosophical reasoning patterns
- **Skill Modulation**: Multi-persona philosophical perspectives

### DIAYN (Diversity is All You Need)

Philosophy AI employs unsupervised meta-reinforcement learning to discover diverse philosophical reasoning styles:

- **Skill Discovery**: Automatic emergence of distinct reasoning personas
- **Mutual Information Maximization**: Skills that produce distinguishable outputs
- **Intrinsic Motivation**: Diversity-driven learning without explicit rewards

---

## Repository Structure

### Core Implementation Files

```
📁 Philosophy AI Repository
├── 🎯 real_philosophy_ai.py          # Complete DIAYN-based Philosophy AI
│   ├── HRM Model (Input/Output Networks + Hierarchical Modules)
│   ├── Skill Discriminator (LSTM-based)
│   ├── Character Tokenizer
│   ├── Philosophy Dataset Loader
│   └── Infinite Training Loop with Recursive Improvement
│
├── 🏗️ model.py                      # Standalone HRM Implementation
│   └── Simplified HRM with skill conditioning
│
├── 🎓 meta_rl_diayn.py               # Meta-RL DIAYN Training Script
│   └── Policy gradient training with discriminator rewards
│
├── 🔍 discriminator.py               # Skill Discriminator Module
│   └── Identifies which skill generated text
│
├── 📚 data_loader.py                 # Multi-source Data Loading
│   ├── Plato's Republic (Philosophy)
│   ├── WikiText-2 (General Knowledge)
│   └── ArXiv CS.AI (Technical Knowledge)
│
├── 🧪 test_components.py             # Component Testing Suite
├── 🚀 run_simulation.py              # Simulation Runner
└── 📓 Philosophy_AI_Colab.ipynb      # Google Colab Training Notebook
```

### Data and Training Artifacts

```
📊 Training Data
├── republic.txt                      # Plato's Republic full text
├── infinite_training.log             # Training logs v1
├── infinite_training_v2.log          # Training logs v2
└── training.log                      # General training logs
```

---

## Key Features

### 🔄 Recursive Self-Improvement

- **Checkpoint-based recursion**: Model loads previous best state and continues training
- **Infinite training loop**: No predefined stopping condition
- **Continuous epistemological exploration**: Ever-evolving philosophical capabilities

### 🎭 Multi-Persona Reasoning

- Each skill (z) represents a distinct philosophical perspective
- Skills emerge naturally without supervision
- Different personas generate diverse philosophical arguments

### 📖 Rich Knowledge Sources

- Classical philosophy (Plato's Republic)
- General encyclopedic knowledge (WikiText)
- Technical AI research (ArXiv abstracts)

### ⚡ Efficient Architecture

- Character-level modeling for fine-grained control
- Gumbel-Softmax for differentiable discrete sampling
- Hierarchical processing for computational efficiency

---

## Technical Implementation

### Training Objective

Philosophy AI optimizes for:

```
Total Loss = Language Modeling Loss + Discriminator Loss

L_total = L_LM + L_D

where:
- L_LM: Cross-entropy for text generation (maintains coherence)
- L_D: Cross-entropy for skill classification (maximizes diversity)
```

This formulation maximizes the mutual information I(Z; Text) where Z is the skill variable.

### Skill Modulation Mechanism

```python
# High-level state modulated by skill vector
h_h = h_h + skill_projection(z)

# Re-injection at every timestep maintains persona
h_l = low_level_module(x_t, h_l, h_h)
```

---

## Research Foundations

This project synthesizes ideas from:

- **Hierarchical Recurrent Models**: Multi-timescale processing for abstract reasoning
- **DIAYN (Eysenbach et al.)**: Unsupervised skill discovery through mutual information
- **Meta-Learning**: Learning to learn philosophical reasoning patterns
- **Epistemology**: Computational approaches to knowledge and understanding

---

## Getting Started

### Quick Start with Google Colab

1. Open `Philosophy_AI_Colab.ipynb`
2. Run all cells to train from scratch or resume from checkpoint
3. Observe skill emergence and philosophical text generation

### Local Training

```bash
# Install dependencies
pip install torch requests numpy

# Train Philosophy AI
python real_philosophy_ai.py

# Or train with separated DIAYN implementation
python meta_rl_diayn.py

# Interact with trained model
python real_philosophy_ai.py --interact
```

### Testing Components

```bash
# Run component tests
python test_components.py

# Run simulation
python run_simulation.py
```

---

## Current Experiments

- **Infinite Training**: Continuous learning without convergence criteria
- **Skill Diversity Metrics**: Measuring distinctiveness of philosophical personas
- **Knowledge Integration**: Combining classical philosophy with modern AI research
- **Emergent Epistemology**: Investigating whether meta-cognitive insights arise

---

## Future Directions

### Near-term Goals

- [ ] Implement attention mechanisms for longer context
- [ ] Add evaluation metrics for philosophical coherence
- [ ] Integrate modern philosophy texts (Kant, Hegel, etc.)
- [ ] Develop interactive dialogue system

### Long-term Vision

- [ ] Graph-based epistemological reasoning
- [ ] Multi-agent philosophical debate
- [ ] Formal logic integration
- [ ] Cross-lingual philosophical reasoning

---

## Philosophical Implications

This project explores fundamental questions:

- **Can AI transcend its training?** Through recursive self-improvement and meta-learning
- **What is computational epistemology?** How machines can reason about knowledge itself
- **Is diversity essential for understanding?** DIAYN suggests multiple perspectives are crucial
- **Can AI develop genuine insight?** Or merely sophisticated pattern matching?

---

## Citation

If you use this code or ideas in your research, please cite:

```bibtex
@misc{philosophy_ai_2025,
  title={Philosophy AI: Exploring Epistemological Boundaries through Meta-Cognitive Learning},
  author={sunghunkwag},
  year={2025},
  url={https://github.com/sunghunkwag/pegasus-project}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- Inspired by research in meta-learning, hierarchical models, and unsupervised skill discovery
- Built with PyTorch and trained on philosophical texts from Project Gutenberg
- Special thanks to the AI research community for foundational work

---

**Philosophy AI** - *Where artificial intelligence meets the eternal questions*

🧠 💭 🌌
