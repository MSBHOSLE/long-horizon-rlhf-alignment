# 🎯 Long-Horizon RLHF Alignment

[![CI Pipeline](https://github.com/MSBHOSLE/long-horizon-rlhf-alignment/actions/workflows/ci.yml/badge.svg)](https://github.com/MSBHOSLE/long-horizon-rlhf-alignment/actions)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-TRL-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/docs/trl)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DPO](https://img.shields.io/badge/DPO-Alignment-blue)](https://arxiv.org/abs/2305.18290)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A post-training LLM alignment framework utilizing **Direct Preference Optimization (DPO)** and reinforcement learning for multi-step reasoning, mathematical problem-solving, and agentic multi-turn trajectories.

---

## ⚡ Highlights

- **Direct Preference Optimization**: Stable closed-form alternative to PPO without requiring an explicit reward model.
- **Reward Margin Tracking**: Evaluates policy drift and implicit log-ratio advantages over the frozen reference model.
- **Multi-Step Reasoning Alignment**: Optimizes long-horizon reasoning trajectories on chain-of-thought datasets.

---

## 🚀 Running Tests

```bash
python -m unittest discover tests
```

---

## 👤 Author

**Mahesh Sanjayrao Bhosle**
- **GitHub:** [@MSBHOSLE](https://github.com/MSBHOSLE)
- **LinkedIn:** [Mahesh Bhosle](https://linkedin.com/in/mahesh-bhosle-916991198)
- **Email:** msbhosle.2@gmail.com
- **Education:** M.Sc. Data & Computer Science, Heidelberg University

---

## 📄 License

MIT License - see [LICENSE](LICENSE).
