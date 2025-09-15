# Context-Aware Dynamic Graph Learning for Multimodal Emotion Recognition with Missing Modalities

Official PyTorch implementation for ICASSP 2026 submission.

## 📌 Abstract

We propose a novel framework that combines context-aware dynamic graph learning with multimodal imagination for robust emotion recognition under missing modality conditions. Our approach leverages LLM-guided semantic extraction and graph neural ODEs to maintain high performance even when up to 67% of modalities are unavailable.

## 🎯 Main Results

Performance on multimodal emotion recognition benchmarks:

| Dataset | Complete | 1 Missing | 2 Missing |
|---------|----------|-----------|-----------|
| IEMOCAP | 84.2%    | 76.8%     | 65.5%     |
| MELD    | 65.8%    | 60.3%     | 51.4%     |

## 🚀 Quick Start

```python
from models import CADGL
import torch

# Initialize model
model = CADGL(config)

# Forward pass with missing modalities
output = model(
    audio=audio_features,      # Can be None
    visual=visual_features,     # Can be None  
    text=text_features,         # Can be None
    modality_masks=masks
)
```

## 📋 Requirements

Core dependencies:
```
torch>=2.0.0
torch-geometric
torchdiffeq
transformers
numpy
```

For complete requirements, see `requirements.txt`

## 🗂️ Repository Structure

```
├── models/           # Model architecture
├── configs/          # Configuration files
├── scripts/          # Training and evaluation scripts
├── demo/            # Inference examples
└── docs/            # Additional documentation
```

## 💾 Pretrained Models

Pretrained models will be made available upon paper acceptance.

For early access, please contact the authors.

## 📊 Dataset Preparation

Please refer to the respective dataset papers for download instructions:
- IEMOCAP: [Link](https://sail.usc.edu/iemocap/)
- MELD: [Link](https://github.com/declare-lab/MELD)

Feature extraction details are provided in Section 3.2 of our paper.

## 🔬 Training

Basic training command:
```bash
python scripts/train.py --config configs/default.yaml
```

For detailed training procedures and hyperparameters, please refer to our paper.

## 📈 Evaluation

```bash
python scripts/evaluate.py --checkpoint path/to/model.pth
```

## 📝 Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{cadgl2026icassp,
  title={Context-Aware Dynamic Graph Learning for Multimodal Emotion Recognition with Missing Modalities},
  author={[Authors]},
  booktitle={ICASSP 2026},
  year={2026}
}
```

## 📄 License

This project is released for academic research use only. For commercial use, please contact [email].

## 📧 Contact

For questions and collaborations, please open an issue or contact us at [email].

## 🔄 Updates

- **[Date]**: Code release (planned after acceptance)
- **[Date]**: Initial repository setup

---

**Note**: This is a preliminary release. Full code and pretrained models will be released upon paper acceptance.