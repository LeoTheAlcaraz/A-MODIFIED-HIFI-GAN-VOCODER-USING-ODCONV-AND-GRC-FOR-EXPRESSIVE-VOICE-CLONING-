# CVSS Dataset Setup Guide

## Overview

CVSS (Common Voice Speech-to-Speech Translation) is a massively multilingual-to-English speech-to-speech translation corpus covering 21 languages. This guide explains how to set up and use CVSS with your thesis project.

## What is CVSS?

CVSS provides:
- **Source Audio**: From Common Voice dataset (21 languages)
- **Translation Audio**: Synthesized English speech from two TTS models
- **Translation Text**: Normalized English text
- **Two Versions**:
  - **CVSS-C**: Single canonical speaker voice
  - **CVSS-T**: Multiple target speaker voices

## Supported Languages

| Language | Code | Language | Code |
|----------|------|----------|------|
| Arabic | ar | Catalan | ca |
| Welsh | cy | German | de |
| Estonian | et | Spanish | es |
| Persian | fa | French | fr |
| Indonesian | id | Italian | it |
| Japanese | ja | Latvian | lv |
| Mongolian | mn | Dutch | nl |
| Portuguese | pt | Russian | ru |
| Slovenian | sl | Swedish | sv |
| Tamil | ta | Turkish | tr |
| Chinese | zh | | |

## Installation Steps

### 1. Download CVSS Dataset

```bash
# Download CVSS-C (canonical voice) for all languages
python scripts/download_cvss_dataset.py --version cvss_c

# Download CVSS-T (target voices) for all languages  
python scripts/download_cvss_dataset.py --version cvss_t

# Download both versions
python scripts/download_cvss_dataset.py --version both

# Download specific language (e.g., Spanish)
python scripts/download_cvss_dataset.py --version cvss_c --language es
```

### 2. Setup Common Voice Dataset

CVSS requires Common Voice release version 4 for source audio:

```bash
# Setup Common Voice instructions
python scripts/download_cvss_dataset.py --setup-cv
```

Then manually download Common Voice from: https://commonvoice.mozilla.org/

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Structure

After download, your data directory will look like:

```
data/datasets/
├── cvss_c_es_en/           # Spanish -> English (CVSS-C)
│   ├── train/              # Training audio clips
│   ├── dev/                # Development audio clips  
│   ├── test/               # Test audio clips
│   ├── train.tsv           # Training metadata
│   ├── dev.tsv             # Development metadata
│   └── test.tsv            # Test metadata
├── cvss_t_es_en/           # Spanish -> English (CVSS-T)
│   └── ...                 # Same structure as above
├── common_voice/           # Common Voice dataset
│   ├── clips/              # Source audio clips
│   ├── train.tsv           # Training metadata
│   ├── dev.tsv             # Development metadata
│   └── test.tsv            # Test metadata
└── cvss_dataset_index.json # Dataset index
```

## Usage in Your Project

### 1. Load CVSS Dataset

```python
import pandas as pd
from pathlib import Path

# Load CVSS metadata
cvss_dir = Path("data/datasets/cvss_c_es_en")
train_df = pd.read_csv(cvss_dir / "train.tsv", sep="\t")

# Load Common Voice metadata  
cv_dir = Path("data/datasets/common_voice")
cv_train_df = pd.read_csv(cv_dir / "train.tsv", sep="\t")

# Match audio files by filename
# CVSS audio files correspond to Common Voice file names
```

### 2. Audio File Pairing

```python
# Example: Get source and target audio for a sample
sample_id = "common_voice_es_12345"

# Source audio (from Common Voice)
source_audio_path = cv_dir / "clips" / f"{sample_id}.mp3"

# Target audio (from CVSS)
target_audio_path = cvss_dir / "train" / f"{sample_id}.flac"

# Translation text
translation_text = train_df[train_df["path"] == f"{sample_id}.flac"]["sentence"].iloc[0]
```

### 3. Integration with Your Models

```python
# For ASR training (source language)
source_audio_files = list(cv_dir.glob("clips/*.mp3"))

# For TTS training (target language)  
target_audio_files = list(cvss_dir.glob("train/*.flac"))

# For translation training
# Use source text from Common Voice + target text from CVSS
```

## Dataset Statistics

Each language pair typically contains:
- **Training**: ~10,000-50,000 samples
- **Development**: ~1,000-5,000 samples  
- **Test**: ~1,000-5,000 samples

Total dataset size: ~1.5TB (both versions)

## Citation

If you use CVSS in your research, please cite:

```bibtex
@inproceedings{jia2022cvss,
    title={{CVSS} Corpus and Massively Multilingual Speech-to-Speech Translation},
    author={Jia, Ye and Tadmor Ramanovich, Michelle and Wang, Quan and Zen, Heiga},
    booktitle={Proceedings of Language Resources and Evaluation Conference (LREC)},
    pages={6691--6703},
    year={2022}
}
```

## Troubleshooting

### Download Issues
- Check internet connection
- Verify disk space (need ~1.5TB for full dataset)
- Try downloading individual languages if bulk download fails

### Audio File Matching
- Ensure Common Voice file names match CVSS file names
- Check audio file formats (Common Voice: MP3, CVSS: FLAC)
- Verify dataset versions match

### Memory Issues
- Process datasets in batches
- Use streaming audio loading
- Consider using subset of languages for initial testing

## Next Steps

1. **Start Small**: Download 1-2 languages for testing
2. **Validate Setup**: Run audio file matching tests
3. **Integrate Models**: Connect to your ASR/Translation/TTS pipeline
4. **Scale Up**: Download additional languages as needed

For more information, see the [original CVSS repository](https://github.com/google-research-datasets/cvss). 