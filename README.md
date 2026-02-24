# KYC Forgery Detection: SOTA

## Project Overview

Deep learning system for detecting forgeries in KYC (Know Your Customer) identity documents.
Operates on Russian ID cards and classifies six attack types: copy and paste manipulation,
face morphing, face replacement, combined methods, inpainting/rewriting, and crop and replace.
Core approach: hybrid retrieval over a pre-built database using biometric (ArcFace face embeddings)
and textual (Donut OCR) signals, fused via a weighted alpha parameter.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Training | PyTorch 2.9.0 + CUDA 12.1, HuggingFace Accelerate 0.27.2 |
| OCR | Donut (VisionEncoderDecoderModel): Swin-Transformer encoder + MBart decoder |
| Face Recognition | InsightFace ArcFace `buffalo_l` (512-d embeddings), ONNX Runtime GPU |
| Vector Search | FAISS 1.9.0 (GPU, IndexFlatL2) |
| Text Search | scikit-learn TF-IDF + cosine similarity |
| Data | HuggingFace Datasets, PIL, OpenCV, NumPy, PyArrow/Parquet |

## Key Directories

```
SOTA/
├── Dataset/RUS/
│   ├── positive/                    # 5,979 authentic ID card images
│   ├── fraud1_copy_and_move/        # 5,979 copy-paste forgeries
│   ├── fraud2_face_morphing/        # 5,979 face-morph attacks
│   ├── fraud3_face_replacement/     # 5,979 face-swap documents
│   ├── fraud4_combined/             # 5,979 multi-method forgeries
│   ├── fraud5_inpaint_and_rewrite/  # 5,979 inpainting forgeries
│   ├── fraud6_crop_and_replace/     # 5,978 cropped-region replacements
│   └── meta/detailed_with_fraud_info/  # Per-class JSON annotation files
├── custom_trained_donut_model/      # Fine-tuned Donut weights (777 MB)
├── training_output/                 # Checkpoints at steps 50 and 100
├── face_index.bin                   # FAISS index (82 MB, 41,852 embeddings)
├── metadata_store.pkl               # ID→fraud-label mapping (35 MB)
└── tfidf_data.pkl                   # TF-IDF vectorizer + matrix (26 MB)
```

## Key Files

| File | Role |
|---|---|
| `Facial and Text Recognition Bank.ipynb` | Inference: loads indices, runs hybrid search, renders dashboard |
| `InfoExtract_updated.ipynb` | Training: fine tunes Donut on RUS dataset, builds FAISS/TF-IDF indices |
| `instalation_instructions.txt` | Conda env setup (`sotaenv`) and full pip/conda install commands |
| `custom_trained_donut_model/config.json` | Model architecture (Swin 1024-d encoder, 4-layer MBart decoder) |
| `Dataset/RUS/meta/detailed_with_fraud_info/*.json` | Ground truth annotation schemas with bbox + field metadata |

## Essential Commands

### Environment Setup
```bash
conda create -n sotaenv python=3.11 && conda activate sotaenv
# Full package list: see instalation_instructions.txt
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install insightface onnxruntime-gpu transformers==4.37.2 accelerate==0.27.2 datasets
conda install -c conda-forge faiss-gpu
pip install opencv-python-headless pillow matplotlib scikit-learn tqdm sentencepiece protobuf tk
```

### Training (`InfoExtract_updated.ipynb`)
```bash
jupyter notebook InfoExtract_updated.ipynb
# Run cells in order: load dataset → tokenize → Seq2SeqTrainer.train() → build indices
# Key hyperparams: lr=2e-5, batch=2, grad_accum=4, max_steps=100
# Outputs: custom_trained_donut_model/, face_index.bin, tfidf_data.pkl, metadata_store.pkl
```

### Inference (`Facial and Text Recognition Bank.ipynb`)
```bash
jupyter notebook "Facial and Text Recognition Bank.ipynb"
# Run cells in order: load indices → select query image → search_pipeline() → render dashboard
# Key param: alpha — 1.0=face-only, 0.0=text-only, 0.5=hybrid (default)
```

## Additional Documentation

| Topic | File |
|---|---|
| Architectural patterns & design decisions | `.logic/architectural_breakdown.md` |
