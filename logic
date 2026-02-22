# Architectural Patterns & Design Decisions

Patterns observed across both notebooks (`Facial and Text Recognition Bank.ipynb` and `InfoExtract_updated.ipynb`).

---

## 1. Dual-Branch Hybrid Retrieval with Alpha Blending

The core inference pattern. Two independent retrieval signals are computed and linearly combined:

```
final_score = alpha * face_score + (1 - alpha) * text_score
```

- `alpha=1.0` → face-only; `alpha=0.0` → text-only; `alpha=0.5` → balanced hybrid
- Neither branch has privileged status — the same fusion formula is used for all modes
- Applied in `search_pipeline()` and `search_pipeline_explainable()` in the inference notebook

This makes the system's confidence tunable post-training without retraining.

---

## 2. Distance-to-Similarity Inversion

Raw L2 distances from FAISS are converted to similarity scores via:

```
similarity = 1 / (1 + L2_distance)
```

- Maps [0, ∞) distances to (0, 1] similarities
- Guarantees identical scale to the [0, 1] cosine similarity from TF-IDF, enabling direct addition in the fusion step
- Applied to every face candidate in the retrieval loop

---

## 3. Candidate Oversampling

The face branch retrieves `k * 10` candidates from FAISS before scoring, while the text branch scores all documents:

```python
D, I = face_index.search(query_embedding, k * 10)
```

Rationale: FAISS L2 search is fast but returns more than needed to give the fusion step a richer candidate pool. Text cosine similarity is computed over the full matrix regardless.

---

## 4. GPU/CPU Fallback Pattern

Used consistently in both notebooks for FAISS indexing and model inference:

```python
if torch.cuda.is_available():
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, dim)
else:
    index = faiss.IndexFlatL2(dim)
```

Same pattern for Donut and face model:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

Allows the same codebase to run on GPU workstations or CPU-only machines without code changes.

---

## 5. Offline Index / Online Query Split

All heavy computation happens once at index-build time; inference is query-only:

| Phase | Output artifact | Cost |
|---|---|---|
| Training | `custom_trained_donut_model/` | ~1+ hours |
| Index build | `face_index.bin`, `tfidf_data.pkl`, `metadata_store.pkl` | ~72 minutes for 41,852 images |
| Inference | Dashboard render | Seconds per query |

The three pickle/binary artifacts decouple inference from the dataset entirely — the raw images are not needed at query time.

---

## 6. Zero-Vector Fallback for Failed Face Detection

When InsightFace finds no face in an image, a zero embedding is stored rather than skipping the record:

```python
if len(faces) > 0:
    emb = faces[0].normed_embedding.astype('float32')
else:
    emb = np.zeros(512, dtype='float32')
```

This preserves index alignment (embedding index `i` always maps to metadata index `i`), avoiding the need for a separate index-to-record mapping.

---

## 7. Global Model State (Load-Once Pattern)

Both the Donut model and the InsightFace `FaceAnalysis` object are loaded once into module-level variables, not re-instantiated per call:

```python
# Top-level — executed once
model, processor = load_custom_donut()
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
```

All pipeline functions (`search_pipeline`, `search_pipeline_explainable`, `extract_text_donut`) reference these globals. This avoids repeated model loading overhead on repeated queries within the same session.

---

## 8. Multi-Key Metadata Lookup

Dataset images are matched to their JSON metadata using a two-step fallback:

```python
meta = metadata_map.get(filename_with_extension)
if not meta:
    meta = metadata_map.get(os.path.splitext(filename)[0])
```

Handles inconsistency between annotation JSON keys (which may or may not include the file extension) and the actual filenames on disk.

---

## 9. GPU→CPU Index Serialization

FAISS GPU indices cannot be saved directly; they are converted to CPU before persisting:

```python
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, "face_index.bin")
```

On load, the reverse conversion is applied:

```python
cpu_index = faiss.read_index("face_index.bin")
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

This pattern appears in both the index-build cells and the inference setup cells.

---

## 10. Streaming Iterable Dataset for Memory-Efficient Training

The full 41,852-image dataset is not loaded into RAM. Instead it is wrapped as a HuggingFace `IterableDataset` with a streaming shuffle:

```python
iterable = hf_dataset.to_iterable_dataset()
iterable = iterable.shuffle(buffer_size=100)
iterable = iterable.map(transform_generator)
```

`transform_generator` applies Donut preprocessing on-the-fly per sample, keeping peak memory bounded regardless of dataset size.

---

## 11. Structured Special-Token Vocabulary

Document fields are demarcated with matched XML-style special tokens added to the MBart tokenizer:

```
<s_name>MALYSHEV</s_name>
<s_id>28.10.1998</s_id>
<s_address>УФА</s_address>
```

Adding these tokens extends the vocabulary (`tokenizer_config.json`: 57,531 total) and requires resizing the model's embedding matrix after tokenizer modification. This pattern allows the decoder to produce structured, parseable output rather than free-form text.

---

## 12. Dual Visualization Outputs

The inference notebook provides two rendering paths for the same result set:

| Path | Style | Use case |
|---|---|---|
| Matplotlib grid | 2×6 subplots with colored borders | Quick review in notebook |
| HTML dashboard (Sparrow) | Dark-theme card layout, base64 images | Sharable audit report |
| HTML dashboard (Explainable) | Terminal-green aesthetic, per-score math | Technical debugging |

Both dashboards convert PIL images to base64-encoded JPEG data URIs, removing any filesystem dependency from the rendered output.
