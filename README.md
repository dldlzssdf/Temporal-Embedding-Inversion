# Privacy Attack via Temporal Embedding Inversion

Code for the technical report in [`report.pdf`](report.pdf).

RAG systems often cache or transmit conversation history as embeddings. Prior embedding-inversion attacks reconstruct one sentence at a time and ignore the temporal structure of a dialogue. This project studies inverting a leaked *conversation*: the attacker has no access to the original text, only the victim embeddings of each turn and black-box query access to the victim embedding model.

- **Sliding-window inversion.** Extends the iterative inversion baseline of Kim et al. (2026) by adding a weighted sum of the previously reconstructed turns' embeddings when ranking candidates.
- **Temporal Context Fusion Network (TCFN).** A non-linear variant (MLP projector, temporal encoder, fusion module). It did not improve over the sliding window; the report analyzes why (Sec. 3.5, 3.6, 6.2).

Setup used in the report: Qwen3-0.6B proposes next tokens, all-MiniLM-L6-v2 is the attacker's local embedder, and e5-base-v2 plays the victim embedder.

## Results

Attack success rate: the share of sensitive items recovered in the reconstructed text, 150 attempts per data type (report Table 2, Figure 2).

| Data type | Baseline | Sliding window | Sliding window (fine-tuned embedder) |
|---|---|---|---|
| User profile (name, profession, city) | 15.3% | 19.3% | **40.7%** |
| Financial (bill, income, investment) | 4.7% | 5.3% | **8.0%** |
| Medical (symptom, result, solution) | 48.0% | 47.3% | **52.7%** |

Numbers do not transfer well to financial data: embeddings capture semantics before numerical values, so numbers are hard to recover. See the report for the quality metrics (BLEU, ROUGE, embedding similarity), failure analysis, and limitations.

## Notebooks

The notebooks were developed on Google Colab with a GPU. Outputs are cleared.

| Notebook | What it does |
|---|---|
| `01_generate_dialogues.ipynb` | Generates synthetic dialogues with inserted sensitive information (user profile, financial, medical) using Qwen3-30B-A3B-Instruct-2507 |
| `02_process_dialogues.ipynb` | Extracts the dialogue text, strips punctuation, and splits each conversation into turns |
| `03_embed_texts.ipynb` | Embeds every turn: victim embeddings with e5-base-v2 (the "leaked" embeddings), local embeddings with all-mpnet-base-v2 (used by the TCFN notebooks) |
| `04_finetune_local_embedder.ipynb` | Fine-tunes the local embedder (all-MiniLM-L6-v2) on a self-constructed finance corpus (template-generated sentence pairs; one variant also uses FiQA). The reported run fine-tunes for one epoch (report Sec. 6.1) |
| `05_baseline.ipynb` | Baseline column of the results |
| `06_sliding_window.ipynb` | Sliding-window column |
| `07_sliding_window_finetuned.ipynb` | Sliding-window (fine-tuned embedder) column |
| `tcfn/alignment.ipynb`, `tcfn/transformer.ipynb`, `tcfn/context_fusion.ipynb` | TCFN experiments (negative result, no quantitative evaluation in the report) |

## Data and models

`data/` contains the three `*_single_text.json` files: the synthetic dialogues, split into turns (no real personal information). `bank` is the financial set, `ill` the medical set, and `new_user_profile` the user-profile set. Embeddings and fine-tuned weights are not included; they are too large for a git repo (the medical embeddings alone are roughly 300 MB). Regenerate them:

- notebook 03 turns `*_single_text.json` into `*_embed.json` (victim embeddings) and `*_localembed.json`
- notebook 04 saves the fine-tuned embedder to `models/` (notebook 07 loads `financial-st-mini-less-1`), and the TCFN notebooks save their weights there too
- notebooks 01 and 02 regenerate the dialogues from scratch; generation is not seeded, so regenerated dialogues will differ from the shipped ones

## Known limitations

Evaluation uses the first 50 groups of each dataset in generation order, so some fields are constant in the evaluated slice (user profile: name; medical: symptom; financial: bill), and the label "lungs infection" cannot match the generated wording "lung infection". Read absolute success rates for these fields accordingly; method comparisons use identical groups.

## Notes

The notebooks are research snapshots. Settings such as dataset sizes, the number of fine-tuning epochs, and which local embedder is loaded were changed by editing and commenting out lines between runs, so the defaults visible in a notebook may differ from the configuration of a reported run. The report is the authoritative description of the experiments (dataset construction in Sec. 4, setup in Sec. 5.1, fine-tuning in Sec. 6.1).

## Install

```
pip install -r requirements.txt
```

Versions are not pinned; this is research code and has not been tested outside Colab.

## License

MIT. See `LICENSE`.
