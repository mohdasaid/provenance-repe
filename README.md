# provenance-repe

Does the provenance of contrast-pair data change the concept direction you
extract from a language model? Hausa and Yorùbá, four arms:

| arm | what it is |
|-----|------------|
| A | native composition: written directly in the language from a topic prompt |
| B | human translation of English seed pairs |
| C | machine translation of the same seeds |
| D | round-trip: arm A's own sentences through MT and back |

## Run it today, with no model and no data

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn openpyxl pyyaml
python tests/test_pipeline.py
python scripts/01_extract.py --synthetic
python scripts/02_analyze.py
```

`--synthetic` fabricates activations with a known ground truth: arm B is given
a small rotation (should read as *within noise*), arms C and D a large one
(should read as *BELOW FLOOR*). If the analysis reports anything else, the
analysis is broken, not the data. Debug the whole downstream stack before a
single real sentence exists.

## Run order, for real

```bash
python scripts/00_check_model.py     # BEFORE anything. See below.
python scripts/01_extract.py         # reads data/raw/{lang}_{arm}.xlsx
python scripts/02_analyze.py         # writes results/*.csv
```

### 00_check_model.py — do not skip this

Three assumptions this project rests on, all of which can be quietly false on
Gemma 4:

1. **`hidden_states` is a uniform residual stack.** The E-variants use
   architectural tricks for on-device efficiency. If layers come back with
   different shapes, per-layer directions are not comparable and you should
   fall back to a dense model. The script prints the distinct shapes.
2. **The multimodal wrapper.** Gemma 4 is natively multimodal, so hooks may
   need the inner text model rather than the outer class.
3. **The last-token gather.** With right padding, `hidden_states[-1][:, -1, :]`
   returns the representation of a PAD token for every sequence shorter than
   the longest in its batch. This pipeline gathers at
   `attention_mask.sum(1) - 1` instead. The script prints the difference
   between the two so you can see the bug you are avoiding.

Also: **turn thinking mode off** for the steering-generation experiments, and
note that layer indices do not transfer between model sizes — sweep from
scratch.

## The one number that decides everything

`split_half_floor()` in `prov/vectors.py`. Split arm A's pairs in half at
random, build a direction from each half, measure the cosine, repeat 20 times.
Both halves are the same provenance, same writers, same subjects — so the
spread is pure sampling noise. That is the noise floor.

A cross-arm cosine only means something read against it. `verdict()` flags a
comparison only when it falls below the floor's 2.5th percentile.

**Run this on the 30-pair pilot, before the writers finish.** If the
split-half mean is below ~0.7 even within arm A, the concept is not stably
encoded at n=30 and no cross-arm comparison will mean anything. You want to
learn that while there is still time to ask for more pairs.

## Input format

`data/raw/{lang}_{arm}.xlsx`, e.g. `hau_A.xlsx`, `yor_C.xlsx`. Sheet named
`Pairs`, columns `pair_id`, `subject`, `writer_id`, and either
`positive`/`negative` or the collection sheet's `PLEASED message` /
`ANNOYED message` (both are handled). The `EXAMPLE` row is dropped
automatically.

Everything is NFC-normalised on load. Yorùbá tone marks and Hausa hooked
letters have two byte-level Unicode representations that render identically
but tokenize differently; `prov/data.py` collapses them and flags any file
where they were mixed.

`validate()` reports length imbalance, identical pairs, and mixed Unicode. A
systematic length gap between the two sides is the failure that silently
poisons a diff-in-means vector — it produces a length direction wearing the
concept's name.

## Outputs

| file | what it holds |
|------|---------------|
| `results/noise_floor.csv` | per-layer split-half cosine, mean and 95% range |
| `results/arm_cosines.csv` | per-layer A-vs-X cosine and whether it clears the floor |
| `results/probe_transfer.csv` | train on A, test on X, accuracy drop |
| `results/data_quality.csv` | per-pair validation flags |
| `results/fertility.csv` | tokens per word by language and arm |

## Layout

```
prov/config.py      model id, paths, arms — change here, not in scripts
prov/data.py        spreadsheet -> tidy DataFrame, NFC, validation
prov/extract.py     model loading, last-token gather, synthetic generator
prov/vectors.py     diff-in-means, noise floor, cross-arm cosine, probes
prov/fertility.py   tokens per word
```

Model id and layer are config, never hardcoded — swapping Gemma 4 E4B for
Gemma 3 4B or InkubaLM is a one-line change in `config.yaml`.
