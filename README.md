# Path-CT four-view subtype discovery

Production uses four patient-level views only: CT, WSI, RNA, and WXS. CNV is not loaded as a required view, fused, or exposed to the subtype-review Agent.

## Production flow

```text
inventory/QC
  → CT, WSI, RNA, WXS feature engineering and affinities
  → SNF
  → 80% patient resampling × 500
  → hierarchical + spectral + k-medoids co-association consensus for K=2…8
  → requested K × repeat Agent reviews
  → after all K=2…8 × repeats 1…3 complete:
      accepted co-membership → recurrent cores → merged states
```

CT discovery features use cached PyRadiomics values, constant/near-constant feature removal, absolute Pearson-correlation pruning above 0.95, feature-wise z-scoring, Euclidean distance, and affinity conversion. WSI uses normalized GigaPath slide embeddings and cosine distance. RNA candidate features use the filtered/log-transformed, top-2000-MAD expression matrix; the full filtered log2 expression matrix is kept separately for Hallmark GSEA. WXS uses nonsynonymous binary mutation features with prevalence ≥5%, Jaccard distance, and `empty_mutation_distance=1`; curated drivers are not forced into candidate-generation features.

The Agent graph has three nodes: Router requests evidence or chooses an action; Verifier selects one of the seven scientific tools and interprets results; Reviser executes one Router-approved split or merge. Evidence uses a single ledger with `set`, `pair`, and `partition` scopes. Structural revision is one operation per round; terminal accept/drop actions cover the full current partition. LLMs do not choose K.

## Run

Run a single review while reusing the candidate-generation cache:

```bash
python main.py --initial-k 2 --repeat 1
```

Run K=4 three times:

```bash
python main.py --initial-k 4 --repeat 1 --repeat 2 --repeat 3
```

With no K/repeat arguments, the default is the complete K=2…8 × repeat=1…3 grid. Pass `--force` to replace the requested Agent-run directories. It does not erase the output root or force regeneration of valid feature/candidate caches.

## Outputs

The default output root is `output_kirc/` and can be changed with `--output-root`.

- `candidate_subtype/`: four affinity matrices, SNF fused similarity, patient order, feature audit, K=2…8 candidate partitions, per-algorithm and equal-weight consensus matrices.
- `subtype_review/runs/K{k}/repeat{r}/`: individual Agent run evidence, reports, history, and accepted sets.
- `subtype_review/multi_k/`: generated only after all 21 runs are complete and match the current input signature; contains accepted co-membership, recurrent cores, core-to-state mapping, and final patient-state membership.

The configured 2/3 acceptance/co-membership thresholds and minimum core size of five are analysis operating parameters, not universal clinical cutoffs.

The `scripts_*` directories contain historical/independent analyses and are not imported by the production pipeline.
