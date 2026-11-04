# Revised table: six encoder compositions

This study runs NAML–SUE, NRMS–SUE, LSTUR–SUE, CNE–NAML,
CNE–NRMS, and CNE–LSTUR on MIND-small with paired seeds 0–4 (30 runs).
It uses the existing MIND-small split and CNE–SUE study's optimizer,
sequence lengths, checkpoint criterion (`avg`), and ten-epoch budget.
All pairings use standard cross-entropy training. The backbone row in the
supplied table is a reference, not a newly measured baseline in this study.

The shared representation width is 900, as in the prior composition study.
NAML, NRMS, and LSTUR news encoders therefore include a trained dimension
projection. SUE and NRMS use the existing representation-level modules.
The study-local adapters reuse native NAML (`ATT`) and LSTUR user encoders,
including LSTUR's learned user embeddings and long-term masking. No existing
core source file is edited. Zero core edits does not mean zero added code:
`components.py` provides additional generic experiment infrastructure.

The supplied NAML–SUE AUC 0.6801 / MRR 0.3211 matches the prior B5
six-seed CNRCL-training result. It must not be reported as a new five-seed
standard-training result. Existing results are never imported into this study.

```bash
OMP_NUM_THREADS=1 python scripts/run_table_composition.py prepare
sbatch experiments/table_composition/train.slurm
# Add --dependency=afterany:<array-job-id> to the following submission:
sbatch experiments/table_composition/report.slurm
python scripts/run_table_composition.py report
```

Array tasks 0–4 run NAML–SUE; 5–9 NRMS–SUE; 10–14 LSTUR–SUE;
15–19 CNE–NAML; 20–24 CNE–NRMS; 25–29 CNE–LSTUR. Each group
uses seeds 0–4. At most three tasks train concurrently on the L4 partition.

Artifacts are under `artifacts/table_composition`: lifecycle evidence,
source hashes, per-run YAML and JSON files, scheduler logs, and report
CSV/JSON/LaTeX rows. Reports only show means and sample standard deviations
after all five seeds for a pairing finish. Lifecycle checks are synthetic
CPU checks of initialization, forward/backward, checkpoint recovery, and
a simulated train/validation/test workflow; full data runs are separate evidence.
LaTeX boldface compares the six measured composition rows only; re-evaluate
boldface after inserting the backbone row into the final paper table.

The shared environment has PyTorch 2.5 but a PyTorch 2.4 scatter wheel,
which crashes in scatter-softmax backward. This study uses an isolated
`torch_scatter==2.1.2+pt25cu121` installation in its artifact `vendor`
directory. The runner prepends that directory before importing models.
Recreate it with the same Python interpreter if moving the study:

```bash
python -m pip install --no-deps --no-index \
  --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html \
  --target artifacts/table_composition/vendor torch_scatter
```
