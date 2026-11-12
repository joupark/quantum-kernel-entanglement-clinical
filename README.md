# Separating State Entanglement from Kernel Effects: A Controlled Study of Quantum and Classical Kernels on Two Clinical Datasets

Code and results for the paper **"Separating State Entanglement from Kernel Effects: A Controlled Study of Quantum and Classical Kernels on Two Clinical Datasets"** (J. Park, *Discover Quantum Science*, under revision).

Every number, table and figure in the paper is produced by the scripts in this repository from the files in `results/`. `build_numbers.py` writes the numbers and tables into LaTeX files that the manuscript includes, and checks each result-dependent statement in the text with an assertion.

## The question and the design

Can a quantum kernel classify ordinary (classical) clinical data better than a well-tuned classical kernel, and is entanglement what would make it do so?

On two clinical datasets, the paper compares a classical RBF kernel with four 6-qubit fidelity quantum kernels. Three of them use the same gate types, single-qubit `RY(s·x)` rotations and a CNOT chain `V`. They differ in whether the entanglement of the encoded states reaches the kernel:

| Kernel | Circuit | Entanglement in the kernel |
|---|---|---|
| Control | `V · R(x)` | cancels exactly (`V†V = I`), so the kernel is `Π cos²(s(x_i − x'_i)/2)` |
| Re-uploading, 2 layers | `R(x) · V · R(x)` | does not cancel |
| Re-uploading, 3 layers | `R(x) · V · R(x) · V · R(x)` | does not cancel |
| ZZ feature map | Qiskit `ZZFeatureMap`, reps=2, linear | does not cancel (data-dependent two-qubit phases) |
| RBF (classical baseline) | `exp(−γ‖x − x'‖²)` | – |

**Protocol**
- Standardization and PCA(6) are fit inside every training fold.
- The input scale `s` (or `γ` for the RBF kernel) and the SVM `C` are chosen by an inner 3-fold CV.
- Evaluation uses an outer 5-fold CV repeated 20 times, with identical splits for all kernels.
- Each quantum kernel is compared with the RBF kernel, and each re-uploading kernel with the control, by the Nadeau–Bengio corrected t-test, using the mean outer-fold training and test sizes.

## Reproducing the paper

```bash
pip install -r requirements.txt
python prepare_bc.py                      # breast cancer cache (from scikit-learn, no download)
python prepare_hd.py                      # heart disease caches (downloads the ISLR Heart.csv; also writes the one-hot version)
python verify_kernels_qiskit.py           # every kernel vs. Qiskit circuits -> "ALL KERNELS MATCH QISKIT"

python compare_quantum_kernels.py --dataset heart_disease     # ~45 min each on a laptop
python compare_quantum_kernels.py --dataset breast_cancer
python kernel_similarity.py --dataset heart_disease
python kernel_similarity.py --dataset breast_cancer
python entanglement_and_metric.py
python control_numerics_check.py --dataset heart_disease
python control_numerics_check.py --dataset breast_cancer
python partition_summary.py --dataset heart_disease
python partition_summary.py --dataset breast_cancer
python robustness_checks.py matched --dataset heart_disease     # and breast_cancer
python robustness_checks.py cgrid   --dataset heart_disease     # and breast_cancer
for k in 1 2 3 4 5; do python robustness_checks.py subset --dataset heart_disease --seed $k; done   # and breast_cancer
python noise_check.py --dataset heart_disease                   # and breast_cancer (~40 min each)
python boundary_ties.py --dataset heart_disease                 # and breast_cancer
python gaussian_surrogate.py --dataset heart_disease            # and breast_cancer
python compare_quantum_kernels.py --dataset heart_disease_onehot   # encoding sensitivity check
python make_paper_figures.py --out paper
python build_numbers.py --out paper       # numbers.tex and table_*.tex; stops if a statement in the text fails
```

| Paper item | Script | Output |
|---|---|---|
| Tables 2–4 (performance, re-uploading vs. control, selected hyperparameters), Fig. 2, Fig. 3a | `compare_quantum_kernels.py` | `results/compare_<dataset>.json` |
| Table 5 (alignment with classical Gaussian kernels) | `kernel_similarity.py` | `results/similarity_<dataset>.json` |
| Table 5, column g(0); Table 6 (Meyer–Wallach entanglement), Fig. 3b | `entanglement_and_metric.py` | `results/entanglement_and_metric.json` |
| Limitations: sensitivity of the selection to numerically equivalent kernel computations | `control_numerics_check.py` | `results/control_numerics_<dataset>.json` |
| Table 1 (data partitioning) | `partition_summary.py` | `results/partition_<dataset>.json` |
| Table 7 and Sec. 4.5 (robustness: gate-matched controls, other subsets, C up to 1000) | `robustness_checks.py` | `results/matched_<dataset>.json`, `results/subset_seed<k>_<dataset>.json`, `results/cgrid1000_<dataset>.json` |
| Table 7 and Sec. 4.5 (noise: shots, gate errors) | `noise_check.py` | `results/noise_<dataset>.json` |
| Sec. 4.2 (selections on the edge of the grid that tie with a pair inside it) | `boundary_ties.py` | `results/boundary_ties_<dataset>.json` |
| Table 7 and Sec. 4.5 (classical Gaussian kernels built from each circuit's metric g(0), used as classifiers) | `gaussian_surrogate.py` | `results/gaussian_surrogate_<dataset>.json` |
| Table 7 and Sec. 4.5 (heart disease with the nominal variables one-hot encoded) | `compare_quantum_kernels.py --dataset heart_disease_onehot` | `results/compare_heart_disease_onehot.json` |
| Fig. 1–3 | `make_paper_figures.py` | `fig*.pdf` |
| All numbers in the text and all tables | `build_numbers.py` | `numbers.tex`, `table_*.tex` |

| File | Role |
|---|---|
| `compare_quantum_kernels.py` | Kernels (NumPy statevector), nested CV, corrected tests and intervals |
| `kernel_similarity.py` | Centered kernel alignment with isotropic RBF and fitted anisotropic Gaussian kernels |
| `entanglement_and_metric.py` | Fubini–Study metric at x = 0; Meyer–Wallach entanglement at the selected scales |
| `control_numerics_check.py` | Reruns the control kernel's selection with its closed form instead of the statevector |
| `partition_summary.py` | Recreates every data split, records its size, and checks it against the stored results |
| `robustness_checks.py` | Gate-matched controls, other random subsets, and the extended C grid, with the main protocol unchanged |
| `boundary_ties.py` | Recomputes the inner cross-validation score of every grid pair and checks whether edge selections tie with pairs inside the grid |
| `noise_check.py` | Finite-shot and depolarizing-noise versions of the quantum kernels (density-matrix simulation), re-tuned by the same nested protocol; records the smallest eigenvalue of every shot-noise training matrix |
| `gaussian_surrogate.py` | Classical Gaussian kernels exp(−s² dᵀ g(0) d) from each circuit's small-scale metric, tuned and compared by the same protocol |
| `verify_kernels_qiskit.py` | Checks each kernel against Qiskit circuits |
| `make_paper_figures.py`, `build_numbers.py` | Figures; numbers and tables for the manuscript, with checks |
| `foldsafe_pipeline.py`, `nested_hybrid_test.py` | Balanced subset; Nadeau–Bengio test |
| `prepare_bc.py`, `prepare_hd.py` | Data preparation |

## Data

No data is redistributed here.

- **Breast cancer:** Wisconsin Diagnostic Breast Cancer dataset, built into scikit-learn.
- **Heart disease:** UCI Cleveland dataset (https://archive.ics.uci.edu/dataset/45/heart+disease).
  - `prepare_hd.py` downloads the ISLR copy (`Heart.csv`), which keeps the original diagnosis labels.
  - It drops the six records with missing `ca`/`thal`, leaving 297 records, 137 of them with disease.
  - **Do not use the Kaggle `heart.csv` copy.** Its `target` column is inverted relative to UCI, and its missing values are coded as ordinary numbers.

## Environment

Use the exact versions in `requirements.txt` (numpy 2.4.6, scipy 1.15.3, scikit-learn 1.8.0; checked with Python 3.13). With these versions, every script reproduces the stored results in `results/` exactly. `requirements-lock.txt` lists the complete environment, including indirect dependencies, in which the stored results, the figures and the numbers in the paper were produced; with it (`pip install -r requirements-lock.txt`) the figures are also reproduced exactly, whereas other versions of matplotlib's dependencies can shift the figure layout by a few points without changing the plotted values.

Hyperparameter selection by cross-validation is sensitive to near-ties. With other library versions, in particular other versions of scikit-learn, whose SVM solver can return slightly different solutions, a few fold-level selections can change, and with them some numbers in the third decimal place. For example, `results/version_check_<dataset>.json` is `compare_quantum_kernels.py` run with scikit-learn 1.9.1, numpy 2.5.3 and scipy 1.18.1 (versions recorded in the file): 1 of the 1000 fold-level selections differs from `results/compare_<dataset>.json`. To recreate these files, run `compare_quantum_kernels.py` with those library versions, copy `results/compare_<dataset>.json` to `results/version_check_<dataset>.json`, and add a `library_versions` field recording the Python, numpy, scipy and scikit-learn versions. The paper discusses this sensitivity in its limitations.

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).
