# Watermark-based Attribution of AI-Generated Content

This repository contains a focused, easy-to-run release for the paper
[Watermark-based Attribution of AI-Generated Content](https://arxiv.org/abs/2404.04254)
(ICLR 2026).

The release covers:

- the theoretical TDR, FDR, and TAR bounds (Theorems 1–4);
- Random, Non-Redundant Guess (NRG), and A-BSTA watermark selection;
- detection and attribution metrics; and
- an optional inference smoke test for our 64-bit HiDDeN model.

It intentionally does **not** include experimental outputs, large watermark
pools, datasets, or model weights.

## Quick start

The core examples require Python 3.10 or newer and run on CPU.

```bash
git clone https://github.com/zhengyuan-jiang/Watermark-Attribution.git
cd Watermark-Attribution

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python examples/theoretical_bounds.py
python examples/selection_demo.py
```

Both scripts provide command-line help:

```bash
python examples/theoretical_bounds.py --help
python examples/selection_demo.py --help
```

## Theoretical bounds

`watermark_attribution/theory.py` implements:

- `tdr_lower_bound`: Theorem 1;
- `fdr_upper_bound_random`: Theorem 2;
- `fdr_upper_bound_independent`: Theorem 3; and
- `tar_lower_bound`: Theorem 4.

The default command uses the setting from Table 5: 100 million users,
64-bit watermarks, `beta=0.99`, `gamma=0.05`, `tau=0.9`,
`alpha_min=0.2`, and `alpha_max=0.8`.

```bash
python examples/theoretical_bounds.py
```

Parameters can be changed directly:

```bash
python examples/theoretical_bounds.py \
  --users 100000 \
  --length 64 \
  --beta 0.99 \
  --gamma 0.05 \
  --tau 0.9 \
  --alpha-min 0.2 \
  --alpha-max 0.8
```

## Watermark selection

The selection demo compares the three methods used in the paper:

```bash
python examples/selection_demo.py
```

The default creates 25 watermarks so that all methods finish quickly on a
laptop. To run one method or change the pool size:

```bash
python examples/selection_demo.py --method absta --users 50 --seed 0
python examples/selection_demo.py --method nrg --users 100
python examples/selection_demo.py --method random --users 1000
```

A-BSTA uses the paper's uniformly random initialization and recursion depth
8 by default. NRG uses the complement of the first watermark for
initialization. All methods accept a seed and return unique binary
watermarks.

Paper-scale pools contain up to one million watermarks and require much more
time and memory. This repository provides the algorithms but does not bundle
those generated pools.

## Use from Python

```python
from watermark_attribution import (
    maximum_pairwise_accuracy,
    select_watermarks,
    tar_lower_bound,
)

pool = select_watermarks("absta", num_users=25, n=64, seed=0)
print(maximum_pairwise_accuracy(pool))

bound = tar_lower_bound(n=64, beta=0.99, tau=0.9, alpha_max=0.8)
print(bound)
```

`watermark_attribution/metrics.py` also provides functions for matching
decoded watermarks to users and calculating empirical TDR, TAR, and FDR.
Detection uses the rule in the paper: maximum bitwise accuracy greater than
or equal to `tau`.

## Optional HiDDeN checkpoint demo

The repository includes only the inference architecture. The trained
checkpoint `64bitsRes099.pth` is not distributed in Git.

Expected checkpoint SHA-256:

```text
1eadaefee4f4a383cae8b55c37094d6952461d894a0899e7f8034e2f0d88ee49
```

Install the optional dependencies and pass a local checkpoint:

```bash
pip install -r requirements-hidden.txt

python examples/hidden_demo.py \
  --checkpoint /path/to/64bitsRes099.pth
```

By default, the script uses a deterministic synthetic image to verify that
the checkpoint loads and the encoder/decoder execute. This is only a smoke
test, not a performance evaluation. A natural image can be supplied for a
more representative check:

```bash
python examples/hidden_demo.py \
  --checkpoint /path/to/64bitsRes099.pth \
  --image /path/to/image.jpg \
  --device auto
```

## Reproduction scope

This compact release supports:

- direct calculation of the published theoretical bounds;
- reproduction of the Table 5 parameter setting;
- small-scale, seeded comparison of Random, NRG, and A-BSTA; and
- optional HiDDeN encoder/decoder inference.

It does not include the paper's trained StegaStamp or PRC models, image
datasets, adversarial attacks, robustness experiments, paper-scale generated
watermark pools, or precomputed figures. These omissions are deliberate so
the repository remains small and understandable.

## Tests

Run the CPU-only tests from the repository root:

```bash
python -m unittest discover -s tests
```

The tests cover theorem thresholds and Table 5 values, attribution metrics,
selection determinism, uniqueness, and the BSTA match constraint.

## Repository structure

```text
watermark_attribution/
  theory.py       # Theorems 1–4
  metrics.py      # TDR, FDR, TAR, and bitwise accuracy
  selection.py    # Random, NRG, and A-BSTA
  hidden.py       # Optional HiDDeN inference-only model
examples/
  theoretical_bounds.py
  selection_demo.py
  hidden_demo.py
tests/
  test_core.py
```

## Citation

```bibtex
@inproceedings{jiang2026watermark,
  title={Watermark-based Attribution of AI-Generated Content},
  author={Jiang, Zhengyuan and Guo, Moyang and Hu, Yuepeng and
          Wang, Yupu and Gong, Neil Zhenqiang},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## License

The new code is released under the MIT License. The optional HiDDeN
inference implementation contains MIT-licensed adaptations; see
`THIRD_PARTY_NOTICES.md` for the original notices.
