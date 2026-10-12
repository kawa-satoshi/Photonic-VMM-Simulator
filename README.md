# Photonic VMM: a small FP32 simulator

Simulate tiled optical matrix multiplication in PyTorch linear layers, measure
causal-LM next-token loss, and run hardware-aware training (HAT). Quantization,
independent X/W/Y noise, and optional dynamic input-feature protection share one
numerical core. This is a floating-point numerical model, not an optical-device,
integer zero-point compensation, or analog clipping simulator.

**Private preparation copy. Publication scope and license are not decided.**
No model weights, datasets, credentials, experiment logs, or original research
repository are included. Paths below are placeholders. No output directory name
changes the arithmetic.

## Layout

```text
photonic_vmm/
  core.py          square tiles, quantization, noise, STE, PhotonicLinear
  protection.py    replaceable feature-selection policy
  convert.py       exact nn.Linear path replacement, shared Parameters retained
  data.py          token JSON, local text/Arrow, HF dataset adapters
  workflows.py     next-token evaluation and short HAT loop
  __init__.py      small public numerical API
evaluate.py        model/data/config CLI
configs/           confirmed defaults and benchmark numerical example
tests/             scalar and gradient references, training and model smoke tests
requirements.txt   core dependency
requirements-hf.txt optional model/data dependencies
```

## Install and test

Use an existing environment when available. Python 3.10+ is required.

```sh
python -m pip install -r requirements.txt
# Only for HF model/tokenizer and HF dataset use:
python -m pip install -r requirements-hf.txt
python -m unittest discover -s tests -v
```

Tested on PyTorch 2.14.1+cpu and 2.6.0+cu118; HF adapters use Transformers
4.50.3 and Datasets 3.6.0. A separate A30 run used the existing PyTorch
2.3.0+cu121 / Transformers 4.40.1 environment without installing dependencies. Core tests need only torch; the random Llama test is
skipped when Transformers is absent. No tests download models or datasets.

## Numerical pipeline

Initial hardware is square G×G: `tile_in == tile_out == group_size == G`.
Tails retain their short physical coordinates; columns are never packed.
All computations and electronic accumulation are FP32. For each input tile K
and output tile N:

1. Apply optional input protection mask, retaining zero-valued physical slots.
2. Quantize/dequantize X and W independently per last-axis row within K.
3. Add independent Gaussian X/W noise, standard deviation = group range × std.
   Re-zero protected X positions after noise so they cannot leak into optics.
4. Compute the optical partial matrix product.
5. Add Y noise based on that partial's output-row range.
6. Apply dynamic min/max ADC Q/DQ **after** Y noise, then add the partial to the
   sequential FP32 electronic accumulator.
7. Add the FP32 protection branch (if enabled), then bias exactly once.
   There is no final output quantization.

`x_bits`, `w_bits`, `y_bits`: null disables Q/DQ; integers 2–16 enable it.
`x_noise`, `w_noise`, `y_noise` independently switch noise on/off. W noise is
redrawn every forward, shared across all tokens within that forward.

Dynamic ADC min/max is the selected specification. Width-one output tiles have
constant range and pass unchanged. Width-two tiles map both values to endpoints,
so Q/DQ introduces no interior error (apart from FP32 rounding). Tests explicitly
cover both cases. Fixed calibrated ADC ranges and saturation are out of scope.

`output_tile_execution` is `spatial_parallel` by default: the same noisy X chunk
is distributed to all output tiles at one instant. `temporal_multiplexed` redraws
X for each output tile's separate time slot. This does not change W sampling:
W still redraws once per forward and is shared over tokens. Mixed spatial/time
groups and a general physical scheduling simulator are not represented.

`constant_range_mode='legacy_abs'` is the confirmed default: when max-min is
zero, noise uses abs(constant) as its range. Thus a zero-valued constant group
still has zero noise, while a nonzero constant group can be noisy. `zero` remains
available only as an explicit alternate comparison. X/W last-axis row ranges
and optional integer-code `clamp_codes` are recorded numerical assumptions.
`metadata()` records the chosen configuration. Stable
centered Q/DQ avoids reciprocal-scale overflow for tiny nonzero ranges.

## HAT and protection

Every Q/DQ or noise transform uses the exact physical value in forward and an
identity straight-through estimator in backward. Range/min/max and noise
amplitudes have no gradient. Matrix multiplication still differentiates using
the transformed X/W operands; the gradient is not generally that of an ideal
whole-layer linear operation. Inference and HAT use the same core; use
`torch.no_grad()` for inference.

`MaxAbsPolicy(threshold=6., aggregation='batch_sequence')` selects a feature
column when max(abs(X)) over all leading dimensions is **strictly greater** than
6. `aggregation='token'` makes a separate selection per token. Threshold and
aggregation are mandatory when enabling the CLI policy. Selection is recomputed
on each forward and has no gradient.

The electronic branch computes `linear(original_X * mask, original_W)` in FP32,
without bias, quantization or noise, and is added to optics. Protected columns
stay in their original tiles. Zero slots participate in optical X ranges; W
ranges are unchanged, including weights at protected coordinates. This explicit
masking convention keeps one W sample shared over tokens even for token masks.
All-protected inputs equal digital FP32; no-protection equals the ordinary core.
Individual weight-coordinate/SuperWeight protection is not implemented.

To replace selection, pass any callable `policy(x) -> bool_tensor` to
`simulate_linear(..., protection=policy)` or `convert(..., protection=policy)`.
Return `[in_features]` or exactly `x.shape` on the same device. Policies receive
a detached tensor; do not mutate it. The numerical core validates masks and does
not depend on the selection algorithm.

## OPT inference and HAT examples

The CLI is offline by default, uses FP32/eager attention with TF32 disabled,
and never enables remote model code. The standard target set is **all named
nn.Linear layers, including a linear lm_head**. In OPT-125M this is 73 layers.
Use explicit targets/exclusions to change that scope. List paths before use:

```sh
python evaluate.py --model /models/opt-125m --list-targets
```

Create `targets.json` as a reviewed JSON array of exact paths, for example:

```json
["model.decoder.layers.0.self_attn.q_proj", "model.decoder.layers.0.fc2", "lm_head"]
```

This file is an optional three-layer subset example; omit `--targets-file` to
use the full standard scope. `--exclude-file excluded.json` subtracts exact
paths from either the standard scope or your explicit target list. For example,
`["lm_head"]` explicitly excludes the head. Embedding/head shared Parameters
stay shared; training a tied head also updates its embedding use.

```sh
python evaluate.py --model /models/opt-125m --tokens-file tokens.json \
  --digital --output results/digital.json
python evaluate.py --model /models/opt-125m --tokens-file tokens.json \
  --config configs/default.json \
  --protect-threshold 6 --protect-aggregation batch_sequence \
  --output results/protected.json
python evaluate.py --model /models/opt-125m --tokens-file tokens.json \
  --config configs/default.json \
  --protect-threshold 6 --protect-aggregation batch_sequence \
  --max-batches 8 --validation-tokens-file heldout.json --validation-max-batches 32 \
  --hat-steps 16 --learning-rate 0.0001 --seed 42 \
  --checkpoint results/hat.pt --output results/hat.json
```

Token JSON is an array of unpadded ID rows, each length ≥2. Default evaluation is
two rows at batch size one. HAT freezes unselected Parameters and uses AdamW on
selected weights/biases, one row per step in cyclic order. Dropout follows model
train/eval mode. `--validation-tokens-file` selects separate evaluation rows;
otherwise HAT metrics reuse training rows and measure fitting only. Duplicate train/evaluation
rows are rejected; article/document-level separation remains the caller's
responsibility. Use disjoint source documents for held-out evaluation.

Before/after evaluation replays `--evaluation-noise-index` (default 1000000),
while training starts at index zero. This pairs optical draws across comparisons;
report the same seed set and data across conditions. Training dropout follows
`--seed`, independently of the optical config seed.

The checkpoint contains state_dict only. Use `--load-checkpoint results/hat.pt`
with the same model, numerical config, targets and protection to evaluate it on
another token/text/Arrow source. Loading uses `weights_only=True` and strict
state-dict validation. Keep the associated metrics/config alongside it: a
state_dict cannot identify the original simulator settings. Full optimizer/RNG
resume is not implemented. Results/checkpoints refuse overwrite.

## Change model or data

`--model` accepts a local directory or a cached HF ID. Models must implement the
HF causal-LM `input_ids`, `use_cache=False`, and `.logits` contract. Only actual
`torch.nn.Linear` modules are replaced; GPT-2 Conv1D, fused/quantized/custom
projections need a separate adapter. Arbitrary architectures are not certified.
Aliased Linear module instances (the same module under multiple paths) are
rejected before conversion to avoid silently leaving an alias electronic.
Distinct modules sharing a Parameter are supported. Inspect the replacement
manifest in the output.

Replace the token input with local text:

```sh
python evaluate.py --model /models/your-model --text-file corpus.txt \
  --sequence-length 128 --max-batches 10 --digital --output results/text.json
```

A local Arrow cache shard can be loaded directly without Hub metadata:

```sh
python evaluate.py --model /models/your-model --arrow-file corpus.arrow \
  --text-column text --sequence-length 128 --max-batches 10 \
  --digital --output results/arrow.json
```

This supports a single Hugging Face Arrow shard. Use the dataset adapter or
prepare token JSON for multi-shard/custom formats. Or use a previously cached
HF dataset:

```sh
python evaluate.py --model /models/your-model --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 --split validation --text-column text \
  --sequence-length 128 --max-batches 10 --digital --output results/data.json
```

Only explicit `--allow-download` permits fetching missing model/data files.
Text is tokenized without added special tokens, chunked independently per
document without padding; one-token tails are discarded. Local text is one
document. No cross-row next-token targets or sliding contexts are evaluated.
Loss is weighted by predicted token count. This is a small evaluation protocol,
not a standard benchmark reproduction. `data.py` is the single data extension
point; pass token rows directly to `evaluate_batches` for other loaders.

## Reproducibility and validation limits

CLI `--seed` controls model-side randomness; the config's `seed` separately
controls optical noise. Layer/stage/forward/tile keys isolate noise streams.
`PhotonicLinear.reset_noise()` resets forward numbering. Evaluation advances
counters internally; the CLI resets to its explicit evaluation index before
each comparison, so pre/post HAT scores use paired draws. Same-seed sequences
are not guaranteed across device/PyTorch versions or changed tensor shapes.
Results record data-row hash, model config, targets, numerical/protection config,
seed, torch version, FP32/TF32 settings, and training gradients/losses.

Validation: 41 tests cover the numerical core, default head scope and
exclusions, module-alias rejection, real Arrow input, random Llama structure,
and end-to-end held-out HAT/checkpoint reload with replayed optical noise.
Optional dependency tests skip when Transformers or Datasets is absent.
The local-text CLI and direct Arrow adapter passed on existing cached data.
HF dataset-name resolution requires matching Hub/cache metadata; direct Arrow
loading avoids that dependency. Missing metadata fails closed offline.

The final OPT benchmark uses the confirmed abs(constant) fallback and all 73
linear layers including lm_head. Training and held-out data come from disjoint
articles in the existing WikiText validation cache (a custom evaluation split,
not the official WikiText benchmark protocol). Each row contains the first 128
tokens of a qualifying text row: 8 training rows / 32 evaluation rows, totaling
1,016 unique training targets / 4,064 held-out targets. Three runs use seeds
42/43/44 for model/dropout and optical noise. Each HAT run takes 16 AdamW steps
(two passes through the eight rows), learning rate 1e-4, weight decay zero.
All comparisons use batch one, FP32, TF32 off, no offload, G=256, 8-bit X/W/Y,
noise std 0.01, and optional threshold-6 batch_sequence protection. Evaluation
starts at optical forward index 1000000 for paired draws; training starts at 0.
Final next-token cross entropy (natural log; lower is better):

| Condition | Seed 42 | Seed 43 | Seed 44 |
| --- | ---: | ---: | ---: |
| Digital baseline | 3.62610446 | same | same |
| No-error tiled | 3.62610444 | same | same |
| Quantization only | 3.63247921 | 3.63247921 | 3.63247921 |
| Quantization + noise | 4.28814346 | 4.26635521 | 4.28594564 |
| + feature protection | 3.76551345 | 3.76697839 | 3.76817460 |
| + 16-step HAT | 4.03192555 | 4.03953559 | 4.04133221 |

The no-error logit probe on the first 128-token held-out row had maximum absolute
difference 2.67e-5 (allclose atol/rtol 1e-4). The full 4,064-target loss agrees as
shown above. Total runtime was 618.55 seconds on one A30, with peak torch CUDA
allocated memory 3.15 GB, in the existing torch 2.3.0+cu121 / Transformers 4.40.1
environment. No model or dataset was downloaded for this experiment.

Protection mitigated noise in all three runs. HAT reduced mean training loss
between the first and second eight-step passes (about 4.16–4.21 to 2.55–2.61),
but **worsened held-out loss relative to protection alone** in every run. This
setup does not demonstrate generalization gains. Earlier zero-range/head-excluded
smoke results are separate experiments and not the final specification.

CPU validation covers independent scalar Q/DQ/multiply-add, independent analytic
STE gradients, physical mask placement, all/no protection, independent noise
switches, per-forward W reuse, tails, shared weights, nonfinite inputs, empty
batches, tiny ranges, bias, ADC order, and short synthetic learning. A random tiny
Llama tests structural conversion and tied weights without downloads; it is not
a pretrained Llama quality benchmark. Pretrained OPT-125M no-error comparison
and short HAT fitting are separate private smoke experiments. The small held-out
code benchmark does not establish general LLM quality or an optimal HAT setup. Broad quality/noise sweeps, calibrated hardware
agreement, and distributed/mixed precision training remain unvalidated.
