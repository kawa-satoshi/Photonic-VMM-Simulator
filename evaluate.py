"""Offline-first FP32 causal-LM evaluation and short HAT experiments."""
import argparse
import json
from importlib.metadata import version
import os
from pathlib import Path
import time
import torch
from photonic_vmm import VMMConfig
from photonic_vmm.convert import convert
from photonic_vmm.data import load_rows
from photonic_vmm.workflows import evaluate_batches, reset_noise, seed_everything, train_hat


def read_paths(filename):
    paths = json.loads(Path(filename).read_text())
    if (not isinstance(paths, list) or not paths
            or not all(isinstance(path, str) and path for path in paths)
            or len(paths) != len(set(paths))):
        raise ValueError('Layer file must be a nonempty JSON array of unique exact module paths')
    return paths


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local model directory or HF model ID')
    p.add_argument('--allow-download', action='store_true', help='Explicitly allow model/data downloads')
    p.add_argument('--config', help='VMM JSON config')
    p.add_argument('--targets-file', help='Exact nn.Linear paths; default: all named Linear layers including lm_head')
    p.add_argument('--exclude-file', help='JSON array to subtract from explicit targets')
    p.add_argument('--tokens-file', help='JSON array of unpadded token-ID rows')
    p.add_argument('--text-file', help='Local UTF-8 text, chunked without padding')
    p.add_argument('--arrow-file', help='Single local HF Arrow shard; no dataset metadata lookup')
    p.add_argument('--dataset', help='HF dataset ID (cached unless downloads enabled)')
    p.add_argument('--dataset-config')
    p.add_argument('--split', default='test')
    p.add_argument('--text-column', default='text')
    p.add_argument('--sequence-length', type=int, default=128)
    p.add_argument('--max-batches', type=int, default=2)
    p.add_argument('--output', help='New metrics JSON path')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--digital', action='store_true')
    p.add_argument('--list-targets', action='store_true')
    p.add_argument('--protect-threshold', type=float)
    p.add_argument('--protect-aggregation', choices=['batch_sequence', 'token'])
    p.add_argument('--hat-steps', type=int, default=0)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--validation-tokens-file', help='Separate token JSON rows for HAT evaluation')
    p.add_argument('--validation-max-batches', type=int, default=2)
    p.add_argument('--evaluation-noise-index', type=int, default=1000000,
                   help='Replay this forward index before each evaluation; training starts at zero')
    p.add_argument('--load-checkpoint', help='Local state_dict; same model/config/targets required')
    p.add_argument('--checkpoint', help='New local HAT state_dict path; reload after identical conversion')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.max_batches < 1 or args.sequence_length < 2 or args.hat_steps < 0:
        p.error('max-batches >= 1, sequence-length >= 2, hat-steps >= 0 required')
    if args.validation_max_batches < 1 or args.evaluation_noise_index < 0:
        p.error('validation-max-batches >= 1 and evaluation-noise-index >= 0 required')
    if args.validation_tokens_file and not args.hat_steps:
        p.error('validation-tokens-file is for HAT; otherwise use tokens-file')
    protection_enabled = args.protect_threshold is not None
    if protection_enabled != (args.protect_aggregation is not None):
        p.error('protection requires both threshold and aggregation')
    if args.digital and (args.hat_steps or protection_enabled or args.config or args.targets_file or args.exclude_file):
        p.error('digital mode does not accept analog, target, protection, or HAT settings')
    if args.checkpoint and not args.hat_steps:
        p.error('checkpoint requires hat-steps > 0')
    if not args.list_targets:
        if not args.output:
            p.error('output is required')
        if sum(v is not None for v in (args.tokens_file, args.text_file, args.arrow_file, args.dataset)) != 1:
            p.error('select exactly one token/text/arrow/dataset source')
        if not args.digital and not args.config:
            p.error('analog evaluation/HAT requires config')
        for filename in (args.output, args.checkpoint):
            if filename and Path(filename).exists():
                p.error(f'Refusing to overwrite: {filename}')
        if args.checkpoint and Path(args.checkpoint).resolve() == Path(args.output).resolve():
            p.error('checkpoint and output must differ')
    if not args.allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    seed_everything(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=not args.allow_download, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation='eager').to(args.device)
    if args.list_targets:
        candidates = [dict(path=name, in_features=layer.in_features, out_features=layer.out_features)
                      for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
        print(json.dumps(candidates, indent=2))
        return
    manifest, numerical_model, targets = [], None, []
    if not args.digital:
        config = VMMConfig(**json.loads(Path(args.config).read_text()))
        targets = read_paths(args.targets_file) if args.targets_file else None
        excludes = read_paths(args.exclude_file) if args.exclude_file else []
        protection = None
        if protection_enabled:
            from photonic_vmm.protection import MaxAbsPolicy
            protection = MaxAbsPolicy(threshold=args.protect_threshold, aggregation=args.protect_aggregation)
        manifest = convert(model, config, targets, protection=protection, exclude_paths=excludes)
        targets = [entry['path'] for entry in manifest]
        numerical_model = config.metadata()
    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state, strict=True)
    tokenizer = None
    if not args.tokens_file:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download,
                                                  trust_remote_code=False)
    rows, data_info = load_rows(
        tokens_file=args.tokens_file, text_file=args.text_file, arrow_file=args.arrow_file,
        dataset=args.dataset,
        dataset_config=args.dataset_config, split=args.split, text_column=args.text_column,
        tokenizer=tokenizer, sequence_length=args.sequence_length, max_rows=args.max_batches,
        allow_download=args.allow_download)
    evaluation_rows, evaluation_data = rows, data_info
    if args.validation_tokens_file:
        evaluation_rows, evaluation_data = load_rows(tokens_file=args.validation_tokens_file,
                                                      max_rows=args.validation_max_batches)
        if set(map(tuple, rows)) & set(map(tuple, evaluation_rows)):
            p.error('training and validation contain an identical token row')
    start = time.monotonic()
    before = evaluate_batches(model, evaluation_rows, args.device,
                              noise_index=args.evaluation_noise_index)
    training = None
    metrics = before
    if args.hat_steps:
        reset_noise(model, 0)
        training = train_hat(model, rows, args.device, targets, steps=args.hat_steps,
                             learning_rate=args.learning_rate)
        metrics = evaluate_batches(model, evaluation_rows, args.device,
                                   noise_index=args.evaluation_noise_index)
        if args.checkpoint:
            checkpoint = Path(args.checkpoint)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open('xb') as stream:
                torch.save(model.state_dict(), stream)
    output = dict(metrics=metrics, before_training=before if training else None, training=training,
                  elapsed_seconds=time.monotonic() - start, digital=args.digital,
                  numerical_model=numerical_model, replaced_layers=manifest,
                  protection=protection.metadata() if protection_enabled else None,
                  model=args.model, model_config=model.config.to_dict(), data=data_info,
                  evaluation_data=evaluation_data, evaluation_noise_index=args.evaluation_noise_index,
                  checkpoint_loaded=bool(args.load_checkpoint),
                  seed=args.seed, torch_version=torch.__version__,
                  transformers_version=version('transformers'),
                  datasets_version=version('datasets') if args.dataset or args.arrow_file else None, device=args.device,
                  tf32=False, weight_dtype='float32', use_cache=False,
                  note=('Separate evaluation rows; document-level separation is the caller responsibility.'
                        if args.validation_tokens_file else 'HAT metrics reuse training rows; not held-out validation.')
                        if training else None)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
