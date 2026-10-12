"""Unpadded token rows from JSON, local text/Arrow, or a Hugging Face dataset."""
import hashlib
import json
from pathlib import Path


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty list of token rows')
    for row in rows:
        if (not isinstance(row, list) or len(row) < 2
                or any(type(token) is not int or token < 0 for token in row)):
            raise ValueError('Each token row needs at least two nonnegative integer IDs')
    return rows


def text_rows(texts, tokenizer, sequence_length, max_rows):
    """Chunk each document separately; discard one-token tails, without padding."""
    if sequence_length < 2 or max_rows < 1:
        raise ValueError('sequence_length >= 2 and max_rows >= 1 are required')
    rows = []
    for text in texts:
        if not isinstance(text, str):
            raise ValueError('The selected text field must contain strings')
        tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        for start in range(0, len(tokens), sequence_length):
            row = tokens[start:start + sequence_length]
            if len(row) >= 2:
                rows.append(row)
                if len(rows) == max_rows:
                    return validate_rows(rows)
    return validate_rows(rows)


def load_rows(*, tokens_file=None, text_file=None, arrow_file=None, dataset=None,
              dataset_config=None, split='test', text_column='text',
              tokenizer=None, sequence_length=128, max_rows=2,
              allow_download=False):
    if sum(value is not None for value in (tokens_file, text_file, arrow_file, dataset)) != 1:
        raise ValueError('Choose exactly one of tokens_file, text_file, arrow_file, or dataset')
    if max_rows < 1:
        raise ValueError('max_rows must be positive')
    if tokens_file is not None:
        rows = validate_rows(json.loads(Path(tokens_file).read_text()))[:max_rows]
        source = {'kind': 'token_json'}
    else:
        if tokenizer is None:
            raise ValueError('Text sources require a tokenizer')
        if text_file is not None:
            texts = [Path(text_file).read_text()]
            source = {'kind': 'local_text'}
        elif arrow_file is not None:
            from datasets import Dataset
            # Read a single local HF Arrow shard directly; no Hub/cache lookup.
            data = Dataset.from_file(str(arrow_file))
            if text_column not in data.column_names:
                raise ValueError(f'Missing dataset text column: {text_column}')
            texts = (row[text_column] for row in data)
            source = {'kind': 'local_arrow', 'text_column': text_column}
        else:
            from datasets import DownloadConfig, config, load_dataset
            # Dataset metadata lookups must also stay offline, including library use.
            previous_offline = config.HF_DATASETS_OFFLINE
            try:
                if not allow_download:
                    config.HF_DATASETS_OFFLINE = True
                data = load_dataset(dataset, name=dataset_config, split=split,
                                    download_config=DownloadConfig(local_files_only=not allow_download))
            finally:
                config.HF_DATASETS_OFFLINE = previous_offline
            if text_column not in data.column_names:
                raise ValueError(f'Missing dataset text column: {text_column}')
            texts = (row[text_column] for row in data)
            source = {'kind': 'hf_dataset', 'name': dataset, 'config': dataset_config,
                      'split': split, 'text_column': text_column}
        rows = text_rows(texts, tokenizer, sequence_length, max_rows)
    digest = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return rows, dict(source, rows_sha256=digest, rows=len(rows),
                     sequence_length=sequence_length if tokens_file is None else None,
                     protocol='batch=1; independent rows; no padding; next-token targets')
