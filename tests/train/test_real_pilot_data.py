# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# http://www.apache.org/licenses/LICENSE-2.0
# ruff: noqa: E402

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_real_conversion_pilot import (
    BUCKETS,
    DEFAULT_TOKENIZER,
    HASH_KEYS,
    boundary_error,
    bucket_for,
    choose,
    digest,
    identities,
    tokenizer_fingerprint,
    write_dataset,
)


class CharacterTokenizer:
    name_or_path = "test-character"
    vocab_size = 65536

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {"a": 97, "b": 98}

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False
        assert truncation is False
        return [ord(char) for char in text]


def row(key, length=32, question=None):
    response = key + "x" * (length - len(key) - len("<turn|>")) + "<turn|>"
    assert len(response) == length
    return {"prompt": f"<bos><|turn>user\n{question or key}<turn|>\n<|turn>model\n", "response": response}


def source(tmp_path, name, rows):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def sources(tmp_path):
    train = source(tmp_path, "train", [row(f"train{i}", low) for i, (low, _) in enumerate(BUCKETS)])
    valid = source(tmp_path, "valid", [row(f"valid{i}", high) for i, (_, high) in enumerate(BUCKETS)])
    return train, valid


def test_normalization_and_all_identity_dimensions():
    assert digest("Ａ b\nＣ") == digest("abc")
    original = row("original", 64, question="Ｑ u e s t i o n")
    duplicate = row("different", 64, question="question")
    assert identities(original)["question_hash"] == identities(duplicate)["question_hash"]
    assert identities(original)["prompt_hash"] == identities(duplicate)["prompt_hash"]
    duplicate["prompt"] = duplicate["prompt"].replace("<bos>", "<bos><|turn>system\ncontext<turn|>\n")
    assert identities(original)["prompt_hash"] != identities(duplicate)["prompt_hash"]
    assert identities(original)["question_hash"] == identities(duplicate)["question_hash"]


@pytest.mark.parametrize(
    "length,bucket",
    [(31, None), (32, 0), (256, 0), (257, 1), (512, 1), (513, 2), (1024, 2), (1025, 3), (4608, 3), (4609, None)],
)
def test_exact_bucket_boundaries(length, bucket):
    assert bucket_for(length) == bucket


def test_valid_first_and_within_split_dedup(tmp_path):
    valid_rows = [row("valid0", 32), row("duplicate-question", 257, question="VALID0")]
    valid_rows += [row(f"valid{i}", low) for i, (low, _) in enumerate(BUCKETS) if i]
    valid = source(tmp_path, "valid", valid_rows)
    duplicate_prompt = row("other-response", 32, question="valid0")
    duplicate_question = row("different-context", 32, question="valid0")
    duplicate_question["prompt"] = duplicate_question["prompt"].replace(
        "<bos>", "<bos><|turn>system\ncontext<turn|>\n"
    )
    duplicate_response = row("other-question", 32)
    duplicate_response["response"] = valid_rows[0]["response"].removesuffix("<turn|>").upper() + "<turn|>"
    train_rows = [duplicate_prompt, duplicate_question, duplicate_response]
    train_rows += [row(f"train{i}", low) for i, (low, _) in enumerate(BUCKETS)]
    train = source(tmp_path, "train", train_rows)
    manifest = write_dataset(train, valid, CharacterTokenizer(), tmp_path / "out", 1, 1)
    assert manifest["splits"]["valid"]["skipped"]["normalized_duplicate"] == 1
    assert manifest["splits"]["train"]["skipped"]["normalized_duplicate"] == 3
    records = [
        json.loads(line)
        for split in ("train", "valid")
        for line in (tmp_path / "out" / f"{split}.jsonl").read_text().splitlines()
    ]
    for key in HASH_KEYS:
        assert len({r[key] for r in records}) == 8


def test_boundaries_prompt_cap_and_no_truncation(tmp_path):
    over = row("over", 32, question="q" * 4096)
    exact = row("exact", 32)
    exact["prompt"] = exact["prompt"].replace("exact", "q" * (4096 - len(exact["prompt"]) + 5))
    assert len(exact["prompt"]) == 4096
    rows = [row("short", 31), row("long", 4609), over, exact]
    rows += [row(f"boundary{i}", high) for i, (_, high) in enumerate(BUCKETS) if i]
    path = source(tmp_path, "src", rows)
    selected, skipped = choose(path, CharacterTokenizer(), 1, {k: set() for k in HASH_KEYS})
    assert skipped["response_out_of_range"] == 2
    assert skipped["prompt_over_cap"] == 1
    assert selected[0]["prompt_length"] == 4096
    for record in selected:
        assert "".join(map(chr, record["prompt_ids"])) == record["prompt"]
        assert "".join(map(chr, record["response_ids"])) == record["response"]
        assert record["response_length"] == len(record["response_ids"])
    assert selected[-1]["response_length"] == 4608


@pytest.mark.parametrize(
    "prompt,response",
    [
        ("<|turn>user\nq<turn|>\n", "answer<turn|>"),
        ("<|turn>user\nq<|turn>model\n", "answer<turn|>"),
        (row("x")["prompt"], "answer"),
        (row("x")["prompt"], "answer<turn|>extra<turn|>"),
        (row("x")["prompt"], "<|turn>model\nanswer<turn|>"),
        (row("x")["prompt"], "<|channel>thought\nunfinished<turn|>"),
        (row("x")["prompt"], "<channel|>before<|channel>thought\nunfinished<turn|>"),
        (row("x")["prompt"], "<|channel>thought\na<|channel>thought\nb<channel|>answer<turn|>"),
    ],
)
def test_malformed_boundaries_and_thought_rejected(prompt, response):
    assert boundary_error(prompt, response) is not None


def test_closed_thought_accepted():
    assert boundary_error(row("x")["prompt"], "<|channel>thought\nreason<channel|>answer<turn|>") is None


def test_insufficient_source_creates_no_output(tmp_path):
    train, valid = sources(tmp_path)
    train.write_text("", encoding="utf-8")
    out = tmp_path / "missing-parent" / "out"
    with pytest.raises(ValueError, match="Insufficient"):
        write_dataset(train, valid, CharacterTokenizer(), out, 1, 1)
    assert not out.parent.exists()


@pytest.mark.parametrize("kind", ["file", "empty_directory", "nonempty_directory", "symlink"])
def test_no_overwrite(tmp_path, kind):
    out = tmp_path / "out"
    if kind == "file":
        out.write_text("keep")
    elif kind == "symlink":
        out.symlink_to(tmp_path / "absent")
    else:
        out.mkdir()
        if kind == "nonempty_directory":
            (out / "keep").write_text("keep")
    with pytest.raises(FileExistsError, match="Refusing"):
        write_dataset(tmp_path / "absent", tmp_path / "absent", CharacterTokenizer(), out, 1, 1)
    if kind == "file":
        assert out.read_text() == "keep"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_per": 0},
        {"valid_per": -1},
        {"prompt_cap": 0},
        {"prompt_cap": 4097},
        {"C": 0},
        {"K": -1},
        {"K": 17},
        {"train_per": True},
    ],
)
def test_invalid_arguments_and_budget(tmp_path, kwargs):
    with pytest.raises(ValueError):
        write_dataset(tmp_path / "absent", tmp_path / "absent", CharacterTokenizer(), tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_reproducible_output_manifest_and_fingerprint(tmp_path):
    train, valid = sources(tmp_path)
    tok = CharacterTokenizer()
    first = write_dataset(train, valid, tok, tmp_path / "a", 1, 1)
    second = write_dataset(train, valid, tok, tmp_path / "b", 1, 1)
    assert first == second
    for name in ("train.jsonl", "valid.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    for split, metadata in first["splits"].items():
        content = (tmp_path / "a" / f"{split}.jsonl").read_bytes()
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
        assert metadata["rows"] == 4
        assert metadata["bucket_counts"] == {0: 1, 1: 1, 2: 1, 3: 1}
    assert first["splits"]["valid"]["response_blocks_max"] == 18
    assert "{" not in DEFAULT_TOKENIZER
    fallback = tokenizer_fingerprint(tok)
    assert fallback["fingerprint_method"] == "get_vocab"
    tok.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"b":2,"a":1}')
    backend = tokenizer_fingerprint(tok)
    tok.backend_tokenizer = SimpleNamespace(to_str=lambda: '{ "a": 1, "b": 2 }')
    assert tokenizer_fingerprint(tok) == backend
    assert backend["sha256"] == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    tok.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"a":2,"b":2}')
    assert tokenizer_fingerprint(tok)["sha256"] != backend["sha256"]
