"""BERT and a small visual encoder jointly predict card type and mark."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import save_file, load_file
from .ingest import render, ocr, extract, fingerprint

BASE = "google/bert_uncased_L-2_H-128_A-2"
REV = "30b0a37ccaaa32f332884b96992754e246e48c5f"


def pixels(image):
    return (
        torch.tensor(np.array(image.crop((426, 36, 508, 118)).resize((20, 20))), dtype=torch.float32).view(
            1, 20, 20
        )
        / 255
    )


class Reader(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.text = AutoModel.from_pretrained(source, revision=REV, use_safetensors=True)
        self.visual = nn.Sequential(
            nn.Conv2d(1, 4, 3), nn.ReLU(), nn.AvgPool2d(3), nn.Flatten(), nn.Linear(144, 16), nn.ReLU()
        )
        self.head = nn.Linear(16 + self.text.config.hidden_size, 4)

    def forward(self, ids, mask, views):
        words = self.text(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([words, self.visual(views)], dim=1)).reshape(-1, 2, 2)


def tokens(tokenizer, words):
    batch = tokenizer(words, padding=True, truncation=True, max_length=64, return_tensors="pt")
    return batch.input_ids, batch.attention_mask


def decode_heads(heads):
    conf, index = heads.softmax(-1).max(-1)
    return ("inspection", "service")[index[0]], ("open", "filled")[index[1]], float(min(conf))


def read_one(model, tokenizer, image):
    words = ocr(image)
    with torch.no_grad():
        pred = model(*tokens(tokenizer, [words]), pixels(image)[None])[0]
    kind, mark, conf = decode_heads(pred)
    data = extract(words, kind, mark, conf)
    return dict(
        source_sha256=fingerprint(image),
        attributes=data,
        confidence=conf,
        disposition="review" if data is None else "extracted",
    )


def load_reader(weights: Path, source):
    model = Reader(source)
    model.load_state_dict(load_file(str(weights)))
    return model.eval(), AutoTokenizer.from_pretrained(source, revision=REV)


def run(source, out):
    torch.manual_seed(220)
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=REV)
    model = Reader(source)
    words = []
    views = []
    targets = []
    truth = []
    hashes = []
    for group in range(32):
        for kind in range(2):
            for mark in range(2):
                img, fields = render(group, kind, mark)
                words.append(ocr(img))
                views.append(pixels(img))
                targets.append([kind, mark])
                truth.append(fields)
                hashes.append(fingerprint(img))
    ids, mask = tokens(tokenizer, words)
    x = torch.stack(views)
    y = torch.tensor(targets)
    n = 96
    opt = torch.optim.AdamW(
        [
            {"params": model.text.parameters(), "lr": 0.0004},
            {"params": model.head.parameters()},
            {"params": model.visual.parameters()},
        ],
        lr=0.005,
    )
    for _ in range(8):
        for batch in torch.randperm(n).split(8):
            logits = model(ids[batch], mask[batch], x[batch])
            opt.zero_grad()
            loss = nn.functional.cross_entropy(logits.reshape(-1, 2), y[batch].reshape(-1))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(ids[n:], mask[n:], x[n:])
    cases = []
    for i, row in enumerate(logits, n):
        kind, mark, conf = decode_heads(row)
        result = extract(words[i], kind, mark, conf)
        cases.append(
            dict(
                asset_group=i // 4,
                sha256=hashes[i],
                truth=truth[i],
                attributes=result,
                confidence=conf,
                text_match=kind == truth[i]["document_type"],
                mark_match=mark == truth[i]["observed_mark"],
            )
        )
    accepted = [r for r in cases if r["attributes"] is not None]
    report = dict(
        revision=REV,
        seed=220,
        epochs=8,
        train=n,
        test=len(cases),
        accepted=len(accepted),
        threshold=0.8,
        exact_correct=sum(r["attributes"] == r["truth"] for r in cases),
        cases=cases,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save_file(model.state_dict(), str(out / "reader.safetensors"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    args = p.parse_args()
    run(os.getenv("BERT_PATH", BASE), args.out)
