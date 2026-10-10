"""TagAI command line interface.

    python -m tagai.cli <stage> [options]

Stages run in order and each one is independently re-runnable:

    info        show hardware and resolved configuration
    sources     list candidate corpora and their licences
    fetch       download a wiki dump into runs/<root>/data/raw (resumable)
    sample      fetch a small slice for pipeline validation
    clean       build the cleaned corpus (download -> clean -> dedupe -> split)
    report      dataset quality + provenance report
    tokenize    train the SentencePiece tokenizer
    pack        tokenize + pack the corpus into uint16 shards
    train       pretrain from randomly initialized weights
    generate    sample text from the newest checkpoint
    evaluate    run the Azerbaijani evaluation suite
    export      package model + tokenizer + reports
    all         clean -> report -> tokenize -> pack -> train (small preset)

Every stage prints the real measured numbers. Nothing is estimated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

from .config import PathsConfig, build_config, load_config_file
from .utils import atomic_write_json, configure_stdio, estimate_device_memory, human_bytes, read_json, setup_logging

DEFAULT_CONFIG = Path("configs/tagai_small.json")


def load_cfg(args: argparse.Namespace) -> dict:
    raw = load_config_file(args.config) if args.config else {}
    for section, keys in (("data", ("sources", "max_documents", "synthetic_ratio", "min_chars")),
                          ("tokenizer", ("vocab_size",)),
                          ("model", ("preset", "name", "max_seq_len", "vocab_size")),
                          ("train", ("preset", "max_steps", "micro_batch_size", "grad_accum_steps", "max_seq_len",
                                     "resume", "out_dir", "durable_dir", "durable_keep_last"))):
        override = {k: v for k, v in ((k, getattr(args, k, None)) for k in keys) if v is not None}
        if override:
            raw.setdefault(section, {}).update(override)
    if getattr(args, "durable_every", False):
        raw.setdefault("train", {})["durable_on_improvement_only"] = False
    if getattr(args, "root", None):
        raw.setdefault("paths", {})["root"] = args.root
    return build_config(raw)


# ---------------------------------------------------------------- commands
def cmd_info(args: argparse.Namespace) -> int:
    hw = estimate_device_memory()
    print(json.dumps(hw, indent=2, ensure_ascii=False))
    if args.config and Path(args.config).exists():
        cfg = load_cfg(args)
        print("\nresolved config:")
        print(json.dumps({k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in cfg.items()}, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    from .data.sources import SourceRegistry

    for s in SourceRegistry.all_sources():
        flag = "IN " if (s.include_by_default and s.enabled) else ("OUT" if s.review_required else "off")
        print(f"[{flag}] {s.id:26s} {s.license:34s} {s.name}")
        if s.review_notes:
            print(f"       review: {s.review_notes}")
    print("\nIN = included by default; OUT = excluded pending licence review")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data.download import download_file, ensure_space, list_wiki_dump_files
    from .data.sources import SourceRegistry

    cfg = load_cfg(args)
    paths: PathsConfig = cfg["paths"]
    paths.ensure()
    ok = True
    for sid in cfg["data"].sources:
        if args.source and sid != args.source:
            continue
        try:
            src = SourceRegistry.get(sid)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            ok = False
            continue
        if not sid.startswith("azwiki") and not sid.endswith("wiki"):
            print(f"skip {sid}: no bulk download implemented ({src.license})")
            continue
        base = src.url.rstrip("/") + "/"
        try:
            files = list_wiki_dump_files(base)
        except Exception as exc:
            print(f"listing {sid} failed: {exc}", file=sys.stderr)
            ok = False
            continue
        target = files[0]
        url = target["url"]
        if not url.startswith("http"):
            url = base + url.lstrip("./")
        size = target.get("bytes")
        print(f"{sid}: {url} ({human_bytes(size) if size else 'unknown size'})")
        if size:
            enough, msg = ensure_space(paths.raw, size)
            if not enough:
                print(f"  NOT downloading: {msg}")
                ok = False
                continue
        res = download_file(url, paths.raw / sid / target["name"], expected_bytes=size)
        print(f"  -> {res.status} {human_bytes(res.bytes)} cached={res.from_cache} {res.error}")
        if res.status != "ok":
            ok = False
    return 0 if ok else 1


def cmd_sample(args: argparse.Namespace) -> int:
    """Fetch a small set of real articles to validate the pipeline cheaply."""
    from .data.download import (
        download_file,
        download_text,
        extract_wiki_streams,
        list_wiki_dump_files,
        read_multistream_index,
    )
    from .data.sources import SourceRegistry

    cfg = load_cfg(args)
    paths: PathsConfig = cfg["paths"]
    paths.ensure()
    source_id = args.source or cfg["data"].sources[0]
    src = SourceRegistry.get(source_id)
    base = src.url.rstrip("/") + "/"

    if args.mode == "api":
        out = Path(args.out or (paths.raw / source_id / "sample_api.json"))
        titles = args.titles or DEFAULT_SAMPLE_TITLES
        joined = "|".join(titles)
        url = (
            "https://az.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content"
            f"&rvslots=main&format=json&formatversion=2&titles={quote(joined)}"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(download_text(url), encoding="utf-8")
        print(f"wrote {out} ({human_bytes(out.stat().st_size)}) for {len(titles)} titles")
        return 0

    files = list_wiki_dump_files(base)
    dump = next((f for f in files if "pages-articles-multistream.xml.bz2" in f["name"]), None)
    index = next((f for f in files if "multistream-index" in f["name"]), None)
    if not dump:
        print(f"no multistream dump available for {source_id}", file=sys.stderr)
        return 1

    def _absurl(entry: dict) -> str:
        return entry["url"] if entry["url"].startswith("http") else base + entry["url"].lstrip("./")

    dump_url = _absurl(dump)
    index_path = paths.raw / source_id / "pages-articles-multistream-index.txt.bz2"
    if index and not index_path.exists():
        res = download_file(_absurl(index), index_path)
        print(f"index: {res.status} {human_bytes(res.bytes)}")
    if not index_path.exists():
        print("no multistream index available; download the full dump instead", file=sys.stderr)
        return 1

    offsets = read_multistream_index(index_path)
    print(f"index lists {len(offsets)} independent bz2 streams")
    xml = extract_wiki_streams(dump_url, offsets, max_streams=args.streams)

    out = Path(args.out or (paths.raw / source_id / "sample_articles.xml"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    print(f"wrote {out} ({human_bytes(len(xml.encode('utf-8')))}, {xml.count('<page>')} pages) from {args.streams} streams")
    return 0


DEFAULT_SAMPLE_TITLES = [
    "Azərbaycan",
    "Bakı",
    "Azərbaycan dili",
    "Azərbaycan Respublikası",
    "Səməd Vurğun",
    "Nizami Gəncəvi",
    "Xəzər dənizi",
    "Azerbaijani language",
    "Türk dili",
    "Qızıldaş",
    "Ağdam",
    "Fizika",
]


def cmd_clean(args: argparse.Namespace) -> int:
    from .data.prepare import build_corpus
    from .data.synthetic import generate as gen_synthetic

    cfg = load_cfg(args)
    paths = cfg["paths"]
    synthetic = []
    if cfg["data"].synthetic_ratio > 0:
        n = int(args.synthetic_n or max(1000, cfg["data"].synthetic_ratio * 20000))
        synthetic, stats = gen_synthetic(n=n, seed=cfg["data"].seed)
        print(f"synthetic: {stats.to_dict()}")
    manifest = build_corpus(
        paths,
        cfg["data"],
        source_ids=cfg["data"].sources,
        limit=args.limit or cfg["data"].max_documents,
        synthetic_texts=synthetic,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["statistics"], indent=2, ensure_ascii=False)[:2000])
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .data.quality import analyse

    cfg = load_cfg(args)
    report = analyse(cfg["paths"])
    print(f"report: {cfg['paths'].reports / 'dataset_quality.json'}")
    print(json.dumps({k: report[k] for k in ("documents_analysed", "total_characters", "unique_characters") if k in report}, indent=2))
    for lim in report.get("limitations", []):
        print(f"  ! {lim}")
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    from .data.prepare import iter_corpus
    from .tokenizer.trainer import train_tokenizer

    cfg = load_cfg(args)
    paths = cfg["paths"]
    texts: list[str] = []
    n = 0
    for row in iter_corpus(paths, "train"):
        texts.append(row["text"])
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"collected {n} documents for tokenizer training ({human_bytes(sum(len(t) for t in texts))})")
    stats = train_tokenizer(texts, paths.tokenizer_dir, cfg["tokenizer"])
    print(json.dumps({k: stats.to_dict()[k] for k in ("vocab_size", "training_docs", "training_bytes", "fragmentation", "vocabulary", "roundtrip")}, indent=2, ensure_ascii=False))
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    from .data.pack import pack_corpus
    from .tokenizer import TagAITokenizer

    cfg = load_cfg(args)
    paths = cfg["paths"]
    tok = TagAITokenizer.from_pretrained(paths.tokenizer_dir)
    for split in ("train", "validation"):
        meta, stats = pack_corpus(paths, tok, split, max_doc_tokens=args.max_doc_tokens or 2048)
        print(f"{split}: {stats.to_dict()}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .data.pack import BatchSampler, PackedDataset
    from .model import TagAIModel
    from .tokenizer import TagAITokenizer
    from .training.trainer import Trainer

    cfg = load_cfg(args)
    paths, mcfg, tcfg = cfg["paths"], cfg["model"], cfg["train"]
    paths.ensure()
    run_dir = Path(tcfg.out_dir).parent if tcfg.out_dir else paths.root_path

    tok = TagAITokenizer.from_pretrained(paths.tokenizer_dir)
    if mcfg.vocab_size != tok.vocab_size:
        print(f"aligning model vocab_size {mcfg.vocab_size} -> tokenizer {tok.vocab_size}")
        mcfg.vocab_size = tok.vocab_size

    train_meta = read_json(paths.processed / "tokens" / "train_meta.json")
    val_meta_path = paths.processed / "tokens" / "validation_meta.json"
    train_ds = PackedDataset(train_meta, tcfg.max_seq_len)
    val_ds = PackedDataset(read_json(val_meta_path), tcfg.max_seq_len) if val_meta_path.exists() else None
    train_batch = BatchSampler(train_ds, tcfg.micro_batch_size, seed=tcfg.seed)

    model = TagAIModel(mcfg)
    print(json.dumps({
        "model": mcfg.name,
        "parameters": model.num_parameters(),
        "non_embedding_parameters": model.num_parameters(non_embedding=True),
        "breakdown": model.parameter_breakdown(),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds) if val_ds else 0,
        "total_tokens": sum(s["tokens"] for s in train_meta["shards"]),
    }, indent=2))

    trainer = Trainer(model, train_batch, val_ds, tcfg, mcfg, run_dir, tokenizer=tok)
    print(json.dumps({"storage_policy": _storage_policy(tcfg), "before": trainer.storage_report()}, indent=2))
    resume_path = trainer.resolve_resume_path()
    if resume_path:
        print(f"resuming from {resume_path}")
        trainer.resume(resume_path)
    else:
        print("starting from randomly initialized weights (no checkpoint found)")

    state = trainer.train()
    trainer.save()
    print(json.dumps(trainer.storage_report(), indent=2))
    print(json.dumps(state.to_dict(), indent=2, default=str)[:800])
    return 0


def _storage_policy(tcfg) -> dict:
    """Explain the checkpoint budget before training starts."""
    return {
        "local_checkpoints_kept": tcfg.keep_last_checkpoints,
        "local_save_every_steps": tcfg.save_every,
        "durable_dir": tcfg.durable_dir or "(none - session-local only)",
        "durable_checkpoints_kept": tcfg.durable_keep_last,
        "durable_only_on_val_improvement": tcfg.durable_on_improvement_only,
        "durable_min_steps_between": tcfg.durable_min_steps_between,
        "durable_includes_optimizer_state": tcfg.durable_save_optimizer,
        "durable_min_free_gb": tcfg.durable_min_free_gb,
    }


def cmd_generate(args: argparse.Namespace) -> int:
    from .inference import GenerateConfig, TagAIGenerator
    from .training.checkpoint import find_latest_valid

    cfg = load_cfg(args)
    paths = cfg["paths"]
    ckpt = args.checkpoint or find_latest_valid(Path(cfg["train"].out_dir).parent)
    if ckpt is None:
        print("no checkpoint found; run `python -m tagai.cli train` first", file=sys.stderr)
        return 1
    gen = TagAIGenerator(ckpt, paths.tokenizer_dir)
    gencfg = GenerateConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    prompts = args.prompt or [
        "Salam, mənə kömək edə bilərsinizmi?",
        "Azərbaycan haqqında qısa məlumat ver.",
        "Bir gün Bakıda gəzirdim və",
    ]
    out_lines = []
    for p in prompts:
        text = gen.generate(p, gencfg)
        print(f"> {p}\n{text}\n")
        out_lines.append(f"> {p}\n{text}\n")
    if paths.samples.exists():
        (paths.samples / f"samples_{int(time.time())}.txt").write_text("\n".join(out_lines), encoding="utf-8")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import EvalSuite, evaluate_ppl
    from .inference import TagAIGenerator
    from .data.pack import PackedDataset
    from .training.checkpoint import find_latest_valid
    from .training.trainer import load_model_from_checkpoint

    cfg = load_cfg(args)
    paths, tcfg = cfg["paths"], cfg["train"]
    ckpt = args.checkpoint or find_latest_valid(Path(tcfg.out_dir).parent)
    if ckpt is None:
        print("no checkpoint found", file=sys.stderr)
        return 1

    model, mcfg = load_model_from_checkpoint(ckpt)
    tok_dir = paths.tokenizer_dir
    from .tokenizer import TagAITokenizer

    tok = TagAITokenizer.from_pretrained(tok_dir)
    gen = TagAIGenerator.from_model(model, tok)

    val_meta = paths.processed / "tokens" / "validation_meta.json"
    ppl = None
    if val_meta.exists():
        val_ds = PackedDataset(read_json(val_meta), tcfg.max_seq_len)
        ppl = evaluate_ppl(model, val_ds, max_batches=args.ppl_batches, micro_batch=tcfg.micro_batch_size)
        print("validation perplexity:", json.dumps(ppl, indent=2))

    suite = EvalSuite(max_new_tokens=args.max_new_tokens)
    report = suite.run(gen, max_new_tokens=args.max_new_tokens, verbose=args.verbose)
    report["validation_perplexity"] = ppl
    atomic_write_json(paths.reports / "evaluation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False))
    print(f"report: {paths.reports / 'evaluation.json'}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_bundle
    from .training.checkpoint import find_latest_valid

    cfg = load_cfg(args)
    paths, tcfg = cfg["paths"], cfg["train"]
    ckpt = args.checkpoint or find_latest_valid(Path(tcfg.out_dir).parent)
    if ckpt is None:
        print("no checkpoint found", file=sys.stderr)
        return 1
    report = paths.reports / "evaluation.json"
    manifest = export_bundle(
        ckpt,
        paths.tokenizer_dir,
        paths.export,
        evaluation_report=read_json(report) if report.exists() else None,
    )
    print(json.dumps({k: manifest[k] for k in ("name", "model", "tokenizer", "weights_human", "archive", "pretrained_weights_used")}, indent=2))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for fn in (cmd_clean, cmd_report, cmd_tokenize, cmd_pack, cmd_train):
        print(f"\n{'=' * 60}\n== {fn.__name__.replace('cmd_', '')}\n{'=' * 60}")
        rc = fn(args)
        if rc != 0:
            return rc
    return 0


COMMANDS = {
    "info": cmd_info,
    "sources": cmd_sources,
    "fetch": cmd_fetch,
    "sample": cmd_sample,
    "clean": cmd_clean,
    "report": cmd_report,
    "tokenize": cmd_tokenize,
    "pack": cmd_pack,
    "train": cmd_train,
    "generate": cmd_generate,
    "evaluate": cmd_evaluate,
    "export": cmd_export,
    "all": cmd_all,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tagai", description="TagAI - Azerbaijani language model from scratch")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to a JSON/JSONC config")
    p.add_argument("--root", default=None, help="override paths.root")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn):
        sp = sub.add_parser(name, help=fn.__doc__.split("\n")[0] if fn.__doc__ else name)
        sp.set_defaults(func=fn)
        return sp

    sp = add("info", cmd_info)
    add("sources", cmd_sources)

    sp = add("fetch", cmd_fetch)
    sp.add_argument("--source", default=None)

    sp = add("sample", cmd_sample)
    sp.add_argument("--source", default=None)
    sp.add_argument("--mode", default="multistream", choices=["multistream", "api"])
    sp.add_argument("--streams", type=int, default=20, help="bz2 streams to fetch (100 articles each)")
    sp.add_argument("--titles", nargs="*", default=None)
    sp.add_argument("--out", default=None)

    sp = add("clean", cmd_clean)
    sp.add_argument("--sources", nargs="*", default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--max-documents", type=int, default=None)
    sp.add_argument("--synthetic-ratio", type=float, default=None)
    sp.add_argument("--synthetic-n", type=int, default=None)
    sp.add_argument("--min-chars", type=int, default=None)

    add("report", cmd_report)

    sp = add("tokenize", cmd_tokenize)
    sp.add_argument("--vocab-size", type=int, default=None)
    sp.add_argument("--limit", type=int, default=200_000)

    sp = add("pack", cmd_pack)
    sp.add_argument("--max-doc-tokens", type=int, default=None)

    sp = add("train", cmd_train)
    sp.add_argument("--preset", default=None, choices=["smoke", "pilot", "main"])
    sp.add_argument("--max-steps", type=int, default=None)
    sp.add_argument("--micro-batch-size", type=int, default=None)
    sp.add_argument("--grad-accum-steps", type=int, default=None)
    sp.add_argument("--max-seq-len", type=int, default=None)
    sp.add_argument("--resume", default=None)
    sp.add_argument("--durable-dir", default=None,
                    help="persistent location for the newest checkpoint (e.g. Google Drive). "
                         "Only the last checkpoint is kept there, and only when validation improves.")
    sp.add_argument("--durable-keep-last", type=int, default=None)
    sp.add_argument("--durable-every", action="store_true",
                    help="mirror every durable checkpoint, not just improvements")

    sp = add("generate", cmd_generate)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--prompt", nargs="*", default=None)
    sp.add_argument("--max-new-tokens", type=int, default=96)
    sp.add_argument("--temperature", type=float, default=0.8)
    sp.add_argument("--top-k", type=int, default=40)
    sp.add_argument("--top-p", type=float, default=0.9)
    sp.add_argument("--seed", type=int, default=None)

    sp = add("evaluate", cmd_evaluate)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--max-new-tokens", type=int, default=96)
    sp.add_argument("--ppl-batches", type=int, default=50)
    sp.add_argument("--verbose", action="store_true")

    sp = add("export", cmd_export)
    sp.add_argument("--checkpoint", default=None)

    add("all", cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(logging, args.log_level))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())