"""Generate notebooks/tagai_from_scratch_colab.ipynb and the Kaggle variant.

The notebooks import the real `tagai` package rather than duplicating its
logic. That is deliberate: a notebook with its own inlined copy of the model
code drifts from the tested code the moment either side changes, and the
acceptance criteria are about the library actually working.

Run:  python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"

REPO_VERSION_HEADER = """\
# TagAI - Azerbaijani language model, trained from randomly initialized weights.
#
# This notebook drives the `tagai` package that lives in this repository.
# The model architecture, tokenizer, data pipeline and training loop all come
# from that package, so what runs here is exactly what the test suite covers.
#
# NOTHING here downloads pretrained weights. Every trainable parameter is
# initialised randomly at the start of the run.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------- Colab cells
COLAB_CELLS = [
    md(
        """
# TagAI - Azərbaycan dili ilə sıfırdan öyrənmə
## TagAI - Azerbaijani language model, trained from scratch

**Bu nət notebook nədir?** TagAI-nin təlim boru xəttini (data -> tokenizer -> model -> təlim -> qiymətləndirmə) bir səhifədə icra edir.

**What this is.** The complete TagAI pipeline in one notebook: data -> tokenizer -> model -> training -> evaluation -> export.

**Nə vacibdir:** Bu notebook heç bir əvvəlcədən təlim edilmiş model yükləmir. Bütün təlim olunan parametrlər təlimin başında təsadüfi şəkildə (random) başlatdılır.

**Critical:** No pretrained weights are downloaded or loaded. Every trainable parameter is randomly initialised at the start of the run.

**Sizə nə lazımdır:** Google Colab-da GPU seçmək (Runtime -> Change runtime type -> T4 GPU). Bu addımı yalnız siz edə bilərsiniz.
"""
    ),
    md("## 1. Mühit və GPU aşkarlama / Environment and GPU detection"),
    code(
        """
#@title 1. Mühit / Environment
import os, sys, platform, subprocess, json, time

# Colab-da GPU seçilməyibsə xəbərdarlıq göstər.
if "google.colab" in sys.modules:
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("not running inside Colab")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "| device", DEVICE)
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        GB = p.total_memory / 1024**3
        print(f"GPU: {p.name} | {GB:.1f} GB | capability {p.major}.{p.minor}")
        print("bf16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("XEBERDARIQ / WARNING: GPU tapilmadi. Təlim CPU-da çox yavaş olacaq.")
        print("WARNING: no GPU found. Training on CPU will be extremely slow.")
except ImportError:
    raise SystemExit("torch quraşdırılmalıdır / torch must be installed")
"""
    ),
    md("## 2. Asılılıqlar / Dependency installation"),
    code(
        """
#@title 2. Asılılıqlar / Dependencies
!pip -q install -q "torch>=2.1" "sentencepiece>=0.1.99" "numpy>=1.24" "tqdm>=4.65" "langdetect>=1.0.9" "safetensors>=0.4.0" "matplotlib>=3.7"

print("installed")
"""
    ),
    md(
        """
## 3. TagAI mənbə kodunu yükləmək / Load the TagAI source

Bu addım GitHub-ı göstərir. Əgər kodu əvvəlcədən yükləmisinizsə, `USER` və `REPO` dəyişdirin.
"""
    ),
    code(
        """
#@title 3. Mənbə kodu / Source
import subprocess, sys, os, shutil
from pathlib import Path

USER = "Tagiyevff"          # <-- öz GitHub istifadəçi adınız / your GitHub username
REPO = "tagiai"             # <-- repo adı / repository name
BRANCH = "main"

WORKDIR = Path("/content/tagai")
URL = f"https://github.com/{USER}/{REPO}.git"

# Kohnə/yarım klon varsa silirik: git clone mövcud qovluğa yazmır, yəni
# bozuk klon üstünde "if not exists" qoruması heç nəsi düzəltmirdi.
if WORKDIR.exists():
    shutil.rmtree(WORKDIR)

r = subprocess.run(
    ["git", "clone", "--depth", "1", "-b", BRANCH, URL, str(WORKDIR)],
    capture_output=True, text=True,
)
print(r.stdout or "", r.stderr or "")

# Yoxlama: tagai/data/ alt paketi klonla birlikdə gəlməlidir.
# Bu səhifə .gitignore problemi ilə itmişdi və 5-ci hücrədə
# ModuleNotFoundError: No module named 'tagai.data' verirdi.
required = [
    WORKDIR / "tagai" / "model" / "transformer.py",
    WORKDIR / "tagai" / "data" / "prepare.py",
    WORKDIR / "tagai" / "data" / "clean.py",
    WORKDIR / "tagai" / "tokenizer" / "trainer.py",
    WORKDIR / "tagai" / "training" / "trainer.py",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(
        "Repo natamam klonlandı / incomplete clone. Eksik fayllar:\\n  "
        + "\\n  ".join(missing)
        + "\\n\\nGitHub'da bu fayllar commit edilməmiş ola bilər - lokal `git push` edin."
    )

sys.path.insert(0, str(WORKDIR))
import tagai
import tagai.data, tagai.model, tagai.tokenizer, tagai.training
print("tagai", tagai.__version__, "from", Path(tagai.__file__).parent)
print("bütün modullar mövcuddur / all modules present")
"""
    ),
    md(
        """
## 4. Daimi saxlama / Persistent storage (Google Drive)

**Dikkat: Drive ücretsiz hesapta 15 GB'tır ve bu hızla dolar.** Bu yüzden
TagAI iki ayrı depolama kullanır:

| Nerede | Ne | Kural |
|---|---|---|
| `/content` (lokal, hızlı) | her `save_every` adımda checkpoint | son 3 tane tutulur |
| Google Drive (durable) | **sadece en son** checkpoint | sadece validation iyileştiğinde, en fazla 1 tane |

Böylece Drive'da sürekli **tek bir** checkpoint durur (~1.6 GB), 15 GB'ın
tamamı dolmaz. Drive kullanmak tamamen isteğe bağlıdır.

**Attention: a free Drive account holds 15 GB and fills fast.** TagAI keeps two
separate stores: frequent local checkpoints on the fast ephemeral disk, and a
single durable checkpoint on Drive written only when validation improves.
"""
    ),
    code(
        """
#@title 4. Storage policy
from pathlib import Path

USE_DRIVE = True   # False ise her şey oturumda kalır, Drive hiç kullanılmaz.

WORK = Path("/content/tagai_runs")     # hızlı, oturum sonunda silinir
if USE_DRIVE and "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DURABLE = Path("/content/drive/MyDrive/tagai/checkpoints")  # sadece 1 checkpoint
else:
    DURABLE = None

WORK.mkdir(parents=True, exist_ok=True)
print("hızlı lokal disk :", WORK)
print("durable (Drive)  :", DURABLE or "(yok - oturum içinde kalacak)")

if DURABLE is not None:
    # Klasör henüz yok; disk_usage() var olmayan yolda patlar.
    # free_space_gb() var olan en yakın üst klasöre kadar yükselir.
    from tagai.training.checkpoint import free_space_gb

    DURABLE.mkdir(parents=True, exist_ok=True)
    free = free_space_gb(DURABLE)
    print(f"Drive boş alanı: {free:.1f} GB / 15 GB")
    if free < 5:
        print("UYARI: Drive'da az yer var. DURABLE = None yapıp oturumda çalıştır.")
        print("WARNING: little Drive space left. Set DURABLE = None to run session-only.")
"""
    ),
    md("## 5. Qurulum / Configuration"),
    code(
        """
#@title 5. Qurulum / Configuration
from dataclasses import asdict
from tagai.config import build_config

RAW = {
    "paths": {"root": str(WORK / "main")},
    "data": {
        "sources": ["azwiki", "azwiktionary", "azwikisource", "azwikiquote", "azwikinews"],
        "min_chars": 250,
        "near_dedupe": True,
        "validation_fraction": 0.002,
        "synthetic_ratio": 0.02,
        "seed": 1337,
    },
    "tokenizer": {"vocab_size": 32000, "num_threads": 4},
    "model": {"preset": "tagai-base", "max_seq_len": 512},
    "train": {
        "preset": "main",
        "micro_batch_size": 8,
        "grad_accum_steps": 8,
        "max_seq_len": 512,
        "max_steps": 60000,
        "resume": "auto",

        # --- depolama bütçesi / storage budget -------------------------
        "keep_last_checkpoints": 3,        # lokal /content (hızlı disk)
        "durable_dir": str(DURABLE) if DURABLE else None,
        "durable_keep_last": 1,            # Drive'da sadece EN SON
        "durable_on_improvement_only": True,
        "durable_min_steps_between": 250,
        "durable_save_optimizer": True,    # False ise ~%65 daha küçük
        "durable_min_free_gb": 3.0,
    },
}

cfg = build_config(RAW)
print(json.dumps({k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in cfg.items()},
                 indent=2, ensure_ascii=False, default=str))
"""
    ),
    md("## 6. Mənbələrin lisenziya statusu / Source licensing"),
    code(
        """
#@title 6. Mənbələr / Sources
from tagai.data.sources import SourceRegistry

for s in SourceRegistry.all_sources():
    state = "DAXIL" if (s.include_by_default and s.enabled) else ("NƏZARƏT" if s.review_required else "kapalı")
    print(f"[{state:7s}] {s.id:24s} {s.license:34s} {s.name}")
print("\\nNƏZARƏT = lisenziya baxışı tamamlanmayıb, korpusa daxil EDİLMİR.")
print("PENDING = licence not reviewed, EXCLUDED from the corpus.")
"""
    ),
    md(
        """
## 7. Məlumatı yükləmək / Dataset download (resumable)

Bu addım bir neçə yüz MB - 1.5 GB yaza bilər. `azwiki` dumpı təxminən 1 GB-dır.
Yükləmə bölünür (resumable): dayananda yenidən başlayanda davam edir.
"""
    ),
    code(
        """
#@title 7. Yükləmə / Download
from tagai.data.download import download_file, list_wiki_dump_files, ensure_space
from tagai.data.sources import SourceRegistry

paths = cfg["paths"]; paths.ensure()

# Sürətli yoxlama üçün kiçik nümunə götürürük (təxminən 2500 məqalə).
sample_dump = paths.raw / "azwiki" / "sample_articles.xml"
if not sample_dump.exists():
    from tagai.data.download import extract_wiki_streams, read_multistream_index
    base = "https://dumps.wikimedia.org/azwiki/latest/"
    files = list_wiki_dump_files(base)
    idx = next(f for f in files if "multistream-index" in f["name"])
    dump = next(f for f in files if "pages-articles-multistream.xml.bz2" in f["name"])
    idx_path = paths.raw / "azwiki" / "pages-articles-multistream-index.txt.bz2"
    download_file(base + idx["url"], idx_path)
    offsets = read_multistream_index(idx_path)
    print("streams in dump:", len(offsets))
    xml = extract_wiki_streams(base + dump["url"], offsets, max_streams=25)
    sample_dump.write_text(xml, encoding="utf-8")
    print("wrote", sample_dump, xml.count("<page>"), "pages")
"""
    ),
    code(
        """
#@title 7b. Tam dump (COST: ~1 GB, vaxt: ~5-15 dəq) / Full dump (optional)
FULL_DOWNLOAD = False   # True olduqda ~1 GB yüklənir.

if FULL_DOWNLOAD:
    from tagai.data.download import download_file, list_wiki_dump_files, ensure_space
    base = "https://dumps.wikimedia.org/azwiki/latest/"
    files = list_wiki_dump_files(base)
    for entry in files:
        if not entry["name"].endswith("multistream.xml.bz2"):
            continue
        ok, msg = ensure_space(paths.raw, entry.get("bytes") or 3_000_000_000)
        print("space check:", msg)
        if not ok:
            continue
        res = download_file(base + entry["url"], paths.raw / "azwiki" / entry["name"],
                            expected_bytes=entry.get("bytes"))
        print(res.to_dict())
else:
    print("FULL_DOWNLOAD=False -> kiçik nümunə ilə davam edirik / continuing with the small sample")
"""
    ),
    md("## 8. Təmizləmə və filtrləmə / Cleaning and filtering"),
    code(
        """
#@title 8. Təmizləmə / Clean
from tagai.data.prepare import build_corpus
from tagai.data.synthetic import generate as gen_synthetic

synthetic, syn_stats = gen_synthetic(n=2000, seed=1337)
print("synthetic:", json.dumps(syn_stats.to_dict(), ensure_ascii=False))

manifest = build_corpus(paths, cfg["data"], synthetic_texts=synthetic, progress_every=5000)
print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
print(json.dumps(manifest["statistics"], indent=2, ensure_ascii=False)[:3000])
"""
    ),
    md("## 9. Deduplikasiya / Deduplication (already applied above, shown separately)"),
    code(
        """
#@title 9. Deduplikasiya / Deduplication
from tagai.data.dedupe import Deduplicator
from tagai.data.prepare import iter_corpus

dd = Deduplicator(threshold=cfg["data"].near_dedupe_threshold)
seen = 0
for row in iter_corpus(paths, "train"):
    seen += 1
    dup, kind = dd.is_duplicate(row["text"])
    if seen <= 5:
        print("sample", seen, "->", "DUPLICATE (" + kind + ")" if dup else "unique")
    if seen >= 5000:
        break
print(f"checked {seen} documents | exact duplicates: {dd.stats.exact_duplicates} "
      f"| near duplicates: {dd.stats.near_duplicates}")
"""
    ),
    md("## 10. Tokenizer təlimi / Tokenizer training"),
    code(
        """
#@title 10. Tokenizer
from tagai.data.prepare import iter_corpus
from tagai.tokenizer.trainer import train_tokenizer
from tagai.utils import human_bytes

texts = []
nbytes = 0
for row in iter_corpus(paths, "train"):
    texts.append(row["text"])
    nbytes += len(row["text"])
    if len(texts) >= 300000:
        break
print(f"{len(texts)} documents, {human_bytes(nbytes)} of Azerbaijani text")

tok_stats = train_tokenizer(texts, paths.tokenizer_dir, cfg["tokenizer"])
print(json.dumps({
    "vocab_size": tok_stats.vocab_size,
    "az_alphabet_coverage": tok_stats.vocabulary["az_alphabet_coverage"],
    "az_alphabet_missing": tok_stats.vocabulary["az_alphabet_missing"],
    "roundtrip_rate": tok_stats.roundtrip.get("roundtrip_rate"),
    "chars_per_token": tok_stats.fragmentation.get("chars_per_token"),
}, indent=2, ensure_ascii=False))
"""
    ),
    md("## 11. Təlim/vərəfiksion hazırlığı / Train and validation preparation"),
    code(
        """
#@title 11. Pack
from tagai.data.pack import pack_corpus
from tagai.tokenizer import TagAITokenizer

tok = TagAITokenizer.from_pretrained(paths.tokenizer_dir)
print("vocab", tok.vocab_size)

for split in ("train", "validation"):
    meta, stats = pack_corpus(paths, tok, split, max_doc_tokens=2048)
    print(split, json.dumps(stats.to_dict(), ensure_ascii=False))

TOTAL_TOKENS = sum(s["tokens"] for s in __import__("tagai.utils", fromlist=["read_json"]).read_json(
    paths.processed / "tokens" / "train_meta.json")["shards"])
print("total training tokens:", TOTAL_TOKENS)
"""
    ),
    md("## 12. Model yaratmaq (təsadüfi başlatma) / Model creation with randomly initialized weights"),
    code(
        """
#@title 12. Model
from tagai.config import ModelConfig
from tagai.model import TagAIModel

model_cfg = ModelConfig(**{**asdict(cfg["model"]), "vocab_size": tok.vocab_size})
print(json.dumps(asdict(model_cfg), indent=2))
"""
    ),
    md("## 13. Parametr sayı və başlatmanın yoxlanması / Parameter count and initialization verification"),
    code(
        """
#@title 13. Initialization verification
model = TagAIModel(model_cfg)
report = {
    "parameters": model.num_parameters(),
    "non_embedding_parameters": model.num_parameters(non_embedding=True),
    "breakdown": model.parameter_breakdown(),
    "tied_embeddings": model.lm_head.weight is model.tok_emb.weight,
}
print(json.dumps(report, indent=2))

# Sübut: təsadüfi başlatma ~ln(V) itkilə verir.
x = torch.randint(0, model_cfg.vocab_size, (4, 64))
with torch.no_grad():
    _, loss = model(x[:, :-1], x[:, 1:])
print(f"initial loss {float(loss):.3f} vs ln(V)={torch.log(torch.tensor(float(model_cfg.vocab_size))):.3f}")
print("bu, modelin əvvəlcədən təlim olunmadığının sübutudur.")
print("This is evidence the model was not pretrained: a pretrained model starts far below ln(V).")
"""
    ),
    md("## 14. Kiçik test (smoke test) / Small-batch smoke test"),
    code(
        """
#@title 14. Smoke test
model = model.to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
x = torch.randint(0, model_cfg.vocab_size, (2, min(64, model_cfg.max_seq_len)), device=DEVICE)
t0 = time.time()
for i in range(3):
    loss = model(x[:, :-1], x[:, 1:])[1]
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize() if DEVICE == "cuda" else None
print(f"3 steps ok in {time.time()-t0:.2f}s | loss {float(loss):.3f}")
"""
    ),
    md("## 15. Təlim konfiqurasiyasının seçilməsi / Training configuration selection"),
    code(
        """
#@title 15. Config selection
# Təlim konfiqurasiyası GPU-yə görə avtomatik tənzimlənir.
train_cfg = cfg["train"]

def pick_batch(mb_default=8, mb_big=16):
    if DEVICE != "cuda":
        return 2
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gb < 10: return mb_default
    if gb < 20: return mb_big
    return mb_big * 2

train_cfg.micro_batch_size = pick_batch()
print("selected micro_batch_size:", train_cfg.micro_batch_size)
print("effective batch (tokens):", train_cfg.micro_batch_size * train_cfg.grad_accum_steps * train_cfg.max_seq_len)
print("total training tokens:", TOTAL_TOKENS)
steps_for_one_epoch = TOTAL_TOKENS // max(1, train_cfg.effective_batch_tokens)
print("steps for ~1 epoch:", steps_for_one_epoch)
print("max_steps:", train_cfg.max_steps, "-> estimated epochs:",
      round(train_cfg.max_steps / max(1, steps_for_one_epoch), 2))
"""
    ),
    md(
        """
## 16. Təlim / Pretraining

**Qeyd:** Colab sessiya limitləri var. Təlim avtomatik dayananda checkpoint eynə yerdə qalır.
**Note:** Colab has session limits. Checkpoints survive interruptions, so the next run resumes.
"""
    ),
    code(
        """
#@title 16. Pretraining
from tagai.data.pack import BatchSampler, PackedDataset
from tagai.training.trainer import Trainer
from tagai.training.checkpoint import find_latest_valid
from tagai.utils import read_json, set_seed

set_seed(train_cfg.seed)
train_meta = read_json(paths.processed / "tokens" / "train_meta.json")
val_meta_path = paths.processed / "tokens" / "validation_meta.json"

train_ds = PackedDataset(train_meta, train_cfg.max_seq_len)
val_ds = PackedDataset(read_json(val_meta_path), train_cfg.max_seq_len) if val_meta_path.exists() else None
train_sampler = BatchSampler(train_ds, train_cfg.micro_batch_size, seed=train_cfg.seed)

run_dir = WORK / "main"
model = TagAIModel(model_cfg)
trainer = Trainer(model, train_sampler, val_ds, train_cfg, model_cfg, run_dir, tokenizer=tok)

print("depolama / storage:", json.dumps(trainer.storage_report(), indent=2))
resume_path = trainer.resolve_resume_path()
if resume_path:
    print("davam edirik / resuming from", resume_path)
    trainer.resume(resume_path)
else:
    print("TƏSADÜFİ BAŞLATMA / starting from RANDOM initialization")

state = trainer.train()
trainer.save()
print(json.dumps(state.to_dict(), indent=2, default=str)[:1000])
"""
    ),
    md("## 17. Nəzarət nöqtəsi yaddaşının verilməsi / Periodic checkpointing (details)"),
    code(
        """
#@title 17. Checkpoints
from tagai.training.checkpoint import list_checkpoints, list_checkpoints_in

print("--- lokal /content (hızlı disk) ---")
for c in list_checkpoints(run_dir):
    print(f"step {c.step:>7d} | {c.bytes/1024**2:8.1f} MB | sha256 ok={c.sha256_ok}")

print("\\n--- durable (Drive) ---")
if DURABLE:
    for c in list_checkpoints_in(DURABLE):
        print(f"step {c.step:>7d} | {c.bytes/1024**2:8.1f} MB | sha256 ok={c.sha256_ok}")
else:
    print("(yok / none)")

print("\\n", json.dumps(trainer.storage_report(), indent=2))
print(f"\\nLokal: her {train_cfg.save_every} addımda, son {train_cfg.keep_last_checkpoints} tane.")
print(f"Durable: sadece validation iyileştiğinde, en fazla {train_cfg.durable_keep_last} tane.")
print("Drive'da kalıcı olarak yalnızca TEK checkpoint tutulur.")
"""
    ),
    md("## 18. Validasiya və mətn yaratma / Validation and text generation"),
    code(
        """
#@title 18. Generate
from tagai.inference import TagAIGenerator, GenerateConfig
from tagai.training.checkpoint import find_latest_valid

ckpt = find_latest_valid(run_dir)
print("checkpoint:", ckpt)
gen = TagAIGenerator(ckpt, paths.tokenizer_dir, device=DEVICE)

PROMPTS = [
    "Salam, mənə kömək edə bilərsinizmi?",
    "Azərbaycan haqqında qısa məlumat ver.",
    "Bir gün Bakıda gəzirdim və",
    "Ən böyük səhər keçirən şəhər hansıdır?",
]
gc = GenerateConfig(max_new_tokens=80, temperature=0.8, top_k=40, top_p=0.9, seed=1337)
samples = []
for p in PROMPTS:
    t = gen.generate(p, gc)
    samples.append(t)
    print(f"\\n> {p}\\n{t}")
(paths.samples / "colab_samples.txt").write_text("\\n\\n".join(samples), encoding="utf-8")
"""
    ),
    md("## 19. Yekun qiymətləndirmə / Final evaluation"),
    code(
        """
#@title 19. Evaluation
from tagai.evaluation import EvalSuite, evaluate_ppl
from tagai.evaluation.perplexity import plot_loss_curve
from tagai.utils import atomic_write_json, read_json as _read

model_eval, _ = __import__("tagai.training.trainer", fromlist=["x"]).load_model_from_checkpoint(ckpt, device=DEVICE)
report = {}
if val_ds is not None:
    from tagai.data.pack import BatchSampler as _BS
    val_sampler = _BS(val_ds, 4, shuffle=False)
    report["validation_perplexity"] = evaluate_ppl(model_eval, val_sampler, max_batches=50, micro_batch=4)
    print("validation ppl:", report["validation_perplexity"])

suite_report = EvalSuite(max_new_tokens=96).run(gen, verbose=False)
suite_report["validation_perplexity"] = report.get("validation_perplexity")
atomic_write_json(paths.reports / "evaluation.json", suite_report)
print(json.dumps({k: v for k, v in suite_report.items() if k != "results"}, indent=2, ensure_ascii=False))
print("\\nQeyd: itki (loss) azalması yazma qabiliyyətini sübut etmir.")
print("Note: a lower loss does not by itself demonstrate writing quality.")
"""
    ),
    md("## 20. Model və tokenizer ixracı / Export model and tokenizer"),
    code(
        """
#@title 20. Export
from tagai.export import export_bundle
from tagai.utils import read_json

eval_path = paths.reports / "evaluation.json"
bundle = export_bundle(
    ckpt, paths.tokenizer_dir, paths.export,
    evaluation_report=read_json(eval_path) if eval_path.exists() else None,
)
print(json.dumps({k: bundle[k] for k in
      ("name", "model", "tokenizer", "weights_human", "archive", "pretrained_weights_used")},
      indent=2, ensure_ascii=False))
"""
    ),
    md("## 21. Təlim hesabatı / Training report"),
    code(
        """
#@title 21. Report
from tagai.utils import atomic_write_json, read_json

train_report = {
    "model_config": asdict(model_cfg),
    "train_config": asdict(train_cfg),
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "torch": torch.__version__,
    "parameters": model_eval.num_parameters(),
    "total_training_tokens": TOTAL_TOKENS,
    "steps_completed": state.step,
    "tokens_seen": state.tokens_seen,
    "best_val_loss": state.best_val_loss,
    "history": state.history[-200:],
    "dataset_manifest": manifest["counts"],
    "pretrained_weights_used": False,
}
atomic_write_json(paths.reports / "training_report.json", train_report)
print(json.dumps({k: v for k, v in train_report.items() if k != "history"}, indent=2, ensure_ascii=False, default=str))

hist = state.history
try:
    p = plot_loss_curve(hist, paths.reports / "loss_curve.png")
    print("loss curve:", p)
except Exception as e:
    print("plot skipped:", e)
"""
    ),
    md("## 22. Faylların paketlənməsi / Packaging the artifacts"),
    code(
        """
#@title 22. Package
import shutil, zipfile
from pathlib import Path

# YALNIZCA küçük artefaktlar paketlenir.
# Token shard'ları (data/processed/tokens) GB'larca yer tutar ve Drive'a ASLA konmaz;
# onlar zaten /content'te lokal olarak durur ve gerekirse yeniden üretilebilir.
STAGE = Path("/content/tagai_stage")
if STAGE.exists():
    shutil.rmtree(STAGE)
(STAGE / "model").mkdir(parents=True, exist_ok=True)
(STAGE / "tokenizer").mkdir(parents=True, exist_ok=True)

for src in [paths.export, paths.reports, paths.tokenizer_dir]:
    if src.exists():
        shutil.copytree(src, STAGE / src.name, dirs_exist_ok=True)

bundle = Path("/content/tagai_bundle.zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for p in STAGE.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(STAGE))

print(f"bundle: {bundle} ({bundle.stat().st_size/1024**2:.1f} MB)")
print("içində / contains:", sorted(x.name for x in STAGE.iterdir()))

if DURABLE is not None:
    # Sadece hazır modeli Drive'a taşı; token verisi DEĞİL.
    dest = DURABLE.parent / "tagai_bundle.zip"
    shutil.copyfile(bundle, dest)
    print(f"Drive'a kopyalandı / copied to Drive: {dest} ({dest.stat().st_size/1024**2:.1f} MB)")
else:
    print("Drive kullanılmadı; bundle'ı 'Files' panelinden indir.")
    print("Drive not used; download the bundle from the Files panel.")

print("\\nBoyut kontrolü / size check:")
for p in sorted(DURABLE.parent.rglob("*")) if DURABLE else []:
    if p.is_file():
        print(f"  {p.relative_to(DURABLE.parent)}  {p.stat().st_size/1024**2:.1f} MB")
"""
    ),
    md(
        """
## Yekun / Summary

Bu notebook icra etdi:

1. Mühit və GPU aşkarlama
2. Asılılıqların quraşdırılması
3. Daimi saxlama (Google Drive)
4. Mənbələrin lisenziya yoxlaması
5. Məlumatın yüklənməsi (dayandırıla bilən)
6. Təmizləmə, filtrləmə, deduplikasiya
7. Tokenizer təlimi + Unicode yoxlaması
8. Token paketləməsi
9. Təsadüfi başlatılmış modelin yaradılması
10. Parametr sayının yoxlanması
11. Smoke test
12. Konfiqurasiya seçimi
13. Təlim + checkpoint + bərpa
14. Validasiya və mətn yaratma
15. Qiymətləndirmə
16. İxrac və hesabat

**Təhlükəsizlik bərddası:** heç bir əvvəlcədən təlim edilmiş çəki yüklənməyib.
**Safety guarantee:** no pretrained weights were loaded anywhere in this notebook.
"""
    ),
]


# ------------------------------------------------------------- Kaggle cells
KAGGLE_CELLS = [
    md(
        """
# TagAI on Kaggle - Azərbaycan dili ilə sıfırdan təlim

**Platform seçimi:** Colab işləmirsə Kaggle alternativdir. Eyni `tagai` paketini işlədir.

**Bu nət notebook nədir?** TagAI-nin təlim boru xəttini (data -> tokenizer -> model -> təlim -> qiymətləndirmə) bir səhifədə icra edir.

**What this is.** The complete TagAI pipeline in one notebook: data -> tokenizer -> model -> training -> evaluation -> export.

**Nə vacibdir:** Heç bir əvvəlcədən təlim edilmiş model yüklənmir.
**Critical:** No pretrained weights are downloaded or loaded. Every trainable parameter is randomly initialised.
"""
    ),
    md("## 0. Qısa yol / Fast path: əvvəlcədən hazırlanmış model yükləmək"),
    md(
        """
Colab və ya Kaggle-da özünüz əvvəlcədən bir `tagai_run.zip` ixrac etmisinizsə, aşağıdakı hüceyrələri işlədin və təlimi səhifə 5-dən başlayın.
If you exported `tagai_run.zip` from a previous session, upload it with the cell below and skip straight to training.
"""
    ),
    code(
        """
#@title 0. Fayl yükləmə (varsa) / Upload an existing run (optional)
from pathlib import Path
import zipfile, shutil, os

UPLOAD = False  # True olduqda "tagai_run.zip" faylını seçin.
RUN_DIR = Path("/kaggle/working/tagai")

if UPLOAD:
    from kaggle_secrets import Environment
    print("Colab-da fayl yükləmək üçün files.upload() istifadə edin.")
    print("On Kaggle use the 'Add Input' panel or files.upload() in Colab.")

if (Path("/kaggle/working") / "tagai_run.zip").exists():
    with zipfile.ZipFile("/kaggle/working/tagai_run.zip") as z:
        z.extractall("/kaggle/working")
    print("extracted an existing run")
"""
    ),
    md("## 1. Mühit və GPU / Environment and GPU"),
    code(
        """
#@title 1. Environment
import os, sys, platform, subprocess, json, time
from pathlib import Path

if "google.colab" in sys.modules:
    subprocess.run(["nvidia-smi"], check=False)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", DEVICE)
if DEVICE == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | {p.total_memory/1024**3:.1f} GB")
else:
    print("GPU tapilmadi / no GPU found")

ROOT = Path("/kaggle/working/tagai") if "kaggle" in sys.modules else Path("/content/tagai_runs")
print("run root:", ROOT)
"""
    ),
    md("## 2. Asılılıqlar / Dependencies"),
    code(
        """
#@title 2. Dependencies
!pip -q install -q "torch>=2.1" "sentencepiece>=0.1.99" "numpy>=1.24" "tqdm>=4.65" "langdetect>=1.0.9" "safetensors>=0.4.0" "matplotlib>=3.7"
print("installed")
"""
    ),
    md("## 3. Mənbə kodu / Load the TagAI source"),
    code(
        """
#@title 3. Source
import subprocess, sys
from pathlib import Path

USER = "Tagiyevff"
REPO = "tagiai"
WORKDIR = ROOT

if not (WORKDIR / "tagai" / "model" / "transformer.py").exists():
    r = subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{USER}/{REPO}.git", str(WORKDIR)],
                       capture_output=True, text=True)
    print(r.stdout or "", r.stderr or "")

sys.path.insert(0, str(WORKDIR))
import tagai
print("tagai", tagai.__version__)
"""
    ),
    md("## 4. Məlumat / Data"),
    code(
        """
#@title 4. Data
from tagai.config import build_config
from tagai.data.download import download_file, list_wiki_dump_files, extract_wiki_streams, read_multistream_index

RAW = {
    "paths": {"root": str(ROOT / "main")},
    "data": {"sources": ["azwiki"], "min_chars": 250, "validation_fraction": 0.002, "seed": 1337},
    "tokenizer": {"vocab_size": 32000, "num_threads": 4},
    "model": {"preset": "tagai-base", "max_seq_len": 512},
    "train": {"preset": "main", "micro_batch_size": 8, "grad_accum_steps": 8,
              "max_seq_len": 512, "max_steps": 60000, "resume": "auto"},
}
cfg = build_config(RAW)
paths = cfg["paths"]; paths.ensure()

base = "https://dumps.wikimedia.org/azwiki/latest/"
files = list_wiki_dump_files(base)
idx = next(f for f in files if "multistream-index" in f["name"])
dump = next(f for f in files if "pages-articles-multistream.xml.bz2" in f["name"])
idx_path = paths.raw / "azwiki" / "pages-articles-multistream-index.txt.bz2"
download_file(base + idx["url"], idx_path)
offsets = read_multistream_index(idx_path)
xml = extract_wiki_streams(base + dump["url"], offsets, max_streams=25)
sample = paths.raw / "azwiki" / "sample_articles.xml"
sample.write_text(xml, encoding="utf-8")
print("wrote", sample, xml.count("<page>"), "pages")
"""
    ),
    code(
        """
#@title 4b. Clean, tokenize, pack
from tagai.data.prepare import build_corpus, iter_corpus
from tagai.tokenizer.trainer import train_tokenizer
from tagai.tokenizer import TagAITokenizer
from tagai.data.pack import pack_corpus

manifest = build_corpus(paths, cfg["data"], progress_every=5000)
print(json.dumps(manifest["counts"], indent=2))

texts = [r["text"] for r in iter_corpus(paths, "train")]
print("tokenizer corpus:", len(texts), "documents")
tok_stats = train_tokenizer(texts, paths.tokenizer_dir, cfg["tokenizer"])
print("vocab", tok_stats.vocab_size, "alphabet coverage", tok_stats.vocabulary["az_alphabet_coverage"])

tok = TagAITokenizer.from_pretrained(paths.tokenizer_dir)
for split in ("train", "validation"):
    _, st = pack_corpus(paths, tok, split, max_doc_tokens=2048)
    print(split, st.to_dict())
"""
    ),
    md("## 5. Model və təlim / Model and training"),
    code(
        """
#@title 5. Train
from dataclasses import asdict
import torch
from tagai.config import ModelConfig
from tagai.model import TagAIModel
from tagai.data.pack import BatchSampler, PackedDataset
from tagai.training.trainer import Trainer
from tagai.utils import read_json, set_seed

train_cfg = cfg["train"]
if DEVICE == "cuda":
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    train_cfg.micro_batch_size = 8 if gb < 20 else 16
print("micro_batch_size:", train_cfg.micro_batch_size)

model_cfg = ModelConfig(**{**asdict(cfg["model"]), "vocab_size": tok.vocab_size})
set_seed(train_cfg.seed)
model = TagAIModel(model_cfg)
print("parameters:", model.num_parameters())

train_meta = read_json(paths.processed / "tokens" / "train_meta.json")
val_meta_path = paths.processed / "tokens" / "validation_meta.json"
train_ds = PackedDataset(train_meta, train_cfg.max_seq_len)
val_ds = PackedDataset(read_json(val_meta_path), train_cfg.max_seq_len) if val_meta_path.exists() else None
train_sampler = BatchSampler(train_ds, train_cfg.micro_batch_size, seed=train_cfg.seed)

run_dir = ROOT / "main"
trainer = Trainer(model, train_sampler, val_ds, train_cfg, model_cfg, run_dir, tokenizer=tok)
rp = trainer.resolve_resume_path()
if rp:
    print("resuming from", rp); trainer.resume(rp)
else:
    print("RANDOM initialization / təsadüfi başlatma")

state = trainer.train()
trainer.save()
print(json.dumps({k: v for k, v in state.to_dict().items() if k != "history"}, indent=2, default=str))
"""
    ),
    md("## 6. Nəticələr, ixrac / Results and export"),
    code(
        """
#@title 6. Evaluate and export
from tagai.inference import TagAIGenerator, GenerateConfig
from tagai.evaluation import EvalSuite
from tagai.export import export_bundle
from tagai.training.checkpoint import find_latest_valid
from tagai.utils import atomic_write_json

ckpt = find_latest_valid(run_dir)
gen = TagAIGenerator(ckpt, paths.tokenizer_dir, device=DEVICE)
for p in ["Salam!", "Azərbaycan haqqında məlumat ver.", "Bir gün Bakıda gəzirdim və"]:
    print(f"> {p}\\n{gen.generate(p, GenerateConfig(max_new_tokens=64, seed=1337))}\\n")

rep = EvalSuite(max_new_tokens=64).run(gen)
atomic_write_json(paths.reports / "evaluation.json", rep)
print(json.dumps({k: v for k, v in rep.items() if k != "results"}, indent=2, ensure_ascii=False))

bundle = export_bundle(ckpt, paths.tokenizer_dir, paths.export,
                       evaluation_report=rep)
print(json.dumps({k: bundle[k] for k in ("name", "weights_human", "archive", "pretrained_weights_used")},
                 indent=2, ensure_ascii=False))
print("\\nKaggle-də çıxışı 'Save Version' -> 'Save & Run All' ilə yükləyə bilərsiniz.")
print("On Kaggle use 'Save Version' with 'Save & Run All' to keep the output files.")
"""
    ),
    md(
        """
## Yekun / Summary

Kaggle alternatividir: eyni `tagai` paketi, eyni model kodu, eyni tokenizer konfiqurasiyası.
Platform fərqləri (GPU növü, RAM, disk) avtomatik aşkarlanır və batch ölçüsü ona görə seçilir.
"""
    ),
]


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    colab_path = NOTEBOOKS / "tagai_from_scratch_colab.ipynb"
    colab_path.write_text(json.dumps(notebook(COLAB_CELLS), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {colab_path} ({len(COLAB_CELLS)} cells)")

    kaggle_path = NOTEBOOKS / "tagai_from_scratch_kaggle.ipynb"
    kaggle_path.write_text(json.dumps(notebook(KAGGLE_CELLS), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {kaggle_path} ({len(KAGGLE_CELLS)} cells)")

    for path in (colab_path, kaggle_path):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["nbformat"] == 4
        code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
        print(f"  {path.name}: {len(data['cells'])} cells, {len(code_cells)} code")


if __name__ == "__main__":
    main()