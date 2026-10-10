"""Sampling from a trained TagAI model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from ..tokenizer.wrapper import TagAITokenizer
from ..utils import get_logger

log = get_logger("inference")


@dataclass
class GenerateConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    seed: Optional[int] = None
    stop_on_eos: bool = True


class TagAIGenerator:
    """Load a checkpoint + tokenizer and produce Azerbaijani text.

    This is a deliberate reference implementation rather than an optimised
    server: KV cache, batching and sampling all work, but throughput is not
    the point.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_dir: str | Path,
        device: Optional[str] = None,
    ) -> None:
        from ..training.trainer import load_model_from_checkpoint

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.model_cfg = load_model_from_checkpoint(checkpoint_path, device=str(self.device))
        self.model.eval()
        self.tokenizer = TagAITokenizer.from_pretrained(tokenizer_dir)

    @classmethod
    def from_model(cls, model, tokenizer, device: Optional[str] = None) -> "TagAIGenerator":
        obj = cls.__new__(cls)
        obj.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        obj.model = model.to(obj.device).eval()
        obj.tokenizer = tokenizer
        obj.model_cfg = model.cfg
        return obj

    def generate(self, prompt: str, cfg: Optional[GenerateConfig] = None) -> str:
        cfg = cfg or GenerateConfig()
        gen = None
        if cfg.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(cfg.seed)
        ids = self.tokenizer.encode(prompt, add_bos=True)
        if not ids:
            ids = [self.tokenizer.bos_id]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            x,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            generator=gen,
            eos_id=self.tokenizer.eos_id if cfg.stop_on_eos else None,
            pad_id=self.tokenizer.pad_id,
        )
        return self.tokenizer.decode(out[0].tolist())

    def generate_batch(self, prompts: Sequence[str], cfg: Optional[GenerateConfig] = None) -> list[str]:
        return [self.generate(p, cfg) for p in prompts]

    def score(self, text: str) -> dict:
        """Per-token negative log-likelihood of ``text`` - a real quality signal."""
        ids = self.tokenizer.encode(text, add_bos=True)
        if len(ids) < 2:
            return {"tokens": len(ids), "nll": float("nan"), "ppl": float("nan")}
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, loss = self.model(x, y)
        nll = float(loss.item())
        return {"tokens": len(ids) - 1, "nll": nll, "ppl": float(torch.exp(torch.tensor(nll)))}


def interactive(generator: TagAIGenerator, cfg: Optional[GenerateConfig] = None) -> None:  # pragma: no cover
    """Tiny REPL for manual inspection."""
    print("TagAI interactive. Commands: /exit, /cfg")
    while True:
        try:
            prompt = input("az> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt in ("/exit", "/quit", ""):
            break
        if prompt == "/cfg":
            print(cfg)
            continue
        started = time.time()
        print(generator.generate(prompt, cfg))
        print(f"  [{time.time() - started:.2f}s]")