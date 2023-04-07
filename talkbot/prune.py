"""Prune a model to reduce its size."""

from torch.nn.utils.prune import l1_unstructured, remove
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


async def prune(src_model: str, save_dir: str, amount: float = 0.5):
    """Prune a model to reduce its size."""

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(src_model)
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(src_model)

    l1_unstructured(model.transformer.wte, name="weight", amount=amount)
    remove(model.transformer.wte, "weight")

    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
