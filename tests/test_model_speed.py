# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.nn.functional as F
from test_policy import TINY_CONFIG
from torch.utils._python_dispatch import TorchDispatchMode

from verne.model import WorldActionModel
from verne.model.config import VIDEO_TEXT_LEN


def tiny_model() -> WorldActionModel:
    torch.manual_seed(0)
    model = WorldActionModel.from_config(TINY_CONFIG).eval()
    with torch.no_grad():
        model.memory.slot_bias.normal_()
    return model


def inputs(seed: int, prefix: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    cfg = TINY_CONFIG
    kw = {
        "latents": torch.randn(1, 48, 1, 8, 16, generator=g),
        "proprio": torch.randn(1, cfg["proprio_dim"], generator=g),
        "noise": torch.randn(1, cfg["chunk_size"], cfg["action_dim"], generator=g),
        "past_latents": torch.randn(1, 48, cfg["num_past_frames"], 8, 16, generator=g),
        "past_padded": torch.arange(cfg["num_past_frames"])[None] < 5,
    }
    if prefix:
        kw["prefix"] = torch.randn(1, prefix, cfg["action_dim"], generator=g)
    return kw


def prompt(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    text = torch.randn(1, TINY_CONFIG["text_len"], TINY_CONFIG["text_dim"], generator=g)
    return text, torch.ones(1, TINY_CONFIG["text_len"], dtype=torch.bool)


def test_kept_prompt_gives_the_chunk_of_a_fresh_model():
    model = tiny_model()
    text, mask = prompt(0)
    model.infer_action(text=text, text_mask=mask, **inputs(1))
    warm = model.infer_action(text=text, text_mask=mask, **inputs(2, prefix=5))
    cold = tiny_model().infer_action(text=text, text_mask=mask, **inputs(2, prefix=5))
    assert torch.equal(warm, cold)

    text.mul_(2.0)
    changed = model.infer_action(text=text, text_mask=mask, **inputs(3))
    assert torch.equal(changed, tiny_model().infer_action(text=text, text_mask=mask, **inputs(3)))
    other, other_mask = prompt(4)
    new = model.infer_action(text=other, text_mask=other_mask, **inputs(3))
    assert torch.equal(
        new, tiny_model().infer_action(text=other, text_mask=other_mask, **inputs(3))
    )


def test_kept_video_text_keys_and_values_match_the_whole_context():
    """The text rows are projected once per prompt and the state row once per chunk, so the
    fp32 matmuls run on other shapes than one projection of all rows. Tolerance 1e-5."""
    model = tiny_model()
    text, mask = prompt(0)
    kw = inputs(1)
    model.infer_action(text=text, text_mask=mask, **kw)
    with torch.no_grad():
        padding = text.new_zeros((1, VIDEO_TEXT_LEN - text.shape[1], text.shape[2]))
        state_row = model.proprio_proj(kw["proprio"].unsqueeze(1))
        context = model.video.text_embedding(torch.cat([text, padding, state_row], dim=1))
        expected = torch.stack(
            [torch.stack(b.cross_attn.project_kv(context)) for b in model.video.blocks[:-1]]
        )
    torch.testing.assert_close(model._prompt.video_kv, expected, rtol=1e-5, atol=1e-5)


def test_open_text_mask_reaches_attention_as_no_mask(monkeypatch):
    sdpa = F.scaled_dot_product_attention
    bool_masks = []

    def spy(q, k, v, attn_mask=None, **kwargs):
        bool_masks.append(attn_mask is not None and attn_mask.dtype == torch.bool)
        return sdpa(q, k, v, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    text, mask = prompt(0)
    tiny_model().infer_action(text=text, text_mask=mask, **inputs(1))
    assert bool_masks and not any(bool_masks)


class Float64Ops(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        outs = out if isinstance(out, (tuple, list)) else (out,)
        self.count += any(isinstance(o, torch.Tensor) and o.dtype == torch.float64 for o in outs)
        return out


def test_a_warm_chunk_builds_no_constants():
    model = tiny_model()
    text, mask = prompt(0)
    for prefix in (0, 5):
        model.infer_action(text=text, text_mask=mask, **inputs(1, prefix))
    for prefix in (0, 5):
        with Float64Ops() as ops:
            model.infer_action(text=text, text_mask=mask, **inputs(2, prefix))
        assert ops.count == 0


def test_rope_tables_move_with_the_weights_and_stay_float32():
    model = tiny_model().to(torch.bfloat16)
    tables = (*model.video.rope, model.action.rope)
    assert all(t.dtype == torch.float32 for t in tables)
    model.to("meta")
    assert all(t.device.type == "meta" for t in (*model.video.rope, model.action.rope))


def test_plain_and_prefix_chunks_share_one_compiled_graph():
    torch._dynamo.reset()
    counter = torch._dynamo.testing.CompileCounter()
    model = tiny_model()
    model.enable_compile(backend=counter)
    text, mask = prompt(0)
    model.infer_action(text=text, text_mask=mask, **inputs(1))
    graphs = counter.frame_count
    for prefix in (5, 0, 12):
        model.infer_action(text=text, text_mask=mask, **inputs(2, prefix))
    assert counter.frame_count == graphs == 3
