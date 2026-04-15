"""
diff_pipeline/pipeline.py — Diffusers-based SDXL pipeline.

Replaces the ldm-patched ``BaseModel.apply_model()`` forward path so that a
HuggingFace ``UNet2DConditionModel`` is used instead.

Activated when BOTH conditions are met at load time:
  - ``sd_model.is_sdxl`` is True
  - ``--forge-diffusers-pipeline`` CLI flag is set

Architecture overview
---------------------
::

    DiffPipeline.apply_model(x, t, c_crossattn, y, control, transformer_options)
        │
        ├── sigma preconditioning  (model_sampling.calculate_input / timestep)
        ├── conditioning bridge    (c_crossattn → encoder_hidden_states,
        │                           adm_text_embeds / adm_time_ids → added_cond_kwargs)
        ├── ControlNet mapping     (control["input"] / ["middle"] →
        │                           down_block_additional_residuals / mid_block_additional_residual)
        ├── HF UNet forward        (cross_attention_kwargs carries transformer_options
        │                           for ForgeAttnProcessor to dispatch patch hooks)
        └── sigma postconditioning (model_sampling.calculate_denoised)

ForgeAttnProcessor
------------------
Installed on every ``attn2`` (cross-attention) sub-module of the HF UNet at
construction time.  Receives ``transformer_options`` via
``cross_attention_kwargs`` and dispatches:

  * ``patches_replace["attn2"][(block_name, block_idx, transformer_idx)]``
    — full Q/K/V replacement (IP-Adapter, etc.)
  * ``patches["attn2_patch"]``
    — Q/K/V input modifier before standard attention (ControlLLite, etc.)

Known limitations
-----------------
* LoRA patches (Phase 5): ``_sync_lora()`` reads ``unet_patcher.patches`` on
  every ``apply_model()`` call and installs PEFT adapters on the HF UNet via
  ``load_lora_adapter()``.  DoRA (``dora_scale``) and tucker mid weights are
  not yet handled.
* Block modifier hooks (``add_block_modifier`` / ``add_block_inner_modifier``)
  are Phase 4 work; not yet wired.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn.functional as F


if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from modules_forge.unet_patcher import UnetPatcher

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded optimisation helpers (mirrors modules_forge/stream.py and
# ldm_patched/modules/model_management.py).  Imported once at first use so
# that the module can be imported without a running webui environment.
# ---------------------------------------------------------------------------
def _get_stream_module():
    """Return modules_forge.stream, or None if unavailable."""
    try:
        import modules_forge.stream as _s
        return _s
    except Exception:
        return None

def _get_pin_shared_memory() -> bool:
    """Return True when --pin-shared-memory was requested."""
    try:
        from ldm_patched.modules.model_management import PIN_SHARED_MEMORY
        return bool(PIN_SHARED_MEMORY)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hardcoded SDXL UNet2DConditionModel config
# Source: stabilityai/stable-diffusion-xl-base-1.0 — unet/config.json
# ---------------------------------------------------------------------------

_SDXL_HF_UNET_CONFIG: dict = {
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256,
    "attention_head_dim": [5, 10, 20],
    "block_out_channels": [320, 640, 1280],
    "center_input_sample": False,
    "cross_attention_dim": 2048,
    "down_block_types": [
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ],
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "default",
    "sample_size": 128,
    "transformer_layers_per_block": [1, 2, 10],
    "up_block_types": [
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    ],
    "use_linear_projection": True,
}

# Full ldm unet_config used for unet_to_diffusers() key mapping.
# Must have num_res_blocks / channel_mult / transformer_depth* fields.
# These match the values detect_unet_config() derives from SDXL weights.
_SDXL_LDM_UNET_CONFIG: dict = {
    "num_res_blocks": [2, 2, 2],
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "channel_mult": [1, 2, 4],
    "transformer_depth_middle": 10,
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
}

# ---------------------------------------------------------------------------
# Block address table: HF (b_idx, a_idx) → ldm block index
# Derived from unet_to_diffusers() with num_res_blocks=[2,2,2].
#   down: n = 1 + 3*b_idx + a_idx   (only b_idx=1,2 have attentions)
#   up:   n = 3*b_idx + a_idx        (only b_idx=0,1 have attentions)
# ---------------------------------------------------------------------------
_DOWN_ATTN_LDM_IDX: dict = {
    (1, 0): 4, (1, 1): 5,
    (2, 0): 7, (2, 1): 8,
}
_UP_ATTN_LDM_IDX: dict = {
    (0, 0): 0, (0, 1): 1, (0, 2): 2,
    (1, 0): 3, (1, 1): 4, (1, 2): 5,
}


# ---------------------------------------------------------------------------
# PassthroughAttnProcessor
# ---------------------------------------------------------------------------

class PassthroughAttnProcessor:
    """Wraps AttnProcessor2_0 for self-attention (attn1) blocks.

    Diffusers warns when cross_attention_kwargs keys don't match the processor
    signature.  By declaring ``transformer_options`` explicitly we silence the
    warning; the value is forwarded to the inner processor (which ignores it).
    """

    def __init__(self):
        from diffusers.models.attention_processor import AttnProcessor2_0
        self._inner = AttnProcessor2_0()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        transformer_options=None,  # noqa: ARG002 — declared for diffusers sig check
        **kwargs,
    ):
        return self._inner(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# ForgeAttnProcessor
# ---------------------------------------------------------------------------

class ForgeAttnProcessor:
    """HF attention processor that dispatches Forge patch hooks.

    Installed on every ``attn2`` module in the HF UNet at construction time.
    The LDM block address is burned in so per-call lookup is O(1).

    Supported hooks (same keys as ldm attention.py):
      * ``transformer_options["patches_replace"]["attn2"][(block, idx, t)]``
        Full Q/K/V replacement.  Fn signature: ``(q, k, v, extra_options) → out``
        where tensors are in flat ``(B, seq, heads*dim)`` format.
      * ``transformer_options["patches"]["attn2_patch"]``
        Q/K/V modifier before standard attention.
        Fn signature: ``(q, k, v, extra_options) → (q, k, v)``
    """

    def __init__(self, block_name: str, block_idx: int, transformer_idx: int) -> None:
        self.block_name = block_name
        self.block_idx = block_idx
        self.transformer_idx = transformer_idx
        

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        transformer_options=None,
        **_cross_attention_kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}
        transformer_patches = transformer_options.get("patches", {})
        transformer_patches_replace = transformer_options.get("patches_replace", {})

        # extra_options mirrors ldm's convention so patch fns get the same dict
        extra_options = {
            "block": (self.block_name, self.block_idx),
            "block_index": self.transformer_idx,
            "n_heads": attn.heads,
            "dim_head": attn.to_q.out_features // attn.heads,
        }
        for k, v in transformer_options.items():
            if k not in ("patches", "patches_replace"):
                extra_options[k] = v

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_len, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # --- Forge patch dispatch ---
        block_tuple = (self.block_name, self.block_idx, self.transformer_idx)
        block_pair = (self.block_name, self.block_idx)

        attn2_replace = transformer_patches_replace.get("attn2", {})
        replace_key = (
            block_tuple if block_tuple in attn2_replace else
            block_pair if block_pair in attn2_replace else
            None
        )

        if replace_key is not None:
            # Full replacement — patch fn owns the attention computation.
            # Q/K/V are in flat (B, seq, inner_dim) format, matching ldm convention.
            hidden_states = attn2_replace[replace_key](
                query, key, value, extra_options
            )
        else:
            if "attn2_patch" in transformer_patches:
                for p in transformer_patches["attn2_patch"]:
                    query, key, value = p(query, key, value, extra_options)

            # Standard scaled-dot-product attention
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if getattr(attn, "norm_q", None) is not None:
                query = attn.norm_q(query)
            if getattr(attn, "norm_k", None) is not None:
                key = attn.norm_k(key)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, inner_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        # Projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Standalone auto-offload helper (used by both DiffPipeline and the hijack path)
# ---------------------------------------------------------------------------

def apply_auto_offload_to_unet(hf_unet, device: "torch.device"):
    """Partition an HF UNet2DConditionModel into Group A (device) and Group B (CPU)
    using infer_auto_device_map, then install forward pre/post hooks on Group B so
    each block is streamed to the compute device only while it runs.

    Returns a list of (pre_handle, post_handle) pairs so the caller can remove them.
    """
    from accelerate import infer_auto_device_map

    device_key = str(device)
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]]
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        max_memory = {device_key: int(free_bytes * 0.85), "cpu": "48GiB"}
    elif device.type == "mps":
        max_memory = {device_key: "8GiB", "cpu": "48GiB"}
    else:
        max_memory = None

    device_map = infer_auto_device_map(
        hf_unet,
        max_memory=max_memory,
        no_split_module_classes=["Transformer2DModel", "ResnetBlock2D"],
    )
    log.info("apply_auto_offload_to_unet: raw device_map=%s", device_map)

    # Collect block-level children (expand ModuleLists like down_blocks/up_blocks)
    def iter_blocks(unet):
        for name, child in unet.named_children():
            if isinstance(child, torch.nn.ModuleList):
                for idx in range(len(child)):
                    yield f"{name}.{idx}", child[idx]
            else:
                yield name, child

    group_a = []
    group_b = []
    for path, module in iter_blocks(hf_unet):
        is_cpu = any(
            str(dev) == "cpu"
            for key, dev in device_map.items()
            if key == path or key.startswith(path + ".")
        )
        (group_b if is_cpu else group_a).append((path, module))

    log.info("apply_auto_offload_to_unet: Group A (device)=%s", [p for p, _ in group_a])
    log.info("apply_auto_offload_to_unet: Group B (CPU offload)=%s", [p for p, _ in group_b])

    # Move Group A to device; Group B stays on CPU
    for _, module in group_a:
        module.to(device=device)

    # Install load/unload hooks on Group B
    hook_handles = []
    unet_module_map = {n: m for n, m in hf_unet.named_modules() if n}

    for path, _ in group_b:
        current = unet_module_map.get(path)
        if current is None:
            log.warning("apply_auto_offload_to_unet: could not resolve path '%s' — skipping hooks", path)
            continue

        def make_hooks(dev):
            def pre_hook(m, inp):
                m.to(device=dev)
            def post_hook(m, inp, out):
                m.to(device="cpu")
            return pre_hook, post_hook

        pre_fn, post_fn = make_hooks(device)
        hook_handles.append(current.register_forward_pre_hook(pre_fn))
        hook_handles.append(current.register_forward_hook(post_fn))

    log.info(
        "apply_auto_offload_to_unet: done — %d Group-A blocks, %d Group-B blocks, %d hooks installed",
        len(group_a), len(group_b), len(hook_handles),
    )
    return hook_handles


# ---------------------------------------------------------------------------
# DiffPipeline
# ---------------------------------------------------------------------------

class DiffPipeline():
    """Diffusers-based SDXL pipeline.

    Wraps a ``UnetPatcher`` and replaces the ldm-patched
    ``BaseModel.apply_model()`` forward path so that a HuggingFace
    ``UNet2DConditionModel`` is used instead.

    Attached to ``sd_model.diff_pipeline`` by
    ``forge_loader.load_model_for_a1111()`` when both conditions are met:
      - the loaded checkpoint is SDXL (``sd_model.is_sdxl``)
      - ``--forge-diffusers-pipeline`` CLI flag is set

    Attributes:
        unet_patcher: The ``UnetPatcher`` that owns model_options, LoRA
            patches, ControlNet linked list, and block modifier hooks.
        sd_model: The a1111 shell object (``StableDiffusionModel``).
        model_sampling: Sigma ↔ timestep conversion (reused from ldm).
    """

    # ------------------------------------------------------------------
    # Alternative constructor: wrap an already-built HF UNet
    # (used by the diffusers hijack path in DiffusersModelAdapter)
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_unet(cls, hf_unet, unet_patcher, sd_model) -> "DiffPipeline":
        """Create a DiffPipeline from an already-loaded HF UNet2DConditionModel.

        Unlike the normal constructor this skips the ldm→HF weight conversion
        (``_build_hf_unet``).  Used when a StableDiffusionXLPipeline has been
        loaded by the diffusers path-hijack and we want DiffPipeline.apply_model
        to be the active forward path so ControlNet mapping, ForgeAttnProcessor
        dispatch, LoRA sync, and offload logic all work correctly.
        """
        instance = cls.__new__(cls)
        instance.unet_patcher = unet_patcher
        instance.sd_model = sd_model
        instance.model_sampling = unet_patcher.model.model_sampling

        from modules.shared_cmd_options import cmd_opts
        instance._auto_offload = getattr(cmd_opts, 'forge_diffusers_auto_offload', False)
        instance._sequential_offload = (
            getattr(cmd_opts, 'forge_diffusers_sequential_offload', False)
            and not instance._auto_offload
        )
        instance._offload = (
            getattr(cmd_opts, 'forge_diffusers_offload', False)
            and not instance._sequential_offload
            and not instance._auto_offload
        )
        instance._seq_hooks_installed = False
        instance._compiled = False
        instance._mps_optimized = False
        instance._auto_offload_ready = False
        instance._b_hooks = []
        instance._b_block_paths = []
        instance._tc_ready = False
        instance._autocast_dtype = None

        instance._hf_unet = hf_unet
        instance._install_attn_processors(hf_unet)

        # No ldm key map available on this path — LoRA sync via PEFT is skipped.
        instance._ldm_to_hf = {}
        instance._synced_patches_uuid = None
        instance._active_adapters = []

        log.info(
            "DiffPipeline.from_hf_unet: attached to existing HF UNet — "
            "diffusers hijack path is ACTIVE."
        )
        return instance

    def __init__(self, unet_patcher: UnetPatcher, sd_model: object) -> None:
        self.unet_patcher = unet_patcher
        self.sd_model = sd_model
        self.model_sampling = unet_patcher.model.model_sampling

        from modules.shared_cmd_options import cmd_opts
        # Priority: auto_offload > sequential_offload > offload > default (whole-model on device).
        self._auto_offload: bool = getattr(cmd_opts, 'forge_diffusers_auto_offload', False)
        self._sequential_offload: bool = (
            getattr(cmd_opts, 'forge_diffusers_sequential_offload', False)
            and not self._auto_offload
        )
        self._offload: bool = (
            getattr(cmd_opts, 'forge_diffusers_offload', False)
            and not self._sequential_offload
            and not self._auto_offload
        )
        self._seq_hooks_installed: bool = False
        self._compiled: bool = False
        self._mps_optimized: bool = False

        # Auto-offload state — populated lazily on first apply_model() call.
        self._auto_offload_ready: bool = False
        self._b_hooks: list = []        # registered hook handles (removed on LoRA change)
        self._b_block_paths: list = []  # Group B block paths (for logging)

        # Tensor Core optimisation state — populated lazily on first apply_model() call.
        self._tc_ready: bool = False
        self._autocast_dtype: Optional[torch.dtype] = None  # set by _maybe_setup_tensor_core_opts

        self._hf_unet: UNet2DConditionModel
        log.info("DiffPipeline: building HF UNet2DConditionModel from checkpoint weights …")
        self._hf_unet = self._build_hf_unet(unet_patcher.model)
        self._install_attn_processors(self._hf_unet)

        # Phase 5: LoRA sync state
        self._ldm_to_hf: dict = self._build_ldm_to_hf_map(unet_patcher.model)
        self._synced_patches_uuid = None
        self._active_adapters: list = []   # list of (adapter_name, strength)

        log.info(
            "DiffPipeline attached to sd_model — "
            "Diffusers SDXL path is ACTIVE."
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_hf_unet(self, ldm_model) -> "UNet2DConditionModel":
        """Convert ldm SDXL state dict → HF UNet2DConditionModel."""
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
        from ldm_patched.modules.utils import unet_to_diffusers

        # Merge detected config (has num_res_blocks etc.) with our known full
        # SDXL config so unet_to_diffusers() gets all required keys.
        unet_cfg = dict(_SDXL_LDM_UNET_CONFIG)
        unet_cfg.update(ldm_model.model_config.unet_config)

        key_map = unet_to_diffusers(unet_cfg)  # hf_key → ldm_key

        ldm_sd = ldm_model.diffusion_model.state_dict()

        # Instantiate on CPU; moved to compute device in apply_model().
        hf_unet = UNet2DConditionModel(**_SDXL_HF_UNET_CONFIG)

        # unet_to_diffusers() generates entries for every possible resnet key
        # (conv_shortcut, downsamplers, upsamplers) regardless of whether the
        # block actually has one.  Filter key_map against the HF model's real
        # parameter set so we only copy weights that exist on both sides, and
        # only warn about keys that are genuinely absent.
        hf_model_keys = set(hf_unet.state_dict().keys())

        hf_sd = {}
        missing_ldm = []
        for hf_key, ldm_key in key_map.items():
            if hf_key not in hf_model_keys:
                # Architectural mismatch (e.g. same-channel resnet has no
                # conv_shortcut, last down/up block has no sampler) — skip silently.
                continue
            if ldm_key in ldm_sd:
                hf_sd[hf_key] = ldm_sd[ldm_key]
            else:
                missing_ldm.append(hf_key)

        if missing_ldm:
            log.warning(
                "DiffPipeline: %d HF keys had no matching LDM key (first 5: %s)",
                len(missing_ldm), missing_ldm[:5],
            )

        missing, unexpected = hf_unet.load_state_dict(hf_sd, strict=False)

        if missing:
            log.warning(
                "DiffPipeline: HF UNet missing %d keys (first 5: %s)",
                len(missing), missing[:5],
            )
        if unexpected:
            log.warning(
                "DiffPipeline: HF UNet unexpected %d keys", len(unexpected)
            )

        log.info(
            "DiffPipeline: loaded %d / %d parameter tensors into HF UNet",
            len(hf_sd), len(hf_model_keys),
        )

        # Match dtype from the ldm diffusion model
        try:
            ref_dtype = next(ldm_model.diffusion_model.parameters()).dtype
            hf_unet = hf_unet.to(dtype=ref_dtype)
        except StopIteration:
            pass

        return hf_unet

    def _build_ldm_to_hf_map(self, ldm_model) -> dict:
        """Build reverse key map: 'diffusion_model.X.weight' → HF parameter path.

        ``unet_to_diffusers()`` returns ``{hf_key: ldm_rel_key}``.  We invert it
        so that ``unet_patcher.patches`` keys (which use the full
        ``diffusion_model.`` prefix) can be looked up in O(1).
        """
        from ldm_patched.modules.utils import unet_to_diffusers

        unet_cfg = dict(_SDXL_LDM_UNET_CONFIG)
        unet_cfg.update(ldm_model.model_config.unet_config)
        key_map = unet_to_diffusers(unet_cfg)   # hf_key → ldm_rel_key

        result: dict = {}
        for hf_key, ldm_rel in key_map.items():
            result["diffusion_model." + ldm_rel] = hf_key
        return result

    def _install_attn_processors(self, hf_unet) -> None:
        """Install ForgeAttnProcessor on every attn2 and PassthroughAttnProcessor
        on every attn1 in the HF UNet.

        Both processors declare ``transformer_options`` explicitly in their
        signatures so diffusers does not emit "not expected … will be ignored"
        warnings when cross_attention_kwargs is forwarded from apply_model().
        """
        def _install_block_processors(attn_mod, block_name, ldm_idx):
            for t_idx, tb in enumerate(attn_mod.transformer_blocks):
                tb.attn2.set_processor(
                    ForgeAttnProcessor(block_name, ldm_idx, t_idx)
                )
                tb.attn1.set_processor(PassthroughAttnProcessor())

        # Down blocks
        for b_idx, down_block in enumerate(hf_unet.down_blocks):
            for a_idx, attn_mod in enumerate(getattr(down_block, "attentions", [])):
                ldm_idx = _DOWN_ATTN_LDM_IDX.get((b_idx, a_idx))
                if ldm_idx is None:
                    continue
                _install_block_processors(attn_mod, "input", ldm_idx)

        # Mid block
        for attn_mod in getattr(hf_unet.mid_block, "attentions", []):
            _install_block_processors(attn_mod, "middle", 0)

        # Up blocks
        for b_idx, up_block in enumerate(hf_unet.up_blocks):
            for a_idx, attn_mod in enumerate(getattr(up_block, "attentions", [])):
                ldm_idx = _UP_ATTN_LDM_IDX.get((b_idx, a_idx))
                if ldm_idx is None:
                    continue
                _install_block_processors(attn_mod, "output", ldm_idx)

    def _install_sequential_offload_hooks(self, device) -> None:
        """Install accelerate per-block CPU offload hooks on the HF UNet.

        Each direct child of the UNet (conv_in, down_blocks, mid_block,
        up_blocks, conv_out, …) gets an AlignDevicesHook that moves its
        parameters to *device* just before its forward pass and returns them
        to CPU immediately after.  Peak VRAM ≈ largest single block.

        Called lazily on the first apply_model() invocation so the compute
        device is known.
        """
        from accelerate import cpu_offload

        for child in self._hf_unet.children():
            cpu_offload(child, execution_device=device, offload_buffers=True)

        self._seq_hooks_installed = True
        log.info(
            "DiffPipeline: sequential CPU offload hooks installed (execution device: %s)", device
        )

    # ------------------------------------------------------------------
    # Auto device-map offload (--forge-diffusers-auto-offload)
    # ------------------------------------------------------------------

    def _iter_unet_blocks(self):
        """Yield (path, setter, module) for each block-level child of the HF UNet.

        For ModuleList children (down_blocks, up_blocks) yields one entry per
        item so the granularity matches infer_auto_device_map output.
        ``setter(m)`` replaces the entry in the UNet in-place (used when
        swapping a block for its torch.compile'd wrapper).
        """
        for name, child in self._hf_unet.named_children():
            if isinstance(child, torch.nn.ModuleList):
                for idx in range(len(child)):
                    path = f"{name}.{idx}"
                    # capture loop variables with distinct helper names
                    def make_list_setter(lst, i):
                        def setter(m): lst[i] = m
                        return setter
                    yield path, make_list_setter(child, idx), child[idx]
            else:
                def make_attr_setter(attr):
                    def setter(m): setattr(self._hf_unet, attr, m)
                    return setter
                yield name, make_attr_setter(name), child

    def _setup_auto_offload(self, device: torch.device) -> None:
        """Partition UNet blocks into Group A (device) and Group B (CPU),
        compile each group regionally, and install load/unload hooks on Group B.

        Called lazily on the first ``apply_model()`` invocation so the compute
        device is known.  Must be re-called (via ``_reset_auto_offload``) after
        any structural change to the UNet (e.g. LoRA adapter swap).
        """
        print(f"[_setup_auto_offload] ENTERED — device={device}")
        from accelerate import infer_auto_device_map

        # --- 1. Determine max_memory budget ---
        # infer_auto_device_map expects Dict[int | str, int | str] keys.
        # Use str(device) (e.g. "cuda:0") rather than a torch.device object.
        device_key = str(device)
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            max_vram = int(free_bytes * 0.85)
            max_memory: Optional[dict] = {device_key: max_vram, "cpu": "48GiB"}
        elif device.type == "mps":
            # MPS doesn't expose free memory; use a conservative fixed budget.
            max_memory = {device_key: "8GiB", "cpu": "48GiB"}
        else:
            max_memory = None
        print(f"[_setup_auto_offload] max_memory={max_memory}")

        # --- 2. Infer device map ---
        # no_split_module_classes prevents splitting a Transformer or ResnetBlock
        # across two devices, which would require cross-device tensor copies mid-block.
        device_map: dict = infer_auto_device_map(
            self._hf_unet,
            max_memory=max_memory,
            no_split_module_classes=["Transformer2DModel", "ResnetBlock2D"],
        )
        log.info("DiffPipeline auto-offload: raw device_map = %s", device_map)

        # --- 3. Classify each block into Group A or Group B ---
        # A block is Group B if *any* of its leaf entries in device_map is "cpu".
        group_a: list[tuple] = []   # (path, setter, module)
        group_b: list[tuple] = []

        for path, setter, module in self._iter_unet_blocks():
            is_cpu = any(
                str(dev) == "cpu"
                for key, dev in device_map.items()
                if key == path or key.startswith(path + ".")
            )
            if is_cpu:
                group_b.append((path, setter, module))
            else:
                group_a.append((path, setter, module))

        log.info(
            "DiffPipeline auto-offload: Group A (device) = %s",
            [p for p, _, _ in group_a],
        )
        log.info(
            "DiffPipeline auto-offload: Group B (CPU offload) = %s",
            [p for p, _, _ in group_b],
        )

        # --- 4. Move Group A to device; Group B stays on CPU ---
        for _, _, module in group_a:
            module.to(device=device)

        # --- 5. Regional compile (skip MPS — Metal inductor issues) ---
        # Compile BEFORE installing hooks so hooks attach to the OptimizedModule.
        if device.type != "mps":
            n_blocks = len(group_a) + len(group_b)
            log.info(
                "DiffPipeline: torch.compile starting — %d UNet blocks "
                "(mode=reduce-overhead). This may take 1–3 minutes on first run …",
                n_blocks,
            )
            for path, setter, module in group_a + group_b:
                log.info("DiffPipeline: compiling block '%s' …", path)
                compiled = torch.compile(module, mode="reduce-overhead", fullgraph=False)
                setter(compiled)
            log.info(
                "DiffPipeline: torch.compile done — %d blocks compiled. "
                "Subsequent runs will reuse the cached graph.",
                n_blocks,
            )

        # --- 6. Install load/unload hooks on Group B ---
        # Re-fetch modules after possible compile replacement.
        self._b_hooks.clear()
        self._b_block_paths.clear()

        # Build a name→module lookup after compile replacements.
        # named_modules() walks the entire tree; we only need direct matches.
        unet_module_map: dict[str, torch.nn.Module] = {
            n: m for n, m in self._hf_unet.named_modules() if n
        }

        for path, *_ in group_b:
            current = unet_module_map.get(path)
            if current is None:
                log.warning(
                    "DiffPipeline auto-offload: could not resolve Group B path '%s' "
                    "after compile — skipping hooks for this block", path
                )
                continue

            def make_hooks(dev: torch.device):
                def pre_hook(m: torch.nn.Module, inp: Any) -> None:
                    m.to(device=dev)
                def post_hook(m: torch.nn.Module, inp: Any, out: Any) -> None:
                    m.to(device="cpu")
                return pre_hook, post_hook

            pre_fn, post_fn = make_hooks(device)
            pre_handle = current.register_forward_pre_hook(pre_fn)
            post_handle = current.register_forward_hook(post_fn)
            self._b_hooks.extend([pre_handle, post_handle])
            self._b_block_paths.append(path)

        self._auto_offload_ready = True
        log.info(
            "DiffPipeline auto-offload: setup complete — %d Group-A blocks, "
            "%d Group-B blocks, %d hooks installed",
            len(group_a), len(group_b), len(self._b_hooks),
        )

    def _reset_auto_offload(self) -> None:
        """Remove Group B hooks and mark auto-offload as needing re-setup.

        Called whenever the UNet structure changes (e.g. LoRA adapter swap)
        so that ``_setup_auto_offload`` re-runs on the next forward pass.
        """
        for handle in self._b_hooks:
            handle.remove()
        self._b_hooks.clear()
        self._b_block_paths.clear()
        self._auto_offload_ready = False

    # ------------------------------------------------------------------
    # Phase 5 — LoRA synchronisation
    # ------------------------------------------------------------------

    def _remove_lora_adapters(self) -> None:
        """Delete all PEFT adapters previously loaded onto the HF UNet."""
        for name, _ in self._active_adapters:
            try:
                self._hf_unet.delete_adapter(name)
            except Exception:
                pass
        self._active_adapters.clear()

    def _sync_lora(self) -> None:
        """Sync ldm LoRA patches → PEFT adapters on the HF UNet.

        Called at the top of every ``apply_model()`` invocation.  No-ops when
        ``unet_patcher.patches_uuid`` has not changed since the last call.

        Each "depth" in the stacked patch list becomes one named PEFT adapter
        (``forge_lora_0``, ``forge_lora_1``, …).  After loading all adapters
        ``set_adapters()`` applies the per-adapter strength weights.
        """
        from ldm_patched.modules.weight_adapter.lora import LoRAAdapter

        # The sampler clones the unet_patcher and adds LoRA patches to the clone,
        # giving it a new patches_uuid.  self.unet_patcher still points to the
        # original (unpatched) patcher whose uuid never changes.  Use
        # model.current_patcher (set by ModelPatcher.pre_run() to the active clone)
        # so we see the real patch state.
        # forge_objects.unet is updated by LoRA loading (networks.py clones + patches it).
        # self.unet_patcher is the original at init time and never gets LoRA patches.
        # current_patcher is only set by ModelPatcher.pre_run(), which is NOT called in
        # the k-diffusion sampler path — so we cannot rely on it.
        # Reading forge_objects.unet directly gives us the live, LoRA-patched patcher.
        forge_objects = getattr(self.sd_model, "forge_objects", None)
        active_patcher = (
            forge_objects.unet
            if forge_objects is not None and getattr(forge_objects, "unet", None) is not None
            else self.unet_patcher
        )

        patches = getattr(active_patcher, "patches", {})
        patches_uuid = getattr(active_patcher, "patches_uuid", None)

        log.debug(
            "[_sync_lora] active_patcher id=%d patches_keys=%d patches_uuid=%s synced_uuid=%s same=%s",
            id(active_patcher), len(patches), patches_uuid, self._synced_patches_uuid,
            patches_uuid == self._synced_patches_uuid,
        )

        if patches_uuid == self._synced_patches_uuid:
            return   # nothing changed

        self._remove_lora_adapters()
        self._synced_patches_uuid = patches_uuid
        # LoRA adapter changes structurally modify the UNet (delete_adapter /
        # load_lora_adapter), so any existing compiled graph is invalid.
        # Force a recompile on the next forward pass.
        self._compiled = False

        if not patches:
            return

        max_depth = max(len(v) for v in patches.values())

        for depth in range(max_depth):
            state_dict: dict = {}
            network_alphas: dict = {}
            adapter_strength = None

            for ldm_key, patch_list in patches.items():
                if depth >= len(patch_list):
                    continue
                strength_patch, adapter, _strength_model, _offset, _fn = patch_list[depth]
                if not isinstance(adapter, LoRAAdapter):
                    continue
                hf_key = self._ldm_to_hf.get(ldm_key)
                if hf_key is None:
                    continue

                lora_up    = adapter.weights[0]  # lora_up.weight   → PEFT lora_B  [out, r]
                lora_down  = adapter.weights[1]  # lora_down.weight → PEFT lora_A  [r, in]
                alpha      = adapter.weights[2]  # scalar or None
                # weights[3] = mid (tucker), weights[4] = dora_scale, weights[5] = reshape
                dora_scale = adapter.weights[4]  # [1, out_dim] or None (DoRA magnitude)
                r = lora_down.shape[0]
                alpha_val = float(alpha) if alpha is not None else float(r)

                module_path = hf_key[: -len(".weight")]
                state_dict[f"{module_path}.lora_A.weight"] = lora_down
                state_dict[f"{module_path}.lora_B.weight"] = lora_up
                network_alphas[module_path] = alpha_val

                # DoRA: add magnitude vector so PEFT detects use_dora=True and applies
                # weight decomposition.  Kohya/A1111 stores it as [1, out_dim]; squeeze
                # to [out_dim] which is the shape PEFT's lora_magnitude_vector expects.
                if dora_scale is not None:
                    mag = dora_scale.squeeze(0) if dora_scale.dim() > 1 and dora_scale.shape[0] == 1 else dora_scale
                    state_dict[f"{module_path}.lora_magnitude_vector"] = mag

                if adapter_strength is None:
                    adapter_strength = float(strength_patch)

            if not state_dict:
                continue

            adapter_name = f"forge_lora_{depth}"
            self._hf_unet.load_lora_adapter(
                state_dict,
                network_alphas=network_alphas,
                adapter_name=adapter_name,
                low_cpu_mem_usage=False,
            )
            

            self._active_adapters.append((adapter_name, adapter_strength or 1.0))

        if self._active_adapters:
            names = [n for n, _ in self._active_adapters]
            weights = [w for _, w in self._active_adapters]
            self._hf_unet.set_adapters(names, weights)
            log.info("DiffPipeline: activated %d LoRA adapter(s): %s", len(names), names)
    # ------------------------------------------------------------------
    # Public interface (mirrors BaseModel.apply_model signature)
    # ------------------------------------------------------------------

    def _maybe_setup_tensor_core_opts(self, device: "torch.device", unet_dtype: "torch.dtype") -> None:
        """Detect GPU Tensor Core capability and apply compatible optimisations once.

        Called lazily on the first ``apply_model()`` after device placement so
        the compute device is known.

        Actions taken (each guarded so it's only applied when appropriate):

        * **Ampere+ (sm_80+) with fp32 UNet** — enables TF32 for matmul and
          cuDNN convolutions.  TF32 uses Tensor Cores internally while keeping
          fp32 inputs/outputs, so existing precision is preserved at higher
          throughput.

        * **Volta+ (sm_70+) with fp16/bf16 UNet** — stores the half-precision
          dtype so ``apply_model()`` wraps the UNet forward in
          ``torch.autocast``.  The UNet weights are already in the target dtype;
          the autocast context ensures *all* intermediate ops (time-step
          embedding, norms, etc.) also stay in half-precision and hit the TC
          path instead of silently widening to fp32.

        SDXL channel sizes (320 / 640 / 1280) are all divisible by 8, and the
        attention head-dim is 64 — both satisfy the minimum alignment required
        for fp16/bf16 Tensor Core GEMMs, so no padding is needed.
        """
        if self._tc_ready:
            return
        self._tc_ready = True

        # --- MPS path: enable fp16 autocast to minimise memory and reduce swap ---
        # torch.autocast on MPS keeps intermediate activations in float16 instead of
        # silently widening to float32, halving the active-memory footprint and
        # reducing the chance that macOS swaps UNet tensors to the unified memory
        # pool's slow region.  We guard with is_autocast_available() so this is
        # skipped on PyTorch builds that do not yet expose MPS autocast (pre-2.3).
        if device.type == "mps":
            mps_autocast_ok = False
            try:
                mps_autocast_ok = torch.amp.autocast_mode.is_autocast_available("mps")
            except Exception:
                pass  # older PyTorch — attribute not present

            if mps_autocast_ok:
                # Always prefer float16 on MPS regardless of the stored UNet dtype.
                # A float32 UNet still benefits: autocast casts eligible ops to fp16
                # so peak live memory is cut roughly in half, reducing swap pressure.
                self._autocast_dtype = torch.float16

                # Log the recommended memory ceiling so the user can see the budget.
                try:
                    rec_mem_bytes = torch.mps.recommended_max_memory()
                    rec_mem_gib = rec_mem_bytes / (1024 ** 3)
                    mem_note = f"  (torch.mps.recommended_max_memory = {rec_mem_gib:.1f} GiB)"
                except Exception:
                    mem_note = ""

                msg = (
                    f"MPS autocast enabled — UNet forward wrapped in "
                    f"torch.autocast(device_type='mps', dtype=float16) "
                    f"to keep activations in fp16 and minimise swap{mem_note}"
                )
                print(f"\n[DiffPipeline] {msg}\n")
                log.info("DiffPipeline: %s", msg)
            else:
                self._autocast_dtype = None
                log.info(
                    "DiffPipeline: MPS autocast not available on this PyTorch build "
                    "(torch.amp.autocast_mode.is_autocast_available('mps') = False) "
                    "— running without autocast."
                )
            return

        if device.type != "cuda" or not torch.cuda.is_available():
            self._autocast_dtype = None
            return

        major, minor = torch.cuda.get_device_capability(device)
        gpu_name = torch.cuda.get_device_name(device)
        msgs: list[str] = []

        # --- Ampere+ (sm_80+): TF32 for fp32 UNet ---
        if major >= 8 and unet_dtype == torch.float32:
            if not torch.backends.cuda.matmul.allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                msgs.append("matmul TF32 enabled  (torch.backends.cuda.matmul.allow_tf32 = True)")
            if not torch.backends.cudnn.allow_tf32:
                torch.backends.cudnn.allow_tf32 = True
                msgs.append("cuDNN conv TF32 enabled  (torch.backends.cudnn.allow_tf32 = True)")

        # --- Volta+ (sm_70+): autocast for fp16 / bf16 UNet ---
        if major >= 7 and unet_dtype in (torch.float16, torch.bfloat16):
            self._autocast_dtype = unet_dtype
            msgs.append(
                f"UNet forward wrapped in torch.autocast({unet_dtype})  "
                f"— keeps time-embed / norm ops in half-precision for consistent TC usage"
            )
        else:
            self._autocast_dtype = None

        # --- Alignment note ---
        if major >= 7:
            msgs.append(
                "SDXL channel dims 320/640/1280 are ÷8 and head-dim=64 is ÷8  "
                "— fp16/bf16 TC alignment satisfied ✓"
            )

        if msgs:
            lines = "\n".join(f"  • {m}" for m in msgs)
            print(
                f"\n[DiffPipeline] Tensor Core optimisations applied"
                f" ({gpu_name}, sm_{major}{minor}, unet_dtype={unet_dtype}):\n"
                f"{lines}\n"
            )
            log.info(
                "DiffPipeline: TC opts (%s sm_%d%d dtype=%s): %s",
                gpu_name, major, minor, unet_dtype, "; ".join(msgs),
            )
        else:
            log.info(
                "DiffPipeline: No TC optimisations applied"
                " (device=%s, sm_%d%d, unet_dtype=%s — no matching rule)",
                gpu_name, major, minor, unet_dtype,
            )

    def apply_model(
        self,
        x,
        t,
        c_concat=None,
        c_crossattn=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        """Forward pass through the HF Diffusers UNet.

        Accepts the same arguments as ``BaseModel.apply_model()`` so the
        sampler calling convention is unchanged.
        """
        if transformer_options is None:
            transformer_options = {}

        log.debug("[DiffPipeline] apply_model called — sigma=%s, x.shape=%s", t, x.shape)

        sigma = t

        # --- 1. Sigma preconditioning (identical to ldm _apply_model) ---
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        # Integer timestep for HF UNet
        timestep = self.model_sampling.timestep(sigma).float()

        # --- 2. Device placement / offload setup ---
        device = x.device
        if self._auto_offload:
            # Partition UNet into Group A (device) / Group B (CPU) with hooks.
            # Done lazily on the first call; re-runs after LoRA swaps.
            if not self._auto_offload_ready:
                self._setup_auto_offload(device)
        elif self._sequential_offload:
            # Install per-block accelerate hooks on the first call.
            # After that the hooks handle GPU↔CPU movement automatically;
            # the UNet itself stays on CPU between blocks.
            if not self._seq_hooks_installed:
                self._install_sequential_offload_hooks(device)
        else:
            # Whole-model placement: move to compute device if needed.
            if next(self._hf_unet.parameters()).device != device:
                self._hf_unet.to(device=device)

            # MPS-specific optimizations — applied once after the UNet lands on device.
            if device.type == "mps" and not self._mps_optimized:
                if hasattr(self._hf_unet, "enable_attention_slicing"):
                    self._hf_unet.enable_attention_slicing()
                    log.info("DiffPipeline: MPS — enabled attention slicing")
                self._mps_optimized = True


        # --- 2b. Tensor Core setup (runs once per session) ---
        dtype = next(self._hf_unet.parameters()).dtype
        self._maybe_setup_tensor_core_opts(device, dtype)

        # Dtype from HF UNet (works regardless of current parameter device).
        xc = xc.to(dtype)
        if c_crossattn is not None:
            c_crossattn = c_crossattn.to(dtype)

        # --- 3. Conditioning bridge ---
        encoder_hidden_states = c_crossattn  # (B, seq, 2048)

        # Use raw ADM components emitted by SDXL.extra_conds() if available.
        # Fallback: split y (2816-dim Fourier-embedded) — text_embeds correct,
        # time_ids will be wrong (already Fourier-embedded), but still usable.
        adm_text_embeds = kwargs.get("adm_text_embeds", None)
        adm_time_ids = kwargs.get("adm_time_ids", None)

        if adm_text_embeds is not None and adm_time_ids is not None:
            added_cond_kwargs = {
                "text_embeds": adm_text_embeds.to(dtype),
                "time_ids": adm_time_ids.to(dtype),
            }
        else:
            y = kwargs.get("y", None)
            if y is not None:
                y = y.to(dtype)
                added_cond_kwargs = {
                    "text_embeds": y[:, :1280],
                    "time_ids": torch.zeros(
                        y.shape[0], 6, device=device, dtype=dtype
                    ),
                }
                log.debug(
                    "DiffPipeline: adm_raw not found; falling back to y-split "
                    "(time conditioning may be imprecise)"
                )
            else:
                # Neither raw ADM tensors nor legacy y — build a zero fallback so
                # the SDXL UNet never receives added_cond_kwargs=None, which causes
                # the add_embedding module to produce NaN.
                log.warning(
                    "[DiffPipeline] No pooled conditioning found (adm_text_embeds/y both absent)"
                    " — using zero fallback for added_cond_kwargs."
                )
                batch = x.shape[0]
                added_cond_kwargs = {
                    "text_embeds": torch.zeros(batch, 1280, device=device, dtype=dtype),
                    "time_ids":    torch.zeros(batch,    6, device=device, dtype=dtype),
                }

        # --- 4. ControlNet residual mapping ---
        # ldm: control["input"] is a list that gets pop()d (reverse order).
        # HF:  down_block_additional_residuals is a tuple in forward order.
        down_block_residuals: Optional[tuple] = None
        mid_block_residual = None

        if control is not None:
            raw_input = control.get("input", [])
            if raw_input:
                # Reverse the list (ldm pops from end = first block first).
                reversed_residuals = [r for r in reversed(raw_input) if r is not None]
                if reversed_residuals:
                    down_block_residuals = tuple(reversed_residuals)

            raw_middle = control.get("middle", [])
            if raw_middle and raw_middle[0] is not None:
                mid_block_residual = raw_middle[0]

        # --- 5. LoRA sync (Phase 5) ---
        self._sync_lora()

        # --- 5b. Diffusers optimization injection ---
        # Override apply_diffusers_optimization() or monkey-patch it to inject
        # custom Diffusers optimizations (xformers, flash-attn, torch.compile, etc.).
        # Called after device placement and LoRA sync, before the forward pass.
        self.apply_diffusers_optimization(self._hf_unet)

        # --- 6. Whole-model torch.compile (single-device path only) ---
        # Compile is safe when the UNet runs in bf16 or fp32 — both have the
        # same fp32 exponent range so inductor kernel fusion cannot overflow.
        # fp16 is excluded: its narrow exponent range (~65504 max) causes
        # silent NaN when inductor fuses ops, as observed with NoobAI XL.
        # (The VAE NaN fix in load_model.py applies the same reasoning to the
        # VAE decoder.)  On offload paths the UNet blocks are spread across
        # devices so whole-model compile does not apply there.
        if not self._compiled and not self._auto_offload and not self._sequential_offload:
            unet_dtype = next(self._hf_unet.parameters()).dtype
            if unet_dtype == torch.float16:
                log.info(
                    "DiffPipeline: torch.compile skipped — UNet is in fp16 "
                    "(narrow exponent range risks NaN under inductor fusion). "
                    "Switch to bf16/fp32 hardware to enable compile.",
                )
                self._compiled = True  # mark so we don't re-check every step
            else:
                log.warning(
                    "DiffPipeline: torch.compile is about to begin (UNet dtype=%s). "
                    "The UI will appear frozen for 1–3 minutes while the inductor "
                    "captures and optimises the compute graph — this is normal. "
                    "Subsequent runs will reuse the cached graph and start instantly.",
                    unet_dtype,
                )
                print(
                    f"\n[DiffPipeline] *** torch.compile starting (dtype={unet_dtype}) ***\n"
                    f"    The UI will be unresponsive for ~1-3 min during graph capture.\n"
                    f"    This only happens once per session.\n"
                )
                if device == "mps":
                    # MPS don't have max auto tune, inductor use as default
                    self._hf_unet = torch.compile(
                    self._hf_unet, mode="reduce-overhead", fullgraph=False
                )
                else:
                    self._hf_unet = torch.compile(
                        self._hf_unet, mode="reduce-overhead", fullgraph=False
                    )
                self._compiled = True
                log.warning("DiffPipeline: torch.compile finished — graph cached, normal speed resumes.")
                print("\n[DiffPipeline] *** torch.compile done — generation will proceed normally. ***\n")

        # Debug: log added_cond_kwargs shapes before the forward pass.
        log.debug(
            "[DiffPipeline] added_cond_kwargs keys=%s text_embeds.shape=%s time_ids.shape=%s",
            list(added_cond_kwargs.keys()) if added_cond_kwargs is not None else "None",
            tuple(added_cond_kwargs["text_embeds"].shape) if added_cond_kwargs and "text_embeds" in added_cond_kwargs else "absent",
            tuple(added_cond_kwargs["time_ids"].shape)    if added_cond_kwargs and "time_ids"    in added_cond_kwargs else "absent",
        )

        # --- 6b. cuda-stream: run HF UNet forward on the dedicated compute stream ---
        # Mirrors the modules_forge/stream.py approach used by the ldm path.
        # When --cuda-stream is active, modules_forge.stream.current_stream holds
        # a pre-warmed torch.cuda.Stream.  Wrapping the forward pass in that
        # stream context lets the CUDA scheduler overlap this compute with any
        # async memory transfers happening on the mover_stream.
        _stream_mod = _get_stream_module()
        _use_stream = (
            _stream_mod is not None
            and getattr(_stream_mod, "using_stream", False)
            and getattr(_stream_mod, "current_stream", None) is not None
        )

        def _unet_forward():
            return self._hf_unet(
                sample=xc,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_residuals,
                mid_block_additional_residual=mid_block_residual,
                cross_attention_kwargs={"transformer_options": transformer_options},
                return_dict=False,
            )

        # Wrap forward in autocast when _maybe_setup_tensor_core_opts decided it
        # would help (fp16/bf16 UNet on Volta+).  This keeps time-embed and norm
        # ops in half-precision so they hit the TC path rather than widening to fp32.
        def _run_forward():
            if self._autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=self._autocast_dtype):
                    return _unet_forward()
            return _unet_forward()

        if _use_stream:
            with torch.cuda.stream(_stream_mod.current_stream):
                unet_output = _run_forward()
            # Synchronise so downstream ops on the default stream see the result.
            _stream_mod.current_stream.synchronize()
        else:
            unet_output = _run_forward()

        model_output = unet_output[0].float()

        # Debug: check for NaN in UNet output.
        if torch.isnan(model_output).any():
            nan_count = torch.isnan(model_output).sum().item()
            valid = model_output[~torch.isnan(model_output)]
            log.warning(
                "[DiffPipeline] NaN in UNet output! nan_count=%d  valid_min=%.3f  valid_max=%.3f",
                nan_count,
                valid.min().item() if valid.numel() > 0 else float("nan"),
                valid.max().item() if valid.numel() > 0 else float("nan"),
            )
        else:
            log.debug(
                "[DiffPipeline] UNet output OK: min=%.3f  max=%.3f",
                model_output.min().item(), model_output.max().item(),
            )

        # --- 7. Sigma postconditioning (identical to ldm _apply_model) ---
        result = self.model_sampling.calculate_denoised(sigma, model_output, x)

        # Debug: check for NaN after calculate_denoised.
        nan_result = torch.isnan(result).sum().item()
        if nan_result > 0:
            print(
                f"[DiffPipeline] NaN after calculate_denoised! nan={nan_result}  "
                f"sigma={sigma.item() if sigma.numel()==1 else float(sigma.max())}  "
                f"x_nan={torch.isnan(x).sum().item()}"
            )

        # --- 8. Offload HF UNet back to CPU if requested ---
        if self._offload:
            # Use non_blocking when cuda-stream is active so the CPU←GPU copy
            # is enqueued asynchronously and doesn't stall the compute stream.
            self._hf_unet = self._hf_unet.to(device="cpu", non_blocking=_use_stream)

            # --pin-shared-memory: pin the offloaded weights to page-locked
            # (pinned) CPU memory so the next GPU upload uses DMA and avoids
            # a kernel-to-user copy.  Mirrors model_patcher.py's usage of
            # PIN_SHARED_MEMORY.  We only pin once — if the parameters are
            # already pinned this is a no-op.
            if _get_pin_shared_memory():
                for p in self._hf_unet.parameters():
                    if not p.data.is_pinned():
                        p.data = p.data.pin_memory()

        return result

    # ------------------------------------------------------------------
    # Diffusers optimization injection point
    # ------------------------------------------------------------------

    def get_pipeline(self) -> "UNet2DConditionModel":
        """Return the underlying HF ``UNet2DConditionModel``.

        Use this to apply Diffusers-level optimizations before inference,
        for example::

            pipeline = diff_pipeline.get_pipeline()
            pipeline.enable_xformers_memory_efficient_attention()
            # or: pipeline.enable_flash_attn_2(), torch.compile(pipeline), etc.
            
        To future developer: DON'T DELETE THIS! 
        UNTIL I FIND A WAY PROPERLY LABEL WHAT THE HECK THE HF NET TYPE IS
        """
        return self._hf_unet

    def apply_diffusers_optimization(self, _hf_unet: "UNet2DConditionModel") -> None:
        """Stub — override or monkey-patch this to inject custom Diffusers optimizations.

        Called once per ``apply_model()`` invocation, after the UNet has been
        moved to the correct compute device and LoRA adapters have been synced,
        but before the HF UNet forward pass.  The ``hf_unet`` argument is the
        same object as ``self.get_pipeline()``.

        Example usage (monkey-patch from outside)::

            from diff_pipeline.pipeline import DiffPipeline

            def my_opt(self, unet):
                unet.enable_xformers_memory_efficient_attention()

            DiffPipeline.apply_diffusers_optimization = my_opt

        Or subclass ``DiffPipeline`` and override this method directly.
        """
        #EXPERIEMENT: TaylorSeer Cache
        
        
        pass  # no-op by default

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True once the HF UNet is wired and apply_model() is usable."""
        return self._hf_unet is not None

    def __repr__(self) -> str:
        status = "active" if self.is_active() else "scaffold-only"
        return f"DiffPipeline({status})"
