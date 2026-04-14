"""
diff_pipeline/adapter.py

DiffusersModelAdapter — wraps a diffusers pipeline (e.g. StableDiffusionXLPipeline)
in an object that satisfies the ldm-style interface expected by stable-diffusion-webui-reForge.

The webui accesses dozens of attributes on `shared.sd_model`. When the diffusers path
hijack replaces normal ldm loading, the raw pipeline object lacks all of these. This
adapter bridges the gap so the UI and processing code don't crash.

Generation path
---------------
All actual denoising is delegated to the underlying diffusers pipeline via
`apply_model`.  The VAE and text-encoding methods wrap the diffusers components
so that `processing.py` (which calls `encode_first_stage`, `decode_first_stage`,
`get_learned_conditioning`) works without modification.

Usage
-----
Wrap the loaded pipeline before returning it from a path-hijack loader:

    from diff_pipeline.adapter import DiffusersModelAdapter
    pipe = StableDiffusionXLPipeline.from_single_file(...)
    return DiffusersModelAdapter(pipe, checkpoint_info, model_type="sdxl")
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import torch


class _FakeEmaScope:
    """No-op context manager satisfying the sd_model.ema_scope() interface."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


class _FakeModel:
    """Minimal stand-in for sd_model.model (the ldm DiffusionWrapper).

    processing.py reads .conditioning_key and extensions read .diffusion_model.
    After the UnetPatcher is built we point diffusion_model at the same
    DiffusersUnetModel wrapper so both paths see the same object.
    """
    def __init__(self, pipe):
        self._pipe = pipe
        self.conditioning_key = "crossattn"   # SDXL always uses cross-attention

    @property
    def diffusion_model(self):
        # Prefer the patcher's model wrapper if already built (set by adapter init)
        return getattr(self, "_diffusion_model", getattr(self._pipe, "unet", None))

    @diffusion_model.setter
    def diffusion_model(self, val):
        self._diffusion_model = val


class _FakeCondStageModel:
    """Thin wrapper around diffusers text encoders that satisfies the
    cond_stage_model interface expected by processing.py and extensions.

    The token-counting methods (process_texts / get_target_prompt_token_count)
    use the pipeline's tokenizer so the UI token counter shows accurate counts
    instead of crashing.  The standard CLIP chunk length is 75 tokens (77
    including BOS/EOS); SDXL uses the same limit.
    """

    CHUNK_LENGTH = 75  # standard CLIP tokens per chunk (excl. BOS/EOS)

    def __init__(self, pipe):
        self._pipe = pipe

    # ---- token counting --------------------------------------------------

    def process_texts(self, texts: list[str]):
        """Tokenise *texts* and return (batch_chunks, max_token_count).

        The UI only needs the token count; we return a minimal structure so
        the caller doesn't crash.  Each 'chunk' is a plain object with a
        ``tokens`` attribute.
        """
        tokenizer = getattr(self._pipe, "tokenizer", None)
        token_count = 0

        class _Chunk:
            def __init__(self, tokens):
                self.tokens = tokens
                self.multipliers = [1.0] * len(tokens)

        batch_chunks = []
        for text in texts:
            if tokenizer is not None:
                try:
                    ids = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                    ).input_ids
                    n = len(ids)
                except Exception:
                    n = 0
            else:
                # rough estimate: ~0.75 tokens per char
                n = max(0, int(len(text) * 0.75))

            token_count = max(token_count, n)
            batch_chunks.append([_Chunk(list(range(n)))])

        return batch_chunks, token_count

    def get_target_prompt_token_count(self, token_count: int) -> int:
        """Return the next chunk-aligned token ceiling for the counter display."""
        import math
        return math.ceil(max(token_count, 1) / self.CHUNK_LENGTH) * self.CHUNK_LENGTH

    # ---- tokenize --------------------------------------------------------

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """Return raw token-id lists for *texts* using pipe.tokenizer."""
        tokenizer = getattr(self._pipe, "tokenizer", None)
        result = []
        for text in texts:
            if tokenizer is not None:
                try:
                    ids = tokenizer(text, add_special_tokens=False,
                                    truncation=False).input_ids
                    result.append(ids)
                    continue
                except Exception:
                    pass
            result.append([])
        return result

    # ---- textual inversion support ---------------------------------------

    def encode_embedding_init_text(self, init_text: str, nvpt: int) -> "torch.Tensor":
        """Return the token embedding(s) for *init_text*, truncated/padded to *nvpt*.

        Used by textual_inversion to determine the expected embedding shape.
        For SDXL (CLIP-L 768 + CLIP-G 1280 = 2048 dims) we replicate what the
        real GeneralConditioner does: concatenate embeddings from both encoders.
        """
        import torch
        results = []

        # CLIP-L (text_encoder, tokenizer)
        te1 = getattr(self._pipe, "text_encoder", None)
        tok1 = getattr(self._pipe, "tokenizer", None)
        if te1 is not None and tok1 is not None:
            try:
                enc = tok1(init_text, return_tensors="pt",
                           add_special_tokens=True, truncation=True,
                           max_length=tok1.model_max_length)
                ids = enc.input_ids.to(next(te1.parameters()).device)
                # token embeddings before transformer
                emb = te1.text_model.embeddings.token_embedding(ids)  # (1, T, 768)
                # take up to nvpt non-special tokens
                emb = emb[:, 1:nvpt + 1, :]  # skip BOS
                results.append(emb)
            except Exception:
                results.append(torch.zeros(1, nvpt, 768))

        # CLIP-G (text_encoder_2, tokenizer_2)
        te2 = getattr(self._pipe, "text_encoder_2", None)
        tok2 = getattr(self._pipe, "tokenizer_2", None)
        if te2 is not None and tok2 is not None:
            try:
                enc = tok2(init_text, return_tensors="pt",
                           add_special_tokens=True, truncation=True,
                           max_length=tok2.model_max_length)
                ids = enc.input_ids.to(next(te2.parameters()).device)
                emb = te2.text_model.embeddings.token_embedding(ids)  # (1, T, 1280)
                emb = emb[:, 1:nvpt + 1, :]
                results.append(emb)
            except Exception:
                results.append(torch.zeros(1, nvpt, 1280))

        if not results:
            # Fallback: return a zero tensor with SDXL combined dim
            return torch.zeros(1, nvpt, 2048)

        return torch.cat(results, dim=2)  # (1, nvpt, 768+1280)

    # ---- conditioning ----------------------------------------------------

    def encode(self, tokens):
        raise NotImplementedError(
            "DiffusersModelAdapter: cond_stage_model.encode() not yet implemented. "
            "Use get_learned_conditioning() on the adapter instead."
        )

    def get_learned_conditioning(self, text: list[str]):
        # Returns a dict {'crossattn': (B,seq,hidden), 'vector': (B,pooled)}
        # so prompt_parser can index it per-prompt and reconstruct_*_batch
        # stacks them back into a DictWithShape for cond_from_a1111_to_patched_ldm.
        return _encode_prompts(self._pipe, text)

    # Extensions may try to iterate over embedders
    @property
    def embedders(self):
        return []


class _DiffusersUnetModel:
    """Minimal model container that ModelPatcher / UnetPatcher expect on .model.

    ModelPatcher.__init__ reads several attributes from the wrapped model object
    (device, model_loaded_weight_memory, lowvram_patch_counter, model_lowvram,
    current_weight_patches_uuid) and calls model.memory_required().  This class
    provides all of them so a real UnetPatcher can be constructed around the
    diffusers UNet2DConditionModel.
    """

    def __init__(self, diffusers_unet, device: torch.device):
        self.diffusion_model = diffusers_unet   # accessed by forge extensions
        self.device = device

        # Attributes ModelPatcher.__init__ sets if missing — pre-set them so
        # the patcher doesn't have to patch the diffusers nn.Module directly.
        self.model_loaded_weight_memory = 0
        self.lowvram_patch_counter = 0
        self.model_lowvram = False
        self.current_weight_patches_uuid = None

        # model_sampling / model_config stubs — accessed by processing.py ZT-SNR
        # and forge_loader latent_channels logic.  Provide safe no-ops.
        self.model_sampling = _FakeModelSampling(diffusers_unet)
        self.model_config = _FakeModelConfig()
        self.latent_channels = 4

        # conditioning_key read by processing.py
        self.conditioning_key = "crossattn"

    def memory_required(self, input_shape, **kwargs):
        """Rough VRAM estimate: batch * channels * H * W * 2 bytes * 50 (unet factor).
        This will not use if user decide to use auto offload, that is handle by diffuser"""
        if not input_shape:
            return 0
        elems = 1
        for d in input_shape:
            elems *= d
        return elems * 2 * 50   # float16 bytes × empirical SDXL factor

    def state_dict(self):
        """Delegate to the underlying diffusers UNet so module_size() works."""
        return self.diffusion_model.state_dict()

    def extra_conds(self, **kwargs):
        """Build model_conds entries for the SDXL diffusers UNet.

        Called by encode_model_conds with params that include:
          cross_attn   — text encoder hidden states (from convert_cond)
          pooled_output — CLIP-G pooled embedding (from _encode_prompts)
          width / height / crop_w / crop_h / target_width / target_height
        """
        import ldm_patched.modules.conds as _conds
        out = {}

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = _conds.CONDCrossAttn(cross_attn)

        pooled_output = kwargs.get("pooled_output", None)
        if pooled_output is not None:
            out["adm_text_embeds"] = _conds.CONDRegular(pooled_output)

            # Build the 6-element time_ids vector expected by SDXL.
            height        = kwargs.get("height", 1024)
            width         = kwargs.get("width", 1024)
            crop_h        = kwargs.get("crop_h", 0)
            crop_w        = kwargs.get("crop_w", 0)
            target_height = kwargs.get("target_height", height)
            target_width  = kwargs.get("target_width", width)
            time_ids = torch.tensor(
                [[height, width, crop_h, crop_w, target_height, target_width]],
                dtype=pooled_output.dtype,
                device=pooled_output.device,
            ).repeat(pooled_output.shape[0], 1)
            out["adm_time_ids"] = _conds.CONDRegular(time_ids)

        return out

    # ---- nn.Module compatibility (required by ModelPatcher) -----------------

    def named_modules(self, *args, **kwargs):
        """Delegate to the underlying diffusers UNet so ModelPatcher can iterate modules."""
        return self.diffusion_model.named_modules(*args, **kwargs)

    def modules(self, *args, **kwargs):
        """Delegate modules() iteration to the diffusers UNet (used by unpatch logic)."""
        return self.diffusion_model.modules(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.diffusion_model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.diffusion_model.parameters(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.diffusion_model.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        self.diffusion_model.train(mode)
        return self

    def eval(self):
        self.diffusion_model.eval()
        return self

    def get_dtype(self):
        """Return the dtype of the diffusers UNet (called by ModelPatcher.model_dtype)."""
        try:
            return next(self.diffusion_model.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ---- Generation interface (required by samplers.py) ---------------------

    def apply_model(self, x, t, c_crossattn=None, c_concat=None, **kwargs):
        """Forward pass called by samplers.py _calc_cond_batch.

        samplers.py unpacks the cond dict and passes it as keyword args.
        For SDXL the model_conds contain:
          c_crossattn — encoder hidden states (text embeddings)
          y           — pooled text embedding (CLIP-G) from cond_from_a1111_to_patched_ldm

        The diffusers UNet2DConditionModel (SDXL) needs these as:
          encoder_hidden_states=c_crossattn
          added_cond_kwargs={"text_embeds": y, "time_ids": time_ids}
        """
        # c_crossattn may arrive as a list of tensors — concatenate them
        if isinstance(c_crossattn, (list, tuple)):
            c_crossattn = torch.cat(c_crossattn, dim=1)

        # 'y' = pooled text embedding injected by cond_from_a1111_to_patched_ldm
        y = kwargs.pop("y", None)

        # Collect any pre-built added_cond_kwargs
        added_cond_kwargs = kwargs.pop("added_cond_kwargs", {}).copy()

        if y is not None and "text_embeds" not in added_cond_kwargs:
            added_cond_kwargs["text_embeds"] = y

        if "time_ids" not in added_cond_kwargs and "text_embeds" in added_cond_kwargs:
            # Build the standard SDXL 6-element time_ids from image dimensions.
            # x shape: (B, C, H_lat, W_lat) → pixel dims × 8
            batch = x.shape[0]
            h_px = x.shape[2] * 8
            w_px = x.shape[3] * 8
            time_ids = torch.tensor(
                [[h_px, w_px, 0, 0, h_px, w_px]],
                dtype=added_cond_kwargs["text_embeds"].dtype,
                device=added_cond_kwargs["text_embeds"].device,
            ).repeat(batch, 1)
            added_cond_kwargs["time_ids"] = time_ids

        # MPS (Apple Silicon) requires all inputs to share the same dtype.
        model_dtype = self.get_dtype()
        x = x.to(dtype=model_dtype)
        if c_crossattn is not None:
            c_crossattn = c_crossattn.to(dtype=model_dtype)
        added_cond_kwargs = {
            k: v.to(dtype=model_dtype) if isinstance(v, torch.Tensor) else v
            for k, v in added_cond_kwargs.items()
        }

        return self.diffusion_model(
            x,
            t,
            encoder_hidden_states=c_crossattn,
            added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
        ).sample

    # ---- Latent pass-through (called on inner_model by KSampler) -----------

    def process_latent_in(self, latent):
        """No-op — diffusers handles latent scaling internally."""
        return latent

    def process_latent_out(self, latent):
        """No-op — diffusers handles latent scaling internally."""
        return latent

    # ---- Model object accessor (used for sigma schedule lookup) -------------

    def get_model_object(self, name: str):
        """Return named sub-objects; supports 'model_sampling' for sigma calc."""
        if name == "model_sampling":
            return self.model_sampling
        return getattr(self, name, None)

    def __call__(self, x, t, c_crossattn=None, **kwargs):
        """Forward pass — delegates to apply_model."""
        return self.apply_model(x, t, c_crossattn=c_crossattn, **kwargs)


class _FakeModelSampling:
    """Stub for unet.model.model_sampling used by processing.py ZT-SNR rescaling."""

    def __init__(self, unet):
        # Build a simple linear sigma schedule so attribute reads don't crash.
        steps = 1000
        self.sigmas = torch.linspace(1.0, 0.0, steps + 1)

    def set_sigmas(self, sigmas):
        self.sigmas = sigmas


class _FakeModelConfig:
    """Stub for unet.model.model_config used by forge_loader latent_format."""

    class _FakeLatentFormat:
        @staticmethod
        def process_out(x):
            return x

        @staticmethod
        def process_in(x):
            return x

    latent_format = _FakeLatentFormat()


def _build_unet_patcher(diffusers_unet, device: torch.device) -> "UnetPatcher":
    """Wrap a diffusers UNet2DConditionModel in a real UnetPatcher.

    This gives extensions the full patcher API (clone(), model_options,
    wrappers, memory management) without any shimming.
    """
    from modules_forge.unet_patcher import UnetPatcher
    import ldm_patched.modules.model_management as model_management

    load_device   = device
    offload_device = torch.device("cpu")

    model_wrapper = _DiffusersUnetModel(diffusers_unet, device)
    patcher = UnetPatcher(
        model=model_wrapper,
        load_device=load_device,
        offload_device=offload_device,
        size=0,
    )
    return patcher


class _ForgeObjects:
    """forge_objects container holding a real UnetPatcher and VAE reference."""

    def __init__(self, unet_patcher, vae, clip=None, clipvision=None):
        self.unet = unet_patcher
        self.vae = vae
        self.clip = clip
        self.clipvision = clipvision

    def shallow_copy(self):
        copy = _ForgeObjects.__new__(_ForgeObjects)
        copy.unet = self.unet
        copy.vae = self.vae
        copy.clip = self.clip
        copy.clipvision = self.clipvision
        return copy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_dtype(device: torch.device) -> torch.dtype:
    """Return the best inference dtype for *device*.

    float16 is not supported on plain CPU in PyTorch (gives a UserWarning and
    silently degrades).  Use float32 on CPU, float16 on CUDA/MPS.
    """
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16   # MPS supports float16
    return torch.float32       # CPU: must use float32


def _pick_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode_prompts(pipe, prompts: list[str]):
    """Encode a list of text prompts using the pipeline's text encoders.

    For SDXL this calls `encode_prompt` which returns
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
     negative_pooled_prompt_embeds).

    Returns a dict compatible with the webui SDXL conditioning format:
      {'crossattn': tensor(B, seq, hidden), 'vector': tensor(B, pooled)}

    prompt_parser indexes this as a dict (conds[i] = {k: v[i] ...}), then
    reconstruct_*_batch stacks per-prompt dicts back into a batched DictWithShape,
    which cond_from_a1111_to_patched_ldm converts to model_conds with
    c_crossattn and y keys.  apply_model then uses y as text_embeds.
    """
    device = next(pipe.unet.parameters()).device
    result = pipe.encode_prompt(
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    prompt_embeds = result[0]          # (B, seq_len, hidden)
    # SDXL returns a 4-tuple; SD1.5 / SD2 return a 2-tuple
    pooled_embeds = result[2] if len(result) >= 4 else None

    if pooled_embeds is not None:
        return {"crossattn": prompt_embeds, "vector": pooled_embeds}
    return {"crossattn": prompt_embeds}


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class DiffusersModelAdapter:
    """ldm-compatible wrapper around a diffusers pipeline.

    Instantiate and return from a path-hijack loader instead of the raw pipeline:

        pipe = StableDiffusionXLPipeline.from_single_file(...)
        return DiffusersModelAdapter(pipe, checkpoint_info, model_type="sdxl")
    """

    def __init__(self, pipe, checkpoint_info, model_type: str = "sdxl"):
        self._pipe = pipe
        self._model_type = model_type  # "sdxl" | "sd1" | "sd2" | "sd3" | "flux"

        # ---- ldm boolean flags ------------------------------------------
        self.is_sdxl          = (model_type == "sdxl")
        self.is_sdxl_inpaint  = False   # would need UNet in_channels check
        self.is_ssd           = False
        self.is_sd1           = (model_type == "sd1")
        self.is_sd2           = (model_type == "sd2")
        self.is_sd3           = (model_type == "sd3")
        self.ztsnr            = False
        self.tiling_enabled   = False
        self._separate_te_loaded = False

        # ---- metadata ---------------------------------------------------
        self.sd_checkpoint_info   = checkpoint_info
        self.sd_model_checkpoint  = checkpoint_info.filename
        self.sd_model_hash        = getattr(checkpoint_info, 'hash', '')[:10]
        self.filename             = checkpoint_info.filename
        self.current_lora_hash    = ""
        self.latent_channels      = 16 if model_type == "sd3" else 4
        self.parameterization     = "eps"

        # ---- conditioning key -------------------------------------------
        self.cond_stage_key = "crossattn"

        # ---- sub-model wrappers -----------------------------------------
        self.model             = _FakeModel(pipe)
        self.cond_stage_model  = _FakeCondStageModel(pipe)
        self.first_stage_model = getattr(pipe, "vae", None)

        # ---- forge compatibility wrappers --------------------------------
        # Build a real UnetPatcher so extensions get the full patcher API
        # (clone(), model_options, wrappers, memory management).
        _device = _pick_device()
        _unet_patcher = _build_unet_patcher(pipe.unet, _device)
        _fo = _ForgeObjects(_unet_patcher, getattr(pipe, "vae", None))
        self.forge_objects                     = _fo
        self.forge_objects_original            = _fo.shallow_copy()
        self.forge_objects_after_applying_lora = _fo.shallow_copy()

        # Point sd_model.model.diffusion_model at the patcher's model wrapper
        # so forge_loader / processing.py reads are consistent.
        self.model.diffusion_model = _unet_patcher.model

        # ---- noise schedule (minimal) ------------------------------------
        # Provide a plausible alphas_cumprod so samplers that read it don't crash.
        # The actual denoising uses the diffusers scheduler, so this is only
        # consulted by extensions that inspect the noise schedule.
        try:
            sc = getattr(pipe, "scheduler", None)
            if sc is not None and hasattr(sc, "alphas_cumprod"):
                self.alphas_cumprod = sc.alphas_cumprod
            else:
                self.alphas_cumprod = torch.linspace(1.0, 0.0, 1000)
        except Exception:
            self.alphas_cumprod = torch.linspace(1.0, 0.0, 1000)
        self.alphas_cumprod_original = self.alphas_cumprod
        self.sigmas_original = None

        # ---- misc -------------------------------------------------------
        self.cond_stage_model_empty_prompt = None
        self.diff_pipeline = pipe          # back-reference expected by some code

    # ---- dtype property -------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self._pipe.unet.parameters()).dtype
        except Exception:
            return torch.float32

    # ---- ldm-compatible methods -----------------------------------------

    def encode_first_stage(self, x: torch.Tensor) -> Any:
        """Encode an image batch to latents using the diffusers VAE."""
        vae = self._pipe.vae
        x = x.to(device=next(vae.parameters()).device, dtype=vae.dtype)
        dist = vae.encode(x)
        # diffusers VAE encode() returns an AutoencoderKLOutput with .latent_dist
        return getattr(dist, "latent_dist", dist)

    def get_first_stage_encoding(self, encoder_output) -> torch.Tensor:
        """Sample from the VAE posterior distribution."""
        if hasattr(encoder_output, "sample"):
            latents = encoder_output.sample()
        elif hasattr(encoder_output, "latent_dist"):
            latents = encoder_output.latent_dist.sample()
        else:
            latents = encoder_output
        # Standard scaling factor used by SD/SDXL
        return latents * getattr(self._pipe.vae.config, "scaling_factor", 0.18215)

    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using the diffusers VAE."""
        vae = self._pipe.vae
        scaling = getattr(vae.config, "scaling_factor", 0.18215)
        z = z / scaling
        z = z.to(device=next(vae.parameters()).device, dtype=vae.dtype)
        decoded = vae.decode(z)
        return getattr(decoded, "sample", decoded)

    def get_learned_conditioning(self, prompts: list[str]) -> Any:
        """Text-encode prompts using the pipeline's text encoders."""
        return _encode_prompts(self._pipe, prompts)

    def apply_model(self, x, t, cond, **kwargs):
        """Run the diffusion UNet forward pass.

        This is a minimal shim — actual generation should go through
        the diffusers pipeline scheduler loop.
        """
        # cond may be a dict or tensor; extract cross-attention context
        c_crossattn = cond
        if isinstance(cond, dict):
            c_crossattn = cond.get("crossattn", cond.get("c_crossattn", None))
            if isinstance(c_crossattn, list):
                c_crossattn = c_crossattn[0]
        unet = self._pipe.unet
        # MPS requires all inputs to share the same dtype as the UNet weights.
        model_dtype = self.dtype
        x = x.to(dtype=model_dtype)
        if c_crossattn is not None:
            c_crossattn = c_crossattn.to(dtype=model_dtype)
        added_cond = {}
        return unet(x, t, encoder_hidden_states=c_crossattn, added_cond_kwargs=added_cond).sample

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        """No-op context manager (EMA weights not used in inference)."""
        yield

    def fix_dimensions(self, width: int, height: int):
        """Snap width/height to the nearest multiples of 8 (VAE requirement)."""
        width  = max(8, (width  // 8) * 8)
        height = max(8, (height // 8) * 8)
        return width, height

    def state_dict(self) -> dict:
        """Return an empty dict — state dict inspection not needed for inference."""
        return {}

    def to(self, device):
        self._pipe.to(device)
        return self

    def set_injections(self, name, lst):
        """No-op — injection management is handled by forge for ldm models."""
        pass

    # ---- __getattr__ fallback for anything not explicitly handled -------

    def __getattr__(self, name: str):
        # First try the underlying pipeline (handles .unet, .vae, .tokenizer, etc.)
        try:
            return getattr(self._pipe, name)
        except AttributeError:
            pass
        raise AttributeError(
            f"DiffusersModelAdapter has no attribute {name!r}. "
            f"If this is needed for generation, add it to diff_pipeline/adapter.py."
        )

    def __repr__(self):
        return (
            f"DiffusersModelAdapter("
            f"model_type={self._model_type!r}, "
            f"pipeline={type(self._pipe).__name__})"
        )
