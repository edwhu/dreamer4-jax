# Overfit a single video with pretrained encoder/decoder loaded from tokenizer Orbax checkpoint.
# Assumes tokenizer checkpoint saved both encoder and decoder params in "state/params/enc"/"dec".

from functools import partial
from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from models import Encoder, Decoder, Dynamics
from data import make_iterator
from utils import (
    temporal_patchify, pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
    make_state, make_manager, with_params, pack_mae_params, temporal_unpatchify
)
from einops import rearrange
import numpy as np
import imageio.v2 as imageio


# ---------- helpers ----------
def psnr_from_mse(mse):
    eps = 1e-12
    return 10.0 * jnp.log10(1.0 / jnp.maximum(mse, eps))

def _to_uint8(img_f32):
    # img in [0,1], shape (..., H, W, C)
    return np.asarray(np.clip(img_f32 * 255.0, 0, 255), dtype=np.uint8)

def _stack_wide(*imgs_hwC):
    # horizontally concat images of equal H/W/C
    return np.concatenate(imgs_hwC, axis=1)


# ---------- load pretrained encoder/decoder ----------
def load_pretrained_tokenizer(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder: Encoder,
    decoder: Decoder,
    enc_vars,
    dec_vars,
    sample_patches_btnd,
):
    """
    Restores encoder/decoder params from Orbax tokenizer checkpoint (same layout as in train_dynamics.py)
    Returns (enc_vars_restored, dec_vars_restored, meta)
    """
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored_meta = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    enc_kwargs = meta["enc_kwargs"]
    dec_kwargs = meta["dec_kwargs"]

    # build dummy trees
    rng_e1, rng_d1 = jax.random.split(rng)
    B, T = sample_patches_btnd.shape[:2]
    n_lat, d_b = enc_kwargs["n_latents"], enc_kwargs["d_bottleneck"]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)
    dec_vars = decoder.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)
    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)
    state_example = make_state(packed_example, opt_state_example, rng_e1, step=0)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    packed_params = restored.state["params"]
    enc_params = packed_params["enc"]
    dec_params = packed_params["dec"]
    new_enc_vars = with_params(enc_vars, enc_params)
    new_dec_vars = with_params(dec_vars, dec_params)
    print(f"[tokenizer] Restored encoder/decoder from {tokenizer_ckpt_dir} (step {latest})")
    return new_enc_vars, new_dec_vars, meta


# ---------- training step (patched to always pass mae RNG) ----------
@partial(
    jax.jit,
    static_argnames=("encoder","decoder","dynamics","tx","patch","n_s","k_max","packing_factor","use_pixel_loss"),
)
def train_step_overfit(
    encoder, decoder, dynamics, tx,
    params, opt_state,
    enc_vars, dec_vars, dyn_vars,
    frames, actions,
    *,
    patch, n_s, k_max, packing_factor,
    use_pixel_loss, lambda_pix,
    master_key,
):
    # Encoder forward (frozen)
    patches_btnd = temporal_patchify(frames, patch)

    # 🔧 split RNG for MAE (encoder) vs noise (corruption)
    mae_key, noise_key = jax.random.split(master_key)

    z_btLd, _ = encoder.apply(
        enc_vars, patches_btnd,
        rngs={"mae": mae_key},              # 🔧 REQUIRED by MAEReplacer
        deterministic=True
    )
    z1 = pack_bottleneck_to_spatial(z_btLd, n_s=n_s, k=packing_factor)

    # Fixed indices for last step
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx_full  = jnp.full(z1.shape[:2], emax, dtype=jnp.int32)
    sigma_full     = jnp.full(z1.shape[:2], 1.0 - (1.0 / k_max), jnp.float32)
    sigma_idx_full = (sigma_full * k_max).astype(jnp.int32)

    # Corrupt
    z0 = jax.random.normal(noise_key, z1.shape, dtype=z1.dtype)  # 🔧 use split noise key
    z_tilde = (1.0 - sigma_full)[...,None,None] * z0 + sigma_full[...,None,None] * z1

    def loss_and_metrics(p):
        local_dyn = {"params": p, **{k:v for k,v in dyn_vars.items() if k != "params"}}
        z1_hat = dynamics.apply(local_dyn, actions, step_idx_full, sigma_idx_full, z_tilde, deterministic=False)
        flow_per = jnp.mean((z1_hat - z1) ** 2, axis=(2,3))
        flow_mse = jnp.mean(flow_per)
        loss = flow_mse
        aux = {"flow_mse": flow_mse}

        if use_pixel_loss:
            z1_hat_btLd = unpack_spatial_to_bottleneck(z1_hat, n_s=n_s, k=packing_factor)
            dec_pred = decoder.apply(dec_vars, z1_hat_btLd, deterministic=True)
            dec_gt = patches_btnd
            pix_mse = jnp.mean((dec_pred - dec_gt) ** 2)
            loss += lambda_pix * pix_mse
            aux |= {"pix_mse": pix_mse}
        return loss, aux

    (loss_val, aux), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, aux


# ---------- main ----------
def main():
    TOKENIZER_CKPT_DIR = "/home/edward/projects/tiny_dreamer_4/logs/test/checkpoints"
    B, T, H, W, C = 1, 64, 32, 32, 3
    patch = 4
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C
    k_max = 4
    packing_factor = 2
    enc_n_latents, enc_d_bottleneck = 16, 32
    n_s = enc_n_latents // packing_factor

    USE_PIXEL_LOSS = True
    LAMBDA_PIX = 0.05

    enc_kwargs = dict(
        d_model=64, n_latents=enc_n_latents, n_patches=num_patches,
        n_heads=4, depth=8, dropout=0.0,
        d_bottleneck=enc_d_bottleneck, mae_p_min=0.0, mae_p_max=0.0,
        time_every=4, latents_only_time=True,
    )
    dec_kwargs = dict(
        d_model=64, n_heads=4, depth=8, n_latents=enc_n_latents,
        n_patches=num_patches, d_patch=D_patch, dropout=0.0,
        mlp_ratio=4.0, time_every=4, latents_only_time=True,
    )
    dyn_kwargs = dict(
        d_model=128, d_bottleneck=enc_d_bottleneck,
        d_spatial=enc_d_bottleneck * packing_factor,
        n_s=n_s, n_r=10, n_heads=4, depth=8, dropout=0.0,
        k_max=k_max, time_every=4, latents_only_time=False,
    )

    # --- fixed single video ---
    next_batch = make_iterator(B, T, H, W, C, pixels_per_step=2, size_min=10, size_max=10, hold_min=6, hold_max=6)
    _, (frames, actions) = next_batch(jax.random.PRNGKey(0))

    # --- init models ---
    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    patches_btnd = temporal_patchify(frames, patch)
    rng = jax.random.PRNGKey(0)
    enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
    fake_z = jnp.zeros((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)
    # load pretrained encoder/decoder
    enc_vars, dec_vars, meta = load_pretrained_tokenizer(
        TOKENIZER_CKPT_DIR, rng=rng,
        encoder=encoder, decoder=decoder,
        enc_vars=enc_vars, dec_vars=dec_vars,
        sample_patches_btnd=patches_btnd,
    )

    # 🔧 fixed MAE RNG for all encoder eval forwards in main()
    mae_eval_key = jax.random.PRNGKey(777)

    # --- build dynamics ---
    z_btLd, _ = encoder.apply(
        enc_vars, patches_btnd,
        rngs={"mae": mae_eval_key},     # 🔧 pass mae key
        deterministic=True
    )
    z1 = pack_bottleneck_to_spatial(z_btLd, n_s=n_s, k=packing_factor)
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jnp.full((B, T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((B, T), k_max - 1, dtype=jnp.int32)
    dynamics = Dynamics(**dyn_kwargs)
    dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions, step_idx, sigma_idx, z1)

    params = dyn_vars["params"]
    tx = optax.adam(3e-4)
    opt_state = tx.init(params)

    # --- log baselines ---
    # Re-encode with same mae_eval_key for fair comparison
    z_btLd, _ = encoder.apply(
        enc_vars, patches_btnd,
        rngs={"mae": mae_eval_key},     # 🔧
        deterministic=True
    )
    dec_floor = decoder.apply(dec_vars, z_btLd, deterministic=True)
    floor_mse = jnp.mean((dec_floor - patches_btnd) ** 2)
    print(f"[floor] decoder reconstruction mse={float(floor_mse):.6g} (PSNR={float(psnr_from_mse(floor_mse)):.2f} dB)")

    # --- training loop ---
    max_steps = 1000
    log_every = 200
    start_t = time()
    for step in range(max_steps):
        rng, key = jax.random.split(rng)
        params, opt_state, aux = train_step_overfit(
            encoder, decoder, dynamics, tx,
            params, opt_state, enc_vars, dec_vars, dyn_vars,
            frames, actions,
            patch=patch, n_s=n_s, k_max=k_max, packing_factor=packing_factor,
            use_pixel_loss=USE_PIXEL_LOSS, lambda_pix=LAMBDA_PIX,
            master_key=key,
        )
        if step % log_every == 0:
            flow = float(aux["flow_mse"])
            msg = f"step {step:06d} | flow_mse={flow:.6g}"
            if USE_PIXEL_LOSS and "pix_mse" in aux:
                pix = float(aux["pix_mse"])
                psnr = float(psnr_from_mse(aux["pix_mse"]))
                msg += f" | pix_mse={pix:.6g} | psnr={psnr:.2f} dB"
            msg += f" | floor_mse={float(floor_mse):.6g} | time={time()-start_t:.1f}s"
            print(msg)

    # --- final decode ---
    dyn_vars_final = with_params(dyn_vars, params)
    # Re-encode once more to keep things consistent (same mae_eval_key)
    z_btLd, _ = encoder.apply(
        enc_vars, patches_btnd,
        rngs={"mae": mae_eval_key},     # 🔧
        deterministic=True
    )
    z1 = pack_bottleneck_to_spatial(z_btLd, n_s=n_s, k=packing_factor)
    z1_hat = dynamics.apply(dyn_vars_final, actions, step_idx, sigma_idx, z1, deterministic=True)
    z1_hat_btLd = unpack_spatial_to_bottleneck(z1_hat, n_s=n_s, k=packing_factor)
    dec_pred = decoder.apply(dec_vars, z1_hat_btLd, deterministic=True)
    final_pix_mse = jnp.mean((dec_pred - patches_btnd) ** 2)
    print(f"[Final] pred pix_mse={float(final_pix_mse):.6g} (PSNR={float(psnr_from_mse(final_pix_mse)):.2f} dB)")

        # === Visualization: GT | Decoder floor | Pred ===
    # Decode GT latents (floor) and predicted latents to pixels
    floor_frames = temporal_unpatchify(dec_floor, H, W, C, patch)         # (B,T,H,W,C)
    pred_frames  = temporal_unpatchify(dec_pred,  H, W, C, patch)         # (B,T,H,W,C)
    gt_frames    = frames                                                 # (B,T,H,W,C)

    # Convert to uint8 on CPU
    floor_np = _to_uint8(floor_frames[0])   # (T,H,W,C)
    pred_np  = _to_uint8(pred_frames[0])    # (T,H,W,C)
    gt_np    = _to_uint8(gt_frames[0])      # (T,H,W,C)

    # Make comparison per frame: [GT | FLOOR | PRED]
    comparison = [_stack_wide(gt_np[t], floor_np[t], pred_np[t]) for t in range(gt_np.shape[0])]

    # Write GIF (≈25fps)
    gif_path = Path("./overfit_predictions.gif")
    imageio.mimsave(gif_path, comparison, duration=1/25)
    print(f"[viz] wrote {gif_path.resolve()}")

    # Also write MP4 (smoother playback)
    mp4_path = Path("./overfit_predictions.mp4")
    try:
        with imageio.get_writer(mp4_path, fps=25, codec="libx264", quality=8) as w:
            for fr in comparison:
                w.append_data(fr)
        print(f"[viz] wrote {mp4_path.resolve()}")
    except Exception as e:
        print(f"[viz] MP4 write skipped ({e}); GIF saved.")


if __name__ == "__main__":
    main()
