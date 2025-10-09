import jax.numpy as jnp
import flax.linen as nn
import jax
import time
from flax.core import FrozenDict

"""
- prelayer RMS norm

Skipped:
- RoPE, using sinusoidal positions instead
- SwiGLU, using GELU instead
- GQA
"""
    
def sinusoid_table(n: int, d: int, base: float = 10000.0, dtype=jnp.float32) -> jnp.ndarray:
    """
    Standard Transformer sinusoid: even dims use sin, odd dims use cos with frequencies
    base^{-2k/d}. Works for odd d too.
    """
    pos = jnp.arange(n, dtype=dtype)[:, None]            # (n,1)
    i = jnp.arange(d, dtype=dtype)[None, :]              # (1,d)
    # k = floor(i/2)
    k = jnp.floor(i / 2.0)
    div = jnp.power(base, -(2.0 * k) / jnp.maximum(1.0, jnp.array(d, dtype)))
    angles = pos * div                                    # (n,d)
    table = jnp.where((i % 2) == 0, jnp.sin(angles), jnp.cos(angles))
    return table.astype(dtype)


def add_sinusoidal_positions(tokens_btSd: jnp.ndarray) -> jnp.ndarray:
    """tokens: (B,T,S,D) -> adds time and step sinusoids and returns same shape."""
    B, T, S, D = tokens_btSd.shape
    pos_t = sinusoid_table(T, D)     # (T,D)
    pos_s = sinusoid_table(S, D)     # (S,D)
    # Optionally scale to keep variance stable (common trick)
    scale = 1.0 / jnp.sqrt(jnp.array(D, dtype=tokens_btSd.dtype))
    return tokens_btSd + scale * (pos_t[None, :, None, :] + pos_s[None, None, :, :])

class MAEReplacer(nn.Module):
    p_min: float = 0.0
    p_max: float = 0.9

    @nn.compact
    def __call__(self, patches_btnd: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # patches_btnd: (B,T,Np,D)
        B, T, Np, D = patches_btnd.shape
        mask_token = self.param("mask_token", nn.initializers.normal(0.02), (D,))
        # draw RNGs from a named stream
        rng = self.make_rng("mae")
        p_rng, m_rng = jax.random.split(rng)
        p_bt = jax.random.uniform(p_rng, (B, T), minval=self.p_min, maxval=self.p_max)  # (B,T)
        keep_prob_bt1 = 1.0 - p_bt[..., None]                                           # (B,T,1)
        keep = jax.random.bernoulli(m_rng, keep_prob_bt1, (B, T, Np))                   # (B,T,Np)
        keep = keep[..., None]                                                          # (B,T,Np,1)
        replaced = jnp.where(keep, patches_btnd, mask_token.reshape(1, 1, 1, D))
        mae_mask = (~keep).astype(jnp.bool_)                                            # (B,T,Np,1)
        return replaced, mae_mask, keep_prob_bt1


# ---------- small building blocks ----------

class RMSNorm(nn.Module):
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * (scale / jnp.sqrt(var + self.eps))

class MLP(nn.Module):
    d_model: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        h = nn.Dense(int(self.d_model * self.mlp_ratio))(x)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        h = nn.Dense(self.d_model)(h)
        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        return h
# ---------- axial attention layers ----------

class SpaceSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.0
    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        # x: (B, T, S, D) -> attend across S within a timestep
        B, T, S, D = x.shape
        x_ = x.reshape(B*T, S, D)
        y_ = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(x_, x_)
        y = y_.reshape(B, T, S, D)
        return y

class TimeSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.0
    latents_only: bool = True
    n_latents: int = 0   # required if latents_only

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        # x: (B, T, S, D) -> attend across T, causal
        B, T, S, D = x.shape
        if self.latents_only:
            assert 0 < self.n_latents <= S
            lat = x[:, :, :self.n_latents, :]                # (B, T, L, D)
            lat_btld = lat.transpose(0, 2, 1, 3).reshape(B*self.n_latents, T, D)  # (B*L, T, D)
            causal = nn.attention.make_causal_mask(jnp.ones((B*self.n_latents, T), dtype=bool))
            out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout,
                deterministic=deterministic,
            )(lat_btld, lat_btld, mask=causal)
            out = out.reshape(B, self.n_latents, T, D).transpose(0, 2, 1, 3)      # back to (B, T, L, D)
            x = x.at[:, :, :self.n_latents, :].set(out)
            return x
        else:
            x_bstd = x.transpose(0, 2, 1, 3).reshape(B*S, T, D)  # (B*S, T, D)
            causal = nn.attention.make_causal_mask(jnp.ones((B*S, T), dtype=bool))
            out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout,
                deterministic=deterministic,
            )(x_bstd, x_bstd, mask=causal)
            out = out.reshape(B, S, T, D).transpose(0, 2, 1, 3)  # back to (B, T, S, D)
            return out

# ---------- a single block-causal layer ----------

class BlockCausalLayer(nn.Module):
    d_model: int
    n_heads: int
    n_latents: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    layer_index: int = 0
    time_every: int = 4
    latents_only_time: bool = True

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        # Space attention (within timestep)
        y = RMSNorm()(x)
        y = SpaceSelfAttention(self.d_model, self.n_heads, self.dropout)(y, deterministic=deterministic)
        x = x + nn.Dropout(self.dropout)(y, deterministic=deterministic)

        # Time attention (causal across timesteps), only on certain layers
        if (self.layer_index + 1) % self.time_every == 0:
            y = RMSNorm()(x)
            y = TimeSelfAttention(
                self.d_model, self.n_heads, self.dropout,
                latents_only=self.latents_only_time, n_latents=self.n_latents
            )(y, deterministic=deterministic)
            x = x + nn.Dropout(self.dropout)(y, deterministic=deterministic)

        # MLP
        y = RMSNorm()(x)
        y = MLP(self.d_model, self.mlp_ratio, self.dropout)(y, deterministic=deterministic)
        x = x + nn.Dropout(self.dropout)(y, deterministic=deterministic)
        return x
# ---------- the transformer stack ----------

class BlockCausalTransformer(nn.Module):
    d_model: int
    n_heads: int
    depth: int
    n_latents: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4
    latents_only_time: bool = True

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        # x: (B, T, S, D)  (S = n_latents + n_patches)
        for i in range(self.depth):
            x = BlockCausalLayer(
                self.d_model, self.n_heads, self.n_latents, self.dropout, self.mlp_ratio,
                layer_index=i, time_every=self.time_every,
                latents_only_time=self.latents_only_time, 
            )(x, deterministic=deterministic)
        return x

class Encoder(nn.Module):
    d_model: int
    n_latents: int
    n_heads: int
    depth: int
    d_bottleneck: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4
    latents_only_time: bool = True
    mae_p_min: float = 0.0
    mae_p_max: float = 0.9
    
    def setup(self):
        self.patch_proj = nn.Dense(self.d_model, name="patch_proj")
        self.bottleneck_proj = nn.Dense(self.d_bottleneck, name="bottleneck_proj")

    @nn.compact
    def __call__(self, patch_tokens_btnd, *, deterministic: bool = True) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # 1) Project patches to D_model
        proj_patches = self.patch_proj(patch_tokens_btnd)  # (B,T,Np,D)

        # 2) MAE mask-and-replace on patch tokens (encoder input only)
        proj_patches_masked, patch_mask, keep_prob = MAEReplacer(name="mae", p_min=self.mae_p_min, p_max=self.mae_p_max)(proj_patches)
        # print(f"proj_patches_masked.shape: {proj_patches_masked.shape}")
        # print(f"patch_mask.shape: {patch_mask.shape}")

        # 3) Prepend learned latents (owned here)
        latents = self.param("latents_enc", nn.initializers.normal(0.02), (self.n_latents, self.d_model))
        # print(f"latents.shape: {latents.shape}")
        B, T = proj_patches_masked.shape[:2]
        lat_btld = jnp.broadcast_to(latents[None, None, ...], (B, T, *latents.shape))
        # print(f"lat_btld.shape: {lat_btld.shape}")
        tokens_btSd = jnp.concatenate([lat_btld, proj_patches_masked], axis=2)  # (B,T,S=(Np+Nl),D)
        # print(f"tokens_btSd.shape: {tokens_btSd.shape}")

        # 4) Add sinusoidal positions (param-free)
        tokens_btSd = add_sinusoidal_positions(tokens_btSd)

        # 5) Feed tokens into transformer
        transformer = BlockCausalTransformer(
            self.d_model, self.n_heads, self.depth, self.n_latents, self.dropout, self.mlp_ratio,
            self.time_every, self.latents_only_time, 
        )
        encoded_tokens_btSd = transformer(tokens_btSd, deterministic=deterministic)
        # print(f"encoded_tokens_btSd.shape: {encoded_tokens_btSd.shape}")

        # 6) Project latent tokens to bottleneck and tanh
        latent_tokens_btNld = encoded_tokens_btSd[:, :, :self.n_latents, :]
        proj_tokens_btNld = nn.tanh(self.bottleneck_proj(latent_tokens_btNld))

        return proj_tokens_btNld, (patch_mask, keep_prob)  # keep mask if you want diagnostics

class Decoder(nn.Module):
    """
    MAE-style decoder that reads temporal info from latent tokens and writes
    reconstructions at per-patch query tokens.

    Inputs:
      - z: (B, T, N_l, d_bottleneck)  -- encoder bottleneck output

    Config:
      - n_patches: number of patch query tokens to use in the decoder
      - d_patch:   dimensionality of each patch to reconstruct (D_patch)
      - d_model, n_heads, depth, dropout, mlp_ratio, time_every, latents_only_time
        typically mirror the encoder.
    """
    d_model: int
    n_heads: int
    depth: int
    n_patches: int
    d_patch: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4
    latents_only_time: bool = True

    @nn.compact
    def __call__(self, z: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        B, T, N_l, d_bottleneck = z.shape

        # 1) Up-project latent bottleneck to d_model (per latent token)
        up = nn.Dense(self.d_model, name="up_proj")
        lat_btNld = nn.tanh(up(z))  # (B, T, N_l, D)

        # 2) Learned per-patch query tokens (owned by the decoder)
        patch_queries = self.param(
            "patch_queries",
            nn.initializers.normal(0.02),
            (self.n_patches, self.d_model),
        )  # (Np, D)
        pq_btNpd = jnp.broadcast_to(
            patch_queries[None, None, ...],
            (B, T, self.n_patches, self.d_model),
        )  # (B, T, Np, D)

        # 3) Concat: [latents, patch queries]  ->  (B, T, S=N_l+N_p, D)
        tokens_btSd = jnp.concatenate([lat_btNld, pq_btNpd], axis=2)

        # 4) Add sinusoidal positions
        tokens_btSd = add_sinusoidal_positions(tokens_btSd)

        # 5) Axial block-causal transformer
        #    - SpaceSelfAttention over all S tokens (latents + queries)
        #    - TimeSelfAttention only over the first N_l latent tokens
        x = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            latents_only_time=self.latents_only_time,
            n_latents=N_l,  # <- causal time over latent tokens only
        )(tokens_btSd, deterministic=deterministic)  # (B, T, S, D)

        # 6) Prediction head over the patch-query slice
        x_patches = x[:, :, N_l:, :]                         # (B, T, Np, D)
        pred_btnd = nn.Dense(self.d_patch, name="patch_head")(x_patches)  # (B,T,Np,D_patch)
        # output is between 0 and 1
        pred_btnd = nn.sigmoid(pred_btnd)

        return pred_btnd


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    B = 2
    T = 10
    num_patches = 4
    D_patch = 3
    enc_n_latents = 2
    enc_d_bottleneck = 3
    x = jnp.ones((B, T, num_patches, D_patch))  # (B,T,Np,D_patch)

    encoder = Encoder(d_model=8, n_latents=enc_n_latents, n_heads=2, depth=2, dropout=0.5, d_bottleneck=enc_d_bottleneck)
    decoder = Decoder(d_model=8, n_heads=2, depth=2, n_patches=num_patches, d_patch=D_patch, dropout=0.5)
    # init: give both "mae" and "dropout" keys (dropout only needed if deterministic=False)
    enc_vars = encoder.init(
        {"params": rng, "mae": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)},
        x,
        deterministic=True,
    )
    # Decode
    fake_z = jnp.ones((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init(
        {"params": rng, "dropout": jax.random.PRNGKey(2)},
        fake_z,
        deterministic=True,
    )

    def forward_apply(enc_vars: FrozenDict, dec_vars: FrozenDict,
                    patches_btnd: jnp.ndarray,
                    *, mae_key=None, drop_key=None, train: bool):
        # Encoder
        rngs_enc = {}
        if train:
            rngs_enc = {"mae": mae_key, "dropout": drop_key}
        else:
            rngs_enc = {"mae": mae_key}  # if you still want masking during eval

        z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd,
                                        rngs=rngs_enc,
                                        deterministic=not train)
        # Decoder
        rngs_dec = {"dropout": drop_key} if train else {}
        pred_btnd = decoder.apply(dec_vars, z_btLd,
                                rngs=rngs_dec,
                                deterministic=not train)
        return pred_btnd, mae_info
    
    jit_forward = jax.jit(forward_apply, static_argnames=["train"])
    mae_key = jax.random.PRNGKey(0)
    drop_key = jax.random.PRNGKey(1)
    # Warm-up (compilation happens here)
    t0 = time.time()
    out = jit_forward(enc_vars, dec_vars, x, mae_key=mae_key, drop_key=drop_key, train=True)
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), out)
    t1 = time.time()
    # Hot run (should be much faster)
    t2 = time.time()
    out = jit_forward(enc_vars, dec_vars, x, mae_key=mae_key, drop_key=drop_key, train=True)
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), out)
    t3 = time.time()

    print(f"Warm-up (compile+run): {t1 - t0:.3f}s")
    print(f"Hot run (cached):      {t3 - t2:.3f}s")

    # print(pred.shape) # should be (B,T,Np,D_patch)
    # masked_pred = jnp.where(patch_mask, pred, 0.0)
    # masked_target = jnp.where(patch_mask, patch_tokens_btnd, 0.0)
    # num_masked = jnp.maximum(patch_mask.sum(), 1)
    # recon_loss = jnp.sum((masked_pred - masked_target) ** 2) / num_masked
    # print(recon_loss)
    # import ipdb; ipdb.set_trace()