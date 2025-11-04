from functools import partial
import jax
import jax.numpy as jnp
import imageio.v2 as imageio
from einops import rearrange

# ============================================================
# Constants
# ============================================================

# Action space (categorical):
#   0: up, 1: down, 2: left, 3: right
# Directions are in (dy, dx) order for image coordinates.
ACTION_DELTAS_YX = jnp.array(
    [[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32
)


# ============================================================
# Frame generation for action-conditioned videos
# ============================================================

@partial(
    jax.jit,
    static_argnames=[
        "batch_size",
        "time_steps",
        "height",
        "width",
        "channels",
        "pixels_per_step",
    ],
)
def generate_batch(
    init_pos: jnp.ndarray,               # (B, 2) int32 — initial square top-left (y, x)
    init_background_color: jnp.ndarray,  # (B, C) uint8 — per-sample background color
    init_foreground_color: jnp.ndarray,  # (B, C) uint8 — per-sample square color
    square_sizes: jnp.ndarray,           # (B,) int32  — side length per sample
    actions: jnp.ndarray,                # (B, T) int32 ∈ {0,1,2,3,4}
    batch_size: int,
    time_steps: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 1,
):
    """
    Simulate a batch of action-conditioned bouncing-square videos.

    Args:
        init_pos: (B,2) top-left corner (y,x) for each sample
        init_background_color: (B,C) RGB background colors, uint8
        init_foreground_color: (B,C) RGB foreground colors, uint8
        square_sizes: (B,) per-sample square side lengths
        actions: (B,T-1) categorical directions in {0:up,1:down,2:left,3:right}
        batch_size, time_steps, height, width, channels: scene dimensions
        pixels_per_step: number of pixels moved per action step

    Returns:
        video:   (B,T,H,W,C) float32 in [0,1]
        actions: (B,T-1) int32
    """
    H, W = height, width
    k_b = jnp.clip(square_sizes, 1, jnp.minimum(H, W))  # ensure squares fit
    deltas = ACTION_DELTAS_YX * pixels_per_step         # motion step table

    # --- Integrate positions over time for one sample ---
    def integrate_one(p0, k, acts):
        """Integrate position sequence given actions for one sample."""
        max_y = H - k
        max_x = W - k

        def step(pos, a):
            dy, dx = deltas[a]
            y, x = pos

            # update position
            y_next = y + dy
            x_next = x + dx

            # reflect off vertical boundaries
            y_next = jnp.where(y_next < 0, -y_next, y_next)
            y_next = jnp.where(y_next > max_y, 2 * max_y - y_next, y_next)

            # reflect off horizontal boundaries
            x_next = jnp.where(x_next < 0, -x_next, x_next)
            x_next = jnp.where(x_next > max_x, 2 * max_x - x_next, x_next)

            nxt = jnp.stack([y_next, x_next])
            return nxt, nxt

        # scan over (T-1) actions, accumulate all positions
        _, pos_seq = jax.lax.scan(step, p0, acts)
        positions = jnp.concatenate([p0[None, :], pos_seq], 0)  # include initial
        return positions

    # Vectorize over batch dimension
    positions = jax.vmap(integrate_one)(init_pos, k_b, actions)  # (B,T,2)

    # --- Paint frames based on positions ---
    ys = jnp.arange(H)
    xs = jnp.arange(W)

    # initialize backgrounds
    video = (
        jnp.ones((batch_size, time_steps, H, W, channels), dtype=jnp.uint8)
        * init_background_color[:, None, None, None, :]
    )

    def paint_one(frame, y, x, color, k):
        """Paint one square of size (k,k) at (y,x) on a single frame."""
        ymask = (ys >= y) & (ys < y + k)
        xmask = (xs >= x) & (xs < x + k)
        mask = (ymask[:, None] & xmask[None, :])[..., None]
        return jnp.where(mask, color[None, None, :], frame)

    # Vectorize over time and batch
    paint_over_time = jax.vmap(paint_one, in_axes=(0, 0, 0, None, None))
    paint_over_batch = jax.vmap(paint_over_time, in_axes=(0, 0, 0, 0, 0))

    y_idx = positions[..., 0]
    x_idx = positions[..., 1]
    video = paint_over_batch(video, y_idx, x_idx, init_foreground_color, k_b)
    return (video.astype(jnp.float32) / 255.0, actions)


# ============================================================
# JIT-friendly iterator with stochastic policy
# ============================================================

def make_iterator(
    batch_size: int,
    time_steps: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 1,
    size_min: int = 5,
    size_max: int = 12,
    hold_min: int = 3,
    hold_max: int = 8,
    fg_min_color: int = 0,
    fg_max_color: int = 255,
    bg_min_color: int = 0,
    bg_max_color: int = 255,
):
    """
    Construct a JIT-friendly data iterator for stochastic action-conditioned videos.

    The iterator implements a *commit-then-switch* policy:
      - The agent commits to its current direction for a random number of steps
        in [hold_min, hold_max].
      - When that interval expires, it switches to a different direction,
        chosen uniformly from the remaining 3.

    Returns:
        next_fn(key) -> (new_key, (video, actions))

    Args:
        batch_size: number of videos per batch
        time_steps: number of frames per video
        height, width, channels: spatial dimensions
        pixels_per_step: step size for each move
        size_min, size_max: range of square sizes
        hold_min, hold_max: range of commit durations
        fg_min_color, fg_max_color: range of foreground colors
        bg_min_color, bg_max_color: range of background colors
    """
    # Wrap renderer with static shape args
    gen = jax.jit(
        lambda pos, bg, fg, sizes, acts: generate_batch(
            pos, bg, fg, sizes, acts,
            batch_size=batch_size,
            time_steps=time_steps,
            height=height,
            width=width,
            channels=channels,
            pixels_per_step=pixels_per_step,
        )
    )

    Tm1 = time_steps - 1

    # --- Define per-sample stochastic policy ---
    def _sample_actions_for_one(key):
        """
        Generate (T-1,) integer action sequence using the commit-then-switch policy.

        Args:
            key: PRNGKey for one sample

        Returns:
            actions_t: (T-1,) int32 in {0,1,2,3}
        """
        # Split RNGs for reproducible scan keys
        step_keys = jax.random.split(key, Tm1 * 2).reshape(Tm1, 2, 2)

        # Initialize with remaining=0 to force a new direction at t=0
        init_action = jnp.int32(0)
        init_remaining = jnp.int32(0)

        def one_step(carry, keys_pair):
            cur_action, remaining = carry
            k_a, k_h = keys_pair

            # sample a new direction (different from current)
            r = jax.random.randint(k_a, (), 0, 3, dtype=jnp.int32)
            new_action = r + (r >= cur_action)  # skip over current

            # sample new hold length in [hold_min, hold_max]
            new_hold = jax.random.randint(k_h, (), hold_min, hold_max + 1, dtype=jnp.int32)

            need_new = (remaining <= 0)
            cur_action = jnp.where(need_new, new_action, cur_action)
            remaining = jnp.where(need_new, new_hold, remaining)

            # emit current action, decrement counter
            action_out = cur_action
            next_remaining = remaining - 1
            return (cur_action, next_remaining), action_out

        (_, _), actions_t = jax.lax.scan(one_step, (init_action, init_remaining), step_keys)
        return actions_t

    # Vectorize across batch
    sample_actions_batch = jax.vmap(_sample_actions_for_one)

    # --- Define main iterator step ---
    @jax.jit
    def next(key):
        """
        One iterator step.

        Samples random initial positions, colors, sizes, and actions,
        then calls `generate_action_batch` to produce a full video batch.

        Args:
            key: PRNGKey

        Returns:
            new_key: updated PRNGKey
            (video, actions):
                video   (B,T,H,W,C) float32 in [0,1]
                actions (B,T) int32 in {0,1,2,3}
        """
        key, sub = jax.random.split(key)
        k_pos, k_bg, k_fg, k_size, k_act = jax.random.split(sub, 5)

        # sample initial states
        init_pos = jax.random.randint(
            k_pos, (batch_size, 2),
            minval=jnp.array([0, 0]),
            maxval=jnp.array([height, width]),
            dtype=jnp.int32,
        )
        init_bg = jax.random.randint(
            k_bg, (batch_size, channels),
            bg_min_color, bg_max_color + 1, dtype=jnp.uint8,
        )
        init_fg = jax.random.randint(
            k_fg, (batch_size, channels),
            fg_min_color, fg_max_color + 1, dtype=jnp.uint8,
        )
        sizes = jax.random.randint(
            k_size, (batch_size,),
            size_min, size_max + 1,
            dtype=jnp.int32,
        )

        # sample actions using stochastic policy
        act_keys = jax.random.split(k_act, batch_size)
        actions = sample_actions_batch(act_keys)  # (B,T-1)

        # generate video
        video, _ = gen(init_pos, init_bg, init_fg, sizes, actions)
        # prepend an empty action (value = 4) at t=0
        actions = jnp.concatenate([jnp.full((batch_size,1), 4, dtype=actions.dtype), actions], axis=1)
        return key, (video, actions)

    return next

def patchify(x: jnp.ndarray, patch: int) -> jnp.ndarray:
    """
    x: (B, H, W, C)  ->  patches: (B, N, D)
      where N = (H/patch)*(W/patch), D = patch*patch*C
    """
    patches = rearrange(x, "b (hp p1) (wp p2) c -> b (hp wp) (p1 p2 c)", p1=patch, p2=patch)
    return patches

def unpatchify(patches: jnp.ndarray, H: int, W: int, C: int, patch: int) -> jnp.ndarray:
    """
    patches: (B, N, D)  ->  x: (B, H, W, C)
      where N = (H/patch)*(W/patch), D = patch*patch*C
    """
    image = rearrange(patches, "b (hp wp) (p1 p2 c) -> b (hp p1) (wp p2) c", hp=H//patch, wp=W//patch, p1=patch, p2=patch, c=C)
    return image



# ============================================================
# Demo / visualization
# ============================================================

def test_iterator():
    """
    Example usage of make_action_iterator.

    Generates a batch of videos where each sample moves stochastically,
    holding a direction for a random duration before switching.
    """
    key = jax.random.PRNGKey(0)
    B, T = 6, 48
    H, W, C = 64, 64, 3

    next_step = make_iterator(
        batch_size=B,
        time_steps=T,
        height=H,
        width=W,
        channels=C,
        pixels_per_step=2,
        size_min=6,
        size_max=14,
        hold_min=4,
        hold_max=9,
    )

    key, (video, actions) = next_step(key)
    print("video", video.shape, video.dtype, "actions", actions.shape, actions.dtype)

    # Concatenate all samples horizontally per frame for visualization
    def render_frame(_, frame):
        grid = jnp.concatenate(frame, axis=1)
        return (), grid

    _, frames = jax.lax.scan(render_frame, (), video.transpose(1, 0, 2, 3, 4))
    imageio.mimsave(
        "stochastic_policy_iter.gif",
        jnp.asarray(frames * 255.0, dtype=jnp.uint8),
        fps=8, loop=1000,
    )
    print("Saved stochastic_policy_iter.gif")


if __name__ == "__main__":
    test_iterator()
