# Dreamer 4 in pure Jax

This is an unofficial implementation of the [Dreamer 4](https://danijar.com/project/dreamer4/) world model in the paper "Training Agents Inside of Scalable World Models".

- [Website](https://danijar.com/project/dreamer4/)
- [Twitter](https://x.com/danijarh/status/1973072288351396320)

## Roadmap

- [x] Toy Video Dataset generation
- [ ] Causal Tokenizer
    - [x] Space-time Axial Attention
    - [x] MAE encoder decoder 
    - [x] MSE loss
    - [x] Training loop
    - [x] LPIPS loss
    - [x] checkpointing
    - [ ] Minor improvements
        - [ ] wandb logging
        - [ ] cli args / config 
        - [ ] RoPE
        - [ ] SwiGLU
        - [ ] GQA
- [ ] Interactive Dynamics Model
    - [x] Add actions to Toy Video Dataset
    - [ ] Add multi-modality support for efficient transformer block
    - [ ] setup architecture
    - [ ] shortcut forcing 
- [ ] Imagination training 
    - [ ] Behavior cloning and reward model
    - [ ] RL Training
- [ ] Small offline RL dataset generation (Atari-5 or Craftax)
- [ ] Interactive decision making

## Installation
Use `uv` to create the virtual environment and install dependencies:
```bash
uv sync      # creates .venv and installs packages from uv.lock
source .venv/bin/activate
```
By default, this installs the CPU version of jax. Follow the instructions in the Jax repo to install the GPU version (e.g. `uv pip install "jax[cuda12]"`).

## Dataset generation

For now, we are generating bouncing square dataset in `data.py`. It outputs a (B,T,H,W,C) batch of data. We eventually need to add in actions and rewards for the dynamics model and RL training. 

<img src="docs/video.gif" width="500px">
 
## Causal Tokenizer
The Causal Tokenizer learns the representation for Dreamer 4.

<img src="docs/step_75900.png" width="300px">

From top to bottom we have: 2 images, their masked variants that are fed to the encoder, then 4 decoder outputs (2 partial decodings, and 2 full decodings). We can see the decoder outputs match the input images well.  

It is a causal transformer encoder, trained through a masked autoencoder loss. Dreamer 4 uses an efficient transformer design which uses axial attention to attend to the space and time dimensions independently. 


<details>
<summary><b>Architecture Details</b></summary>
<br>

Data variables:

| Variable | Description |
|----------|-------------|
| B        | Batch size  |
| T        | Number of timesteps in the video sequence |
| H        | Height of the video frames |
| W        | Width of the video frames |
| C        | Number of channels in the video frames (e.g., 3 for RGB) |
| P        | Patch size for patchifying the video frames |
| N_p      | Number of patches per frame, calculated as (H/P)*(W/P) |
| D        | Dimensionality of each patch, calculated as P*P*C |

Encoder model settings:
| Variable | Description |
|----------|-------------|
| N_l      | Number of latent tokens |
| D_bottleneck | Dimensionality of the bottleneck space |



Data to Encoder steps:
1. Take in video data of shape (B, T, H, W, C). 
2. Patchify with patches of size P, to $(B, T, H/P, W/P, C)$, and then flatten the patches to a sequence $(B, T, N_p, D)$ where $N_p = (H/P)*(W/P)$ and $D = P*P*C$.
3. Prepend learnable latent tokens before each patch token in each timestep to get a batch of sequences, shape $(B, T,  N_l + N_p, D)$, where $N_l$ is the number of latent tokens.
4. Spatial attention: First, let's zoom in on just one timestep with its $N_l$ latent tokens and $N_p$ patch tokens. We do full attention over these tokens. Now, zooming out, this is done on each timestep independently. 
5. Then, we perform causal attention over just the $N_l$ latent tokens to aggregate info across time. In practice, we do causal attention only once every 4 spatial attention layers, to save computation. 
6. Then, we project the last attention block's latent tokens into a lower dimensional bottleneck space, using a linear projection and a tanh. This is the $z$ space in the paper.

Then, we train the $z$ representation using a decoder. The decoder is also a transformer, using the same axial attention idea. 
1. Take in the sequence of $z$ tokens, of shape $(B, T, N_l, D_\text{bottleneck})$. Prepend learnable patch query tokens to each timestep, to get shape $(B, T, N_l + N_p, D_\text{bottleneck})$.
2. Do spatial attention over the $N_l + N_p$ tokens at each timestep independently, then do causal attention over the $N_l$ latent tokens every 4 layers.
3. Finally, project the patch query tokens back to pixel space using a linear layer, and unpatchify to get back to $(B, T, H, W, C)$.
</details>




### Training the Causal Tokenizer
The causal tokenizer is trained using a masked autoencoder loss. We randomly mask out a portion of the input patches with probability $U(0, 0.9)$, and replace the missing portion with a learnable embedding. We then pass the masked input through the encoder and decoder, and train the model with reconstruction loss (MSE) between the original and reconstructed images.

## Interactive Dynamics 
- need to make sure we are handling multiple modalities correctly, where latent tokens can read from everything but other tokens can only attend amongst tokens with the same modality 