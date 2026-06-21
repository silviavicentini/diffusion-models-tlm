# Diffusion Models

Hands-on lab on diffusion models for the [Trento Local Minimum](https://github.com/tlm-journalclub-org) journal club.

We build a DDPM from scratch: first on a 1D toy you can plot end to end, then on 16x16 pixel sprites, and finally text-to-image with Stable Diffusion. The notebooks pair with the slide deck *Diffusion Models, Hands-On Approach* (Silvia Vicentini and Daniele Contessi).

**Runtime:** Google Colab with a T4 GPU for notebooks 01 and 02. Notebook 00 runs on CPU in under a minute.

## Notebooks

| File | What it covers | Colab |
|------|----------------|-------|
| `00_diffusion_from_scratch_1d.ipynb` | The full pipeline in 1D: train a tiny network to denoise a bimodal distribution and watch it rebuild the two peaks. Made to follow the theory slides. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/silviavicentini/diffusion-models-tlm/blob/master/00_diffusion_from_scratch_1d.ipynb) |
| `01_diffusion_from_scratch_sprites.ipynb` | DDPM from scratch on 16x16 sprites (or Pokémon): forward process, training, DDPM and DDIM sampling. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/silviavicentini/diffusion-models-tlm/blob/master/01_diffusion_from_scratch_sprites.ipynb) |
| `02_text_to_image.ipynb` | Text-to-image with Stable Diffusion: architecture, image generation, and the main parameters to play with. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/silviavicentini/diffusion-models-tlm/blob/master/02_text_to_image.ipynb) |

## Setup

### Option A: Colab
1. Click the Colab badge for the notebook you want.
2. For notebooks 01 and 02, set the runtime to T4 GPU (Runtime, then Change runtime type).
3. Run the cells from top to bottom.

### Option B: VSCode with a Colab GPU
You can run the notebooks in VSCode and borrow a free Colab GPU, without opening the browser.
1. Install the Colab extension from the VSCode marketplace.
2. Open the `.ipynb` file.
3. Top right, click Select Kernel, then Existing Jupyter Server.
4. Choose Connect to Google Colab and sign in with your Google account.
5. Pick the T4 GPU runtime.

## Dataset note

Notebook 01 can also train on a small set of 16x16 Pokémon sprites. Those images are included for teaching only, they are non-commercial, and all rights belong to Nintendo, Game Freak, and The Pokémon Company. Details in [data/dataset/POKEMON_DATASET_NOTICE.md](data/dataset/POKEMON_DATASET_NOTICE.md).
