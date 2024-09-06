# Exploring the Inner Mechanisms of Large Generative Music Models

This repository contains the code and data associated with the paper "Exploring the Inner Mechanisms of Large Generative Music Models" In this paper, we focus on MusicGen, a state-of-the-art generative music model, and explore how existing interpretability techniques from the text domain can be transferred to the music domain. Our research provides insights into how MusicGen constructs human-interpretable musicological concepts.
[Read the Paper](./ISMIR_2024_Exploring_the_Inner_Mechanisms_of_Large_Generative_Music_Models.pdf), 
[Read the supplemental material](./ISMIR_2024_Exploring_the_Inner_Mechanisms_of_Large_Generative_Music_Models_supplemental.pdf)

Key contributions include:

- We demonstrate the application of DecoderLens, a technique that aims to provide insights into how MusicGen composes musical concepts over time.
- We employ interchange interventions to observe how individual components of the model contribute to the generation of specific instruments and genres.
- We identify several limitations of these techniques when applied to music, highlighting the need for adaptations tailored to the complexity of audio data.

## Contents
1. `audio_samples/`: contains audio samples of the DecoderLens experiments, Interchange intervention experiments, and the original unintervened audio samples, as displayed on our GitHub Pages: https://marcel-velez.github.io/musicgen-mech-interp/
2. `decoder_lens_outputs/`: directory where it will save the DecoderLens outputs.
3. `decoderlens/`: the codebase for the DecoderLens experiments.
4. `interchange_interventions/`: the codebase for the Interchange intervention experiments.
5. `mean_activations/`: directory we put the mean activations of the interchange intervention experiments, automatically saves mean activations when running the intervention pipeline, so you do not have to recompute every time.
6. `intervention_main.py`: the main file to run the interchange intervention experiments, see "usage of code - running interchange interventions" below for more details.
7. `decoderlens_main.py`: the main file to run the DecoderLens experiments, see "usage of code - running DecoderLens" below for more details.

## Installation
```bash
pip install -r requirements.txt
```

## Usage of code

### Running DecoderLens
```bash
python decoderlens_main.py
```

### Running interchange interventions

Running all intervention concept and desired concept pairs:
```bash
python intervention_main.py
```

#### Configuration
Explain the configuration options available to users. Describe each command-line argument or configuration file option and what it does.

- **`--fixed_concept`**: Set the fixed concept for training or evaluation.
  - _default_: `-1`
  - _type_: `int`
  - _description_: Use `-1` if no fixed concept is desired.

- **`--model_size`**: Choose the model size to be used.
  - _default_: `'small'`
  - _options_: ['small', 'medium', 'large']
  - _type_: `str`

- **`--n_prompts_per_concept`**: Specify the number of prompts to be generated per concept.
  - _default_: `100`
  - _type_: `int`

- **`--pre_computed_means`**: Bool for whether to use pre-computed means, or save the pre-computed means when they are not found.
  - _default_: `False`
  - _type_: `store_true` (sets to `True` when the flag is used)

- **`--gpus`**: Enable GPU support for training or evaluation.
  - _default_: `False`
  - _type_: `bool`
  - _action_: `store_true` (sets to `True` when the flag is used)

- **`--save_extremes`**: Save the extreme cases, top-k and bottom-k.
  - _default_: `False`
  - _type_: `bool`
  - _action_: `store_true` (sets to `True` when the flag is used)



If you have any questions, please feel free to contact us.

Citation:
```
@article{velezvasquez2024,
  title   = {Exploring the Inner Mechanisms of Large Generative Music Models},
  author  = {Vélez Vásquez, M.A. and Pouw, C. and Burgoyne, J.a. and Zuidema, W.},
  booktitle = {Proceedings of the 25th International Society for Music Information Retrieval Conference},
  year    = {2024},
}
```


