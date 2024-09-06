import os
import torch

from decoderlens import load_musicgen, set_custom_forward_musicgen, generate_music


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MusicGen model
    music_model = load_musicgen(device)
    os.makedirs('./decoder_lens_outputs', exist_ok=True)
    # Iterate over layers
    for TARGET in range(1, 25):

        set_custom_forward_musicgen(music_model, TARGET)

        text_prompt = 'Compose a happy rock piece with a guitar melody. Use a fast tempo.'
        audio = generate_music(music_model, text_prompt, save_file=True, output_file=f'./decoder_lens_outputs/layer_{TARGET}.wav')
        print(f"*** Audio for layer {TARGET} generated. ***")
