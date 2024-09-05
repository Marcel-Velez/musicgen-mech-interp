import torch
from utils import load_musicgen
from prompts import guitar_desc
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from librosa.display import waveshow
import numpy as np
from audiocraft.modules.transformer import create_sin_embedding
from scipy.io import wavfile


def set_custom_forward_musicgen(model, target_layer):
    def custom_forward_transformer(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        if 'offsets' in self._streaming_state:
            offsets = self._streaming_state['offsets']
        else:
            offsets = torch.zeros(B, dtype=torch.long, device=x.device)

        if self.positional_embedding in ['sin', 'sin_rope']:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
            x = x + self.positional_scale * pos_emb

        for i, layer in enumerate(self.layers):
            if i == target_layer:
                break
            else:
                x = self._apply_layer(layer, x, *args, **kwargs)

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return x
    # Add custom_forward function to the transformer, so that it can return outputs of intermediate layers
    bound_method_transformer = custom_forward_transformer.__get__(model.lm.transformer, model.lm.transformer.__class__)
    setattr(model.lm.transformer, 'forward', bound_method_transformer)
    
    
def generate_music(model: nn.Module, prompts: str|list, duration: int=4, output_file: str="musicgen_out.wav", save_file: bool=False) -> torch.Tensor:
  if isinstance(prompts, str):
    prompts = [prompts]
  model.set_generation_params(duration=duration)  # generate 4 seconds.
  torch.manual_seed(42)
  generation = model.generate(prompts)

  # save the files if desired
  if isinstance(prompts, list) and len(prompts)>1:
    for ind, aud in enumerate(generation):
      if save_file:
        sf.write(f'{ind}_{output_file}', aud.cpu().numpy().squeeze(), MUSICGEN_SAMPLE_RATE)
  else:
      if save_file:
        sf.write(f'{output_file}', generation.cpu().numpy().squeeze(), MUSICGEN_SAMPLE_RATE)

  return generation


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MusicGen model
    model = load_musicgen(device)

    # Iterate over layers
    for TARGET in range(1, 25):

        set_custom_forward_musicgen(music_model, TARGET)
	text_prompt = 'Compose a happy rock piece with a guitar melody. Use a fast tempo.'
	audio = generate_music(music_model, text_prompt, save_file=True, output_file=f'musicgen_layer_{TARGET}.wav')
	print(f"*** Audio for layer {TARGET} generated. ***")
        wavfile.write(f'decoder_lens_outputs/layer{TARGET}.wav', 32000, np.array(audio))
