import torch
from argparse import ArgumentParser
from musicgen import MusicGen  # Assuming MusicGen is imported from somewhere
from utils import load_model, save_to_file
from intervention_pipeline import single_intervene_pipeline
from prompt_generation import get_prompts_for


def main():
    # Argument Parsing
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--fixed_concept', type=int, default=-1)
    argument_parser.add_argument('--model_size', type=str, default='medium', choices=['small', 'medium', 'large'])
    argument_parser.add_argument('--n_prompts_per_concept', type=int, default=100)
    argument_parser.add_argument('--gpus', type=bool, default=False, action='store_true')



    args = argument_parser.parse_args()

    # Device Configuration
    if args.gpus:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load MusicGen model and instrument classifier
    model = MusicGen.get_pretrained(f'facebook/musicgen-{args.model_size}', device=device)
    classifier = load_model(mode="logits").to(device)
    model.set_generation_params(duration=4, use_sampling=True)
    classifier.eval()

    # Fixed IDs of PAssT model and names
    instruments = {
        0: ('guitar', 140),
        1: ('piano', 153),
        2: ('trumpet', 187),
        3: ('violin', 191)
    }

    genres = {
        4: ('classical', 237),
        5: ('pop', 216),
        6: ('jazz', 235),
        7: ('rock', 219),
        8: ('electronic', 245),
        9: ('hiphop', 217)
    }

    all_stuff = {**instruments, **genres}

    # Load Prompts
    prompts = {}
    for key, (name, _) in all_stuff.items():
        prompts[key] = get_prompts_for(name, '[INSTRUMENTS]' if key < 4 else '[GENRE]',
                                       n_total=args.n_prompts_per_concept, n_per_type=1000)[0]

    # Intervention Loop
    for i in range(10):
        for j in range(10):
            if i != j and (i == args.fixed_concept or args.fixed_concept == -1):
                to_be_intervened = all_stuff[i]
                to_intervene_with = all_stuff[j]
                type_prompt = f"{to_be_intervened[0]}_{to_intervene_with[0]}"

                # Mean Activation File Paths
                mean_a_file = f'./mean_activations/{args.model_size}_{to_be_intervened[0]}_mean_activations.pkl'
                mean_b_file = f'./mean_activations/{args.model_size}_{to_intervene_with[0]}_mean_activations.pkl'

                print(f"Intervening {to_be_intervened[0]} with {to_intervene_with[0]}")

                generated_audio_single_interv = single_intervene_pipeline(
                    model, classifier,
                    prompts[i], prompts[j],
                    to_be_intervened[1], to_intervene_with[1],
                    batch_size=48,
                    device=device,
                    type_prompt=type_prompt,
                    mean_a_file=mean_a_file,
                    mean_b_file=mean_b_file,
                    model_size=args.model_size
                )


if __name__ == '__main__':
    main()
