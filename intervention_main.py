import torch
from argparse import ArgumentParser
from audiocraft.models import MusicGen
from decoderlens.utils import load_model
from interchange_interventions import single_intervene_pipeline, PromptGenerator

def main():
    # Argument Parsing
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--fixed_concept', type=int, default=-1)
    argument_parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large'])
    argument_parser.add_argument('--n_prompts_per_concept', type=int, default=100)
    argument_parser.add_argument('--pre_computed_means', default=False, action='store_true')
    argument_parser.add_argument('--gpus', default=False, action='store_true')
    argument_parser.add_argument('--save_extremes', default=False, action='store_true')

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

    all_concepts = {**instruments, **genres}

    # Load Prompts
    prompt_gen = PromptGenerator()

    prompts = {}
    for key, (name, _) in all_concepts.items():
        prompts[key] = prompt_gen.get_prompts_for(name, '[INSTRUMENTS]' if key < 4 else '[GENRE]',
                                       n_total=args.n_prompts_per_concept, n_per_type=1000)

    # Intervention Loop
    for i in range(10):
        for j in range(10):
            if i != j and (i == args.fixed_concept or args.fixed_concept == -1):
                to_be_intervened = all_concepts[i]
                to_intervene_with = all_concepts[j]
                prompt_type = f"{to_be_intervened[0]}_{to_intervene_with[0]}"

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
                    prompt_type=prompt_type,
                    mean_a_file=mean_a_file,
                    mean_b_file=mean_b_file,
                    model_size=args.model_size,
                    save_extremes=args.save_extremes,
                    pre_computed=args.pre_computed_means
                )


if __name__ == '__main__':
    main()
