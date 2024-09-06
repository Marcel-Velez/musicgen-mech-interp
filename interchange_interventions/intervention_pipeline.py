import torch
from torch.nn.functional import kl_div
from functools import partial
import os

from .model_utils import model_generate, classifier_forward, clear_model_hooks, apply_pre_transformer_hook, clear_pre_transformer_hook, unnest_intervention_lists
from .activations import get_activations_for_dataset, load_or_compute_mean_activations, generate_sample_value, ActivationIterator, DoubleActivationIterator, inject_activation_hook_fn
from .utils import save_to_file, listen_to_extremes


def apply_model_hooks(model, mean_activations_a, batch_size, double_act=False, device=None, mean_activations_b=None,
                      intervention_type="adjust"):
    iterator_class = DoubleActivationIterator if double_act else ActivationIterator

    for i, layer in enumerate(model.lm.transformer.layers):
        def create_iterator(attention_type, attention_id):
            return iterator_class(
              mean_activations_a[attention_type][i],
              mean_activations_b[attention_type][i] if double_act else None,
              layer_id=i,
              attention_id=attention_id,
              batch_size=batch_size,
              device=device
            )

    activation_iterator_self = create_iterator('self_attention', 0)
    activation_iterator_cross = create_iterator('cross_attention', 1)
    activation_iterator_mlp = create_iterator('mlp', 2)

    hook_fn_self = partial(inject_activation_hook_fn, activation_iterator_self, intervention_type=intervention_type)
    hook_fn_cross = partial(inject_activation_hook_fn, activation_iterator_cross, intervention_type=intervention_type)
    hook_fn_mlp = partial(inject_activation_hook_fn, activation_iterator_mlp, intervention_type=intervention_type)

    layer.self_attn.out_proj.register_forward_hook(hook_fn_self)
    layer.cross_attention.out_proj.register_forward_hook(hook_fn_cross)
    layer.linear2.register_forward_hook(hook_fn_mlp)



def prepare_model_for_intervention(model, mean_activations_a, batch_size, mean_activations_b=None, device=None):
    # Clear existing hooks and prepare for intervention
    clear_model_hooks(model)
    clear_pre_transformer_hook(model)
    apply_pre_transformer_hook(model)

    # Determine if we are using double activations and apply appropriate hooks
    double_act = mean_activations_b is not None
    apply_model_hooks(
        model,
        mean_activations_a,
        batch_size,
        double_act=double_act,
        mean_activations_b=mean_activations_b,
        device=device
    )



def single_intervene_ablation(
    model, classifier, descriptions, old_probs,
    mean_activations_a, mean_activations_b,
    old_instr_idx, new_instr_idx,
    batch_size=4, model_layers=24, device=None
):
    prepare_model_for_intervention(
        model, mean_activations_a=mean_activations_a,
        mean_activations_b=mean_activations_b,
        batch_size=batch_size, device=device
    )

    total_interventions = 3 * model_layers  # 3 attention types
    results = {
        'ratios': {}, 'kl_divs': {}, 'all_probs': {}, 'all_audio': {}, 'old_instr': {}, 'new_instr': {}
    }

    for descr_idx, description in enumerate(descriptions):
        print(f"Intervening: {description}")
        initialize_result_containers(results, descr_idx)

        batch_descriptions = [description] * batch_size

        for batch_id in range(0, total_interventions, batch_size):
            upper = min(batch_size, total_interventions - batch_id)
            process_intervention_batch(
                model, classifier, batch_descriptions[:upper],
                old_probs[descr_idx], old_instr_idx, new_instr_idx,
                results, descr_idx, batch_id
            )

    return post_process_results(results)


# Main pipeline function
def single_intervene_pipeline(model, classifier, corpus_a, corpus_b, instr_a_idx, instr_b_idx,
                              batch_size, device=None, prompt_type=None, mean_a_file=None,
                              mean_b_file=None, model_size=None, save_extremes=False, pre_computed=False):
    num_layers = len(model.lm.transformer.layers)

    print("Starting activations for corpus A")
    corpus_a_audio, corpus_a_activations = get_activations_for_dataset(model, corpus_a, batch_size)
    corpus_a_mean_activations = load_or_compute_mean_activations(mean_a_file, model, corpus_a, batch_size,
                                                                 pre_computed=pre_computed)

    print("Starting activations for corpus B")
    print("Generating probabilities for corpus A")
    corpus_a_probs = generate_sample_value(classifier, corpus_a_audio, instr_a_idx, device=device)

    print("Starting mean activations for corpus B")
    corpus_b_mean_activations = load_or_compute_mean_activations(mean_b_file, model, corpus_b, batch_size)

    print("Performing intervention")
    a_b_ratios, generated_audio, corpus_kl_div, all_probs = single_intervene_ablation(
        model, classifier, corpus_a, corpus_a_probs, corpus_a_mean_activations,
        corpus_b_mean_activations, instr_a_idx, instr_b_idx, batch_size, num_layers, device
    )

    os.mkdirs("./intervention_results", exist_ok=True)
    results_dir = "./intervention_results"
    save_to_file(corpus_kl_div, f"{results_dir}/{model_size}_{prompt_type}_kl_divs.pkl")
    save_to_file(all_probs, f"{results_dir}/{model_size}_{prompt_type}_all_probs.pkl")
    save_to_file(a_b_ratios, f"{results_dir}/{model_size}_{prompt_type}_concept_ratios.pkl")

    if save_extremes:
        listen_to_extremes(generated_audio, a_b_ratios, num_layers=num_layers,
                                   top_k=1, type_prompt=prompt_type, save_wav=True, model_size=model_size)

    return a_b_ratios, generated_audio, corpus_kl_div, all_probs


def initialize_result_containers(results, descr_idx):
    results['ratios'][descr_idx] = []
    results['kl_divs'][descr_idx] = []
    results['all_probs'][descr_idx] = []
    results['all_audio'][descr_idx] = []


def process_intervention_batch(
    model, classifier, batch_descriptions, old_probs,
    original_concept_idx, desired_concept_idx, results, descr_idx, batch_id,
):
    with torch.no_grad():
        r_ablate = model_generate(model, batch_descriptions)
        logits_ablate = classifier_forward(classifier, r_ablate.squeeze())

        results['all_audio'][descr_idx].append(r_ablate.squeeze().cpu().detach())

        new_probs = torch.nn.functional.softmax(logits_ablate.cpu(), dim=1)

        score = calculate_intervention_scores(old_probs, new_probs, original_concept_idx, desired_concept_idx)
        results['ratios'][descr_idx].append(score)

        kl_div_val = kl_div(torch.log(old_probs), torch.log(new_probs), reduction='none', log_target=True)
        results['kl_divs'][descr_idx].append(kl_div_val)

        results['all_probs'][descr_idx].append(new_probs)


def calculate_intervention_scores(old_probs, new_probs, original_concept_idx, desired_concept_idx):
    old_scores = torch.log(old_probs[original_concept_idx] / old_probs[desired_concept_idx])
    new_scores = torch.log(new_probs[:, original_concept_idx] / new_probs[:, desired_concept_idx])
    return old_scores - new_scores


def post_process_results(results):
    unnested_ratio = unnest_intervention_lists(results['ratios'])
    unnested_all_probs = unnest_intervention_lists(results['all_probs'])
    unnested_kl_div = unnest_intervention_lists(results['kl_divs'])
    unnested_audio = unnest_intervention_lists(results['all_audio'])

    return unnested_ratio, unnested_audio, unnested_kl_div, unnested_all_probs


