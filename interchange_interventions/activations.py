import torch
from collections import defaultdict
import pickle as pkl
import math

from model_utils import clear_model_hooks, apply_pre_transformer_hook, clear_pre_transformer_hook


def get_activations_for_dataset(model, desc: list, batch_size: int = 4, pre_comp: bool = False):
    # Create a list of dictionaries to store the activations of each sample.
    activations = []

    # clear any existing hooks and apply new hooks
    clear_pre_transformer_hook(model)
    clear_model_hooks(model)
    apply_pre_transformer_hook(model)

    def initialize_activations():
        # Prepare new dictionary for the activations of a new batch.
        activations.append({'self_attention': defaultdict(list), 'cross_attention': defaultdict(list),
                            'mlp': defaultdict(list)})

    def create_hook_fn(att_type, layer_id):
        def hook(module, input, output):
            processed_out = average_cond_uncond(output.unsqueeze(0).detach().cpu())
            if isinstance(activations[-1][att_type][layer_id], list):
                activations[-1][att_type][layer_id] = processed_out
            else:
                activations[-1][att_type][layer_id] = torch.concat(
                    (activations[-1][att_type][layer_id], processed_out)
                )
        return hook

    # Register the hooks.
    for i, layer in enumerate(model.lm.transformer.layers):
        layer.self_attn.out_proj.register_forward_hook(create_hook_fn('self_attention', i))
        layer.cross_attention.out_proj.register_forward_hook(create_hook_fn('cross_attention', i))
        layer.linear2.register_forward_hook(create_hook_fn('mlp', i))

    # Go through your dataset and compute the forward passes.
    generations = []
    if len(desc) > batch_size:
        with torch.no_grad():
            for i in range(0, len(desc), batch_size):
                initialize_activations()
                torch.manual_seed(42)

                batch_generations = model.generate(desc[i:(i + batch_size)]).detach().cpu()
                generations = batch_generations if not generations else torch.concat((generations, batch_generations))
    else:
        with torch.no_grad():
            initialize_activations()
            torch.manual_seed(42)
            generations = model.generate(desc).detach().cpu()

    activations = unnest_activations(activations)
    # Unregister the hooks to make sure they don't interfere with the next dataset.
    clear_model_hooks(model)

    return generations, activations

def generate_sample_value(classifier: torch.nn.Module, generations: torch.Tensor, concept_idx: int, batch_size: int = 4, as_distribution=True, value_type='prob', device=None):
    assert value_type in ['softmax', 'sigmoid', 'logit'], f"value_type: '{value_type}' \
                                                                not in ['softmax','sigmoid', 'logit']"

    results = generations.squeeze(1)
    num_samples = len(results)

    all_logits = None

    def process_logits(logits):
        if value_type == 'prob':
            processed_logits = torch.nn.functional.softmax(logits.cpu(), dim=1)
        elif value_type == 'sigmoid':
            processed_logits = torch.nn.functional.sigmoid(logits.cpu())
        else:
            processed_logits = logits[:, concept_idx].cpu().detach().numpy()
        return processed_logits

    with torch.no_grad():
        if batch_size < num_samples:
            for i in range(0, num_samples, batch_size):
                logits = classifier(results[i:i + batch_size].to(device))
                logits = process_logits(logits)
                if all_logits is None:
                    all_logits = logits
                else:
                    all_logits = torch.concat((all_logits, logits)) if as_distribution else logits
        else:
            logits = classifier(results.to(device))
            all_logits = process_logits(logits)

    return all_logits


def unnest_activations(nested_activation_dict):
    unnested_activation_dict = {'self_attention': {}, 'cross_attention': {}, 'mlp': {}}

    for batch in nested_activation_dict:
        for att_type, layers in batch.items():
            for layer_id, activations in layers.items():
                if layer_id not in unnested_activation_dict[att_type].keys():
                    unnested_activation_dict[att_type][layer_id] = activations
                else:
                    previous_midpoint = unnested_activation_dict[att_type][layer_id].shape[1]//2
                    new_midpoint = activations.shape[1]//2

                    cond_prev, uncond_prev = unnested_activation_dict[att_type][layer_id].split(previous_midpoint, dim=1)
                    cond_new, uncond_new = activations.split(new_midpoint, dim=1)
                    unnested_activation_dict[att_type][layer_id] = torch.concat((cond_prev, cond_new, uncond_prev, uncond_new),dim=1)
    return unnested_activation_dict


def load_or_compute_mean_activations(file_path, model, corpus, batch_size):
    if file_path:
        with open(file_path, 'rb') as file:
            mean_activations = pkl.load(file)
    else:
        _, activations = get_activations_for_dataset(model, corpus, batch_size)
        mean_activations = get_mean_activations(activations)
    return mean_activations


def average_cond_uncond(latent_tensor):
    cond_tensor = latent_tensor[:, :latent_tensor.shape[1] // 2].mean(dim=1, keepdim=True)
    uncond_tensor = latent_tensor[:, latent_tensor.shape[1] // 2:].mean(dim=1, keepdim=True)
    return torch.concat([cond_tensor, uncond_tensor], dim=1)



def get_mean_activations(activations):
    mean_activations = defaultdict(lambda: defaultdict(list))

    for attention_type, layers in activations.items():
        for layer_id, activation in layers.items():
            mean_activations[attention_type][layer_id] = average_cond_uncond(activation)
    return dict(mean_activations)


class BaseActivationIterator:
    def __init__(self, activations, layer_id, attention_id, batch_size, max_layers=None, batch_offset=0, num_attention=3,
                 device=None):
        assert attention_id < 3, "Only support 0: self_attention, 1: cross_attention, and 2: mlp"

        self.activations = activations
        self.device = device
        self.counter = 0
        self.layer_id = layer_id
        self.attention_id = attention_id
        self.batch_offset = batch_offset
        self.batch_size = batch_size
        self.max_layers = max_layers

        self.new_input = len(activations)
        self.new_sample = math.ceil(self.max_layers * num_attention / self.batch_size)
        self.att_offset = self.attention_id * self.max_layers
        self.intervene_idx = self.calculate_intervene_idx()

    def calculate_intervene_idx(self):
        return self.layer_id + self.att_offset - (self.batch_offset * self.batch_size)

    def update_counter(self):
        self.counter += 1
        if self.counter % self.new_input == 0 and self.counter != 0:
            self.batch_offset += 1
            self.intervene_idx = self.calculate_intervene_idx()
            if self.batch_offset >= self.new_sample:
                self.batch_offset = 0

    def get_next_activation(self):
        raise NotImplementedError("This method should be implemented by subclasses")


class ActivationIterator(BaseActivationIterator):
    def get_next_activation(self):
        activation = self.activations[self.counter % self.new_input]
        self.update_counter()
        return activation


class DoubleActivationIterator(BaseActivationIterator):
    def __init__(self, activations_a, activations_b, layer_id, attention_id, batch_size, batch_offset=0, num_attention=3,
                 max_layers=None, device=None):
        super().__init__(activations_a, layer_id, attention_id, batch_size, batch_offset, num_attention, max_layers, device)
        assert len(activations_a) == len(activations_b), "Lengths of activations A and B must match"
        self.activations_b = activations_b

    def get_next_activation(self):
        activation_a = self.activations[self.counter % self.new_input]
        activation_b = self.activations_b[self.counter % self.new_input]
        self.update_counter()
        return activation_b - activation_a


def inject_activation_hook_fn(activation_iterator, module, input, output, intervention_type="adjust"):
    intervene_idx = activation_iterator.intervene_idx
    device = activation_iterator.device
    if 0 <= intervene_idx < (input[0].shape[0] // 2):
        intervention_output = activation_iterator.get_next_activation()
        if intervention_type == "adjust":
            output[intervene_idx] += intervention_output[0].to(device)
        elif intervention_type == "replace":
            output[intervene_idx] = intervention_output[0].to(device)
    else:
        activation_iterator.update_counter()
    return output

