import torch
from collections import defaultdict


def model_generate(model, descriptions):
    torch.manual_seed(42)
    return model.generate(descriptions)


def classifier_forward(model, generation):
    torch.manual_seed(42)
    return model(generation)


def inject_fixate_transformer_pre_hook_fn(module, input, **kwargs):
    cond_inp, uncond_inp = input[0].split(input[0].shape[0] // 2)
    cond_inp[:] = cond_inp[0] if True else cond_inp
    uncond_inp[:] = uncond_inp[0] if False else uncond_inp
    new_input = torch.concat((cond_inp, uncond_inp))
    return new_input


def inject_fixate_fuser_hook_fn(module, input, output, **kwargs):
    cond_inp, uncond_inp = output[0].split(output[0].shape[0] // 2)
    cond_inp[:] = cond_inp[0] if True else cond_inp
    uncond_inp[:] = uncond_inp[0] if True else uncond_inp
    new_input = torch.concat((cond_inp, uncond_inp))

    cond_cross, uncond_cross = output[1].split(output[1].shape[0] // 2)
    cond_cross[:] = cond_cross[0] if True else cond_cross
    uncond_cross[:] = uncond_cross[0] if True else uncond_cross
    new_cross = torch.concat((cond_cross, uncond_cross))

    return new_input, new_cross


def clear_model_hooks(model):
    for layer in model.lm.transformer.layers:
        layer.self_attn.out_proj._forward_hooks.clear()
        layer.cross_attention.out_proj._forward_hooks.clear()
        layer.linear2._forward_hooks.clear()

        layer.self_attn.out_proj._forward_pre_hooks.clear()
        layer.cross_attention._forward_pre_hooks.clear()
        layer.linear2._forward_pre_hooks.clear()



def apply_pre_transformer_hook(model, fuser=False, both=False):
    if fuser:
        model.lm.fuser.register_forward_hook(inject_fixate_fuser_hook_fn)
    elif both:
        model.lm.fuser.register_forward_hook(inject_fixate_fuser_hook_fn)
        model.lm.transformer.register_forward_pre_hook(inject_fixate_transformer_pre_hook_fn)
    else:
        model.lm.transformer.register_forward_pre_hook(inject_fixate_transformer_pre_hook_fn)


def clear_pre_transformer_hook(model):
    model.lm.transformer._forward_pre_hooks.clear()
    model.lm.fuser._forward_hooks.clear()


def unnest_intervention_lists(nested_list):
    unnested_intervention_dict = {}

    for descr_idx, description_batches in enumerate(nested_list):
        for batch_idx, batch in enumerate(description_batches):
            unnested_list = batch if batch_idx == 0 else torch.cat((unnested_list, batch))

        unnested_intervention_dict[descr_idx] = unnested_list

    return unnested_intervention_dict


def compute_average_logit_difference(logit_differences_per_song):
    accumulated_logits = defaultdict(int)

    for song, logit_differences in logit_differences_per_song.items():
        for layer_index, logit_diff in enumerate(logit_differences):
            accumulated_logits[layer_index] += logit_diff

    average_logits_diff = {
        layer: (total_logit_diff / len(logit_differences_per_song)).item()
        for layer, total_logit_diff in accumulated_logits.items()
    }

    return average_logits_diff
