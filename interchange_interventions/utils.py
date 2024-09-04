
# utils.py
import torch
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from audiocraft.utils.notebook import display_audio
import pickle as pkl

MUSICGEN_SAMPLE_RATE = 32_000

def load_labels(filename):
    labels = []
    with open(filename, "r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            labels.append(row["display_name"])
    return labels


def reorganize_for_plotting(logit_diff_by_layers, num_layers):
    attention_ordered = {'self': [], 'cross': [], 'mlp': []}

    for module_index, module_name in enumerate(attention_ordered.keys()):
        for layer in range(num_layers):
            attention_ordered[module_name].append(logit_diff_by_layers[layer + module_index * num_layers])

    return attention_ordered


def plot_heatmap(logit_differences, title, xlabel='Module', ylabel='Layer', figsize=(10, 8)):
    logit_diff_df = pd.DataFrame(logit_differences)
    plt.figure(figsize=figsize)
    sns.heatmap(logit_diff_df, cmap='coolwarm', annot=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()


# Helper function to save audio files
def save_audio_file(file_path, audio_tensor):
    wavfile.write(file_path, MUSICGEN_SAMPLE_RATE, np.array(audio_tensor.unsqueeze(1)))


# Helper function to display or save audio
def handle_audio_output(audio_tensor, save_wav, file_path=None):
    if save_wav:
        save_audio_file(file_path, audio_tensor)
    else:
        display_audio(audio_tensor.unsqueeze(0), MUSICGEN_SAMPLE_RATE)


def listen_to_extremes(audio_data, scores, num_layers, top_k=4, prompt_type=None, save_wav=False, model_size=None):
    for prompt_id, score_list in scores.items():
        print(f"Prompt {prompt_id}")
        best_indices = torch.topk(torch.Tensor(score_list), top_k).indices
        worst_indices = torch.topk(torch.Tensor(score_list), top_k, largest=False).indices

        print("Best adjusted:")
        for idx in best_indices:
            print('\t', idx.item())
            audio_tensor = audio_data[prompt_id][idx.item()]
            file_path = f"./intervened_audio/{model_size}_{prompt_type}_prompt_{prompt_id}_best_{idx.item()}.wav"
            handle_audio_output(audio_tensor, save_wav, file_path)

        print("Worst adjusted:")
        for idx in worst_indices:
            print('\t', idx.item())
            audio_tensor = audio_data[prompt_id][idx.item()]
            file_path = f"./intervened_audio/{model_size}_{prompt_type}_prompt_{prompt_id}_worst_{idx.item()}.wav"
            handle_audio_output(audio_tensor, save_wav, file_path)


# Function to save data to a file
def save_to_file(data, file_path):
    with open(file_path, 'wb') as file:
        pkl.dump(data, file)