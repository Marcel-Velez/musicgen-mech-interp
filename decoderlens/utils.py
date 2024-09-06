import csv
from audiocraft.models import musicgen
from hear21passt.base import load_model
from math import log2

def load_musicgen(device):
    model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small', device=device)
    model.set_generation_params(duration=4, use_sampling=True)
    return model

def load_classifier(device):
    classifier = load_model(mode="logits")
    classifier.to(device)
    return classifier

def load_labels():
    # run this in command line: wget https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv
    filename = "class_labels_indices.csv"
    labels = []

    with open(filename, "r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            labels.append(row["display_name"])

    return labels

def kl_divergence(p, q):
 return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))