import pickle
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def print_label():
    label_file = open("meta", "rb")
    label_pkl = pickle.load(label_file)
    label_pkl = label_pkl["fine_label_names"]
    label_file.close()


dataset = "cifar10"
le_name = dataset + "_le.pickle"
label_name = dataset + "_label.txt"

# with open(label_name, "rb") as label_file:
#     labels = label_file.read().splitlines()
label_pkl = [line.strip() for line in open(label_name, 'r')]

labels = []
i = 0
for l in label_pkl:
    # l = l.split(',')[0]
    l = l.replace("_", " ")
    l = l.replace("-", " ")
    if l[0] in ["a", "o", "u", "e", "i"]:
        l = "a image of " + l
    else:
        l = "a image of " + l
    labels.append(l)

print(labels)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

if dataset == "imagenet":
    for i in range(100):
        print(i)
        text = clip.tokenize(labels[i*10:(i+1)*10]).to(device)
        if i == 0:
            text_features = model.encode_text(text).detach().cpu().numpy()
        else:
            t = model.encode_text(text).detach().cpu().numpy()
            text_features = np.concatenate((text_features, t))

text_features = torch.Tensor(text_features)
with open(le_name, "wb") as output_file:
    pickle.dump(text_features, output_file)

with open(le_name, "rb") as label_file:
    le = pickle.load(label_file)

print(text_features)
print(le)

label_emb = text_features.cpu().detach().numpy()
X_embedded = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity = 3).fit_transform(label_emb)
for i in range(10):
    txt = label_pkl[i]
    x = X_embedded[i,1]
    y = X_embedded[i,0]
    plt.scatter(x, y, linewidth=5, label=txt)
    plt.text(s=txt, x=x, y=y+2, fontsize=10)

plt.legend()
plt.show()