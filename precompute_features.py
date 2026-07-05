import torch, os, pickle, glob, re
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from app import Vocab, ResNetEncoder, GRUDecoder, TinyViTEncoder, TransformerDecoder
from app import load_resnet_gru, load_tinyvit, BASE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(os.path.join(BASE_DIR, "captions.csv"))
img_base = BASE_DIR

eval_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_class(path):
    return path.split("/")[-2]

def compute_features(encoder, name):
    encoder.eval()
    features, classes, nepali_captions, img_paths = [], [], [], []
    paths = df["image_path"].tolist()
    for p in tqdm(paths, desc=f"Extracting {name} features"):
        full_path = os.path.join(img_base, p)
        if not os.path.exists(full_path):
            continue
        img = Image.open(full_path).convert("RGB")
        img_t = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = encoder(img_t).cpu().numpy().flatten()
        features.append(feat)
        classes.append(extract_class(p))
        row = df[df["image_path"] == p].iloc[0]
        nepali_captions.append(row["caption_ne"])
        img_paths.append(p)
    return features, classes, nepali_captions, img_paths

for model_name, model_path, load_fn in [
    ("resnet_gru", "1_out", load_resnet_gru),
    ("tinyvit", "2_out", load_tinyvit),
]:
    encoder, decoder, vocab, _, config, _ = load_fn(os.path.join(BASE_DIR, model_path))
    features, classes, nepali_captions, img_paths = compute_features(encoder, model_name)
    out = {
        "features": features,
        "classes": classes,
        "nepali_captions": nepali_captions,
        "img_paths": img_paths,
    }
    out_path = os.path.join(BASE_DIR, f"features_{model_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {len(features)} features to {out_path}")
