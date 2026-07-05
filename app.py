import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os, re, pickle
from sklearn.neighbors import NearestNeighbors

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", text)
    return text.split()

class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(self.itos)}
    def __len__(self):
        return len(self.itos)

# ========== ResNet-50 + Bahdanau Attention + GRU ==========

class ResNetEncoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super().__init__()
        resnet = torchvision.models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
    def forward(self, images):
        x = self.backbone(images)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 3, 1)
        return x

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.tanh(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, decoder_dim, attention_dim, encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
    def init_hidden_state(self, encoder_out):
        return torch.tanh(self.init_h(encoder_out.mean(dim=1)))

# ========== Tiny ViT + Transformer Decoder ==========

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TinyViTEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=256, depth=4, nhead=8, dim_feedforward=512, dropout=0.1, image_size=224):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth, enable_nested_tensor=False)
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, depth=4, nhead=8, dim_feedforward=512, dropout=0.1, max_len=22):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, depth)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        tgt = self.embedding(tgt)
        tgt = tgt + self.pos_embed[:, :tgt.size(1), :]
        tgt = self.pos_drop(tgt)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        return self.fc(out)

# ========== Model loading ==========

BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_resnet_gru(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(model_path, map_location=device)
    config = ck["config"]
    vocab = Vocab(ck["vocab_itos"])
    encoder = ResNetEncoder(config["ENC_IMAGE_SIZE"]).to(device)
    decoder = GRUDecoder(len(vocab), config["EMBED_DIM"], config["DECODER_DIM"],
                         config["ATTENTION_DIM"], config["ENCODER_DIM"], config["DROPOUT"]).to(device)
    encoder.load_state_dict(ck["encoder"])
    decoder.load_state_dict(ck["decoder"])
    encoder.eval(); decoder.eval()
    eval_tf = T.Compose([
        T.Resize((config["IMAGE_SIZE"], config["IMAGE_SIZE"])),
        T.ToTensor(),
        T.Normalize(config["NORM_MEAN"], config["NORM_STD"])
    ])
    return encoder, decoder, vocab, eval_tf, config, device

@st.cache_resource
def load_tinyvit(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(model_path, map_location=device)
    config = ck["config"]
    vocab = Vocab(ck["vocab_itos"])
    encoder = TinyViTEncoder(
        in_channels=3, patch_size=config["PATCH_SIZE"], embed_dim=config["HIDDEN_DIM"],
        depth=config["ENC_LAYERS"], nhead=config["ENC_NHEAD"],
        dim_feedforward=config["ENC_DIM_FEED"], dropout=config["ENC_DROPOUT"],
        image_size=config["IMAGE_SIZE"]).to(device)
    decoder = TransformerDecoder(
        len(vocab), embed_dim=config["HIDDEN_DIM"], depth=config["DEC_LAYERS"],
        nhead=config["DEC_NHEAD"], dim_feedforward=config["DEC_DIM_FEED"],
        dropout=config["DEC_DROPOUT"], max_len=config["MAX_CAPTION_LEN"]).to(device)
    encoder.load_state_dict(ck["encoder"])
    decoder.load_state_dict(ck["decoder"])
    encoder.eval(); decoder.eval()
    eval_tf = T.Compose([
        T.Resize((config["IMAGE_SIZE"], config["IMAGE_SIZE"])),
        T.ToTensor(),
        T.Normalize(config["NORM_MEAN"], config["NORM_STD"])
    ])
    return encoder, decoder, vocab, eval_tf, config, device

@st.cache_resource
def load_features(features_path):
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    feat = np.array(data["features"])
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(feat)
    return nn_model, data["classes"], data["nepali_captions"]

def predict_class(encoder, image, eval_tf, device, nn_model):
    img = eval_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = encoder(img).cpu().numpy().flatten().reshape(1, -1)
    dists, idxs = nn_model.kneighbors(feat)
    return idxs[0][0]

# ========== Inference ==========

@torch.no_grad()
def caption_resnet_gru(image, encoder, decoder, vocab, eval_tf, config, device):
    pad_idx, start_idx, end_idx = vocab.stoi[PAD], vocab.stoi[START], vocab.stoi[END]
    img = eval_tf(image).unsqueeze(0).to(device)
    enc = encoder(img).view(1, -1, config["ENCODER_DIM"])
    k = config.get("BEAM_SIZE", 3)
    num_pixels = enc.size(1)
    enc = enc.expand(k, num_pixels, config["ENCODER_DIM"])
    seqs = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
    top_scores = torch.zeros(k, 1, device=device)
    h = decoder.init_hidden_state(enc)
    complete_seqs, complete_scores = [], []
    V = len(vocab)
    max_len = config["MAX_CAPTION_LEN"]
    for step in range(1, max_len + 1):
        emb = decoder.embedding(seqs[:, -1])
        context, _ = decoder.attention(enc, h)
        context = decoder.sigmoid(decoder.f_beta(h)) * context
        h = decoder.gru(torch.cat([emb, context], dim=1), h)
        scores = F.log_softmax(decoder.fc(h), dim=1)
        scores = top_scores.expand_as(scores) + scores
        if step == 1:
            top_scores, top_words = scores[0].topk(k, 0)
        else:
            top_scores, top_words = scores.view(-1).topk(k, 0)
        prev = torch.div(top_words, V, rounding_mode="floor")
        next_word = top_words % V
        seqs = torch.cat([seqs[prev], next_word.unsqueeze(1)], dim=1)
        incomplete = [i for i, w in enumerate(next_word) if w.item() != end_idx]
        complete = [i for i in range(len(next_word)) if i not in incomplete]
        for i in complete:
            complete_seqs.append(seqs[i].tolist())
            complete_scores.append(top_scores[i].item())
        k -= len(complete)
        if k == 0:
            break
        seqs = seqs[incomplete]
        h = h[prev[incomplete]]
        enc = enc[prev[incomplete]]
        top_scores = top_scores[incomplete].unsqueeze(1)
    if not complete_seqs:
        complete_seqs = seqs.tolist()
        complete_scores = top_scores.squeeze(1).tolist()
    best = complete_seqs[int(np.argmax(complete_scores))]
    words = [vocab.itos[w] for w in best if w not in (pad_idx, start_idx, end_idx)]
    return " ".join(words)

@torch.no_grad()
def caption_tinyvit(image, encoder, decoder, vocab, eval_tf, config, device):
    pad_idx, start_idx, end_idx = vocab.stoi[PAD], vocab.stoi[START], vocab.stoi[END]
    img = eval_tf(image).unsqueeze(0).to(device)
    memory = encoder(img)
    seq = torch.full((1, 1), start_idx, dtype=torch.long, device=device)
    max_len = config["MAX_CAPTION_LEN"]
    for _ in range(1, max_len):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1), device=device)
        logits = decoder(seq, memory, tgt_mask=tgt_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
        if next_token.item() == end_idx:
            break
    words = [vocab.itos[w] for w in seq[0].tolist() if w not in (pad_idx, start_idx, end_idx)]
    return " ".join(words)

# ========== Streamlit UI ==========

st.set_page_config(page_title="Nepali Cultural Dress Captioning", layout="centered")
st.title("Nepali Cultural Dress Captioning")
st.markdown("Upload an image and select a model, then click **Generate Nepali Caption**.")

CLASS_LABELS = {
    "daura_suruwal": "Daura Suruwal",
    "gunyo_cholo": "Gunyo Cholo",
    "gurung_dress": "Gurung Dress",
    "haku_patasi": "Haku Patasi",
    "limbu_dress_mekhli_and_chaubandi": "Limbu Dress (Mekhli & Chaubandi)",
    "magar_dress": "Magar Dress",
    "sherpa_dress_chuba_bakkhu": "Sherpa Dress (Chuba Bakkhu)",
    "tamang_dress": "Tamang Dress",
    "tharu_dress": "Tharu Dress",
}

model_choice = st.selectbox("Select Model", [
    "ResNet-50 + GRU (1_out)",
    "Tiny ViT + Transformer (2_out)"
])

model_key = "resnet_gru" if "ResNet" in model_choice else "tinyvit"
model_path = os.path.join(BASE_DIR, "1_out" if "ResNet" in model_choice else "2_out")
features_path = os.path.join(BASE_DIR, f"features_{model_key}.pkl")
load_fn = load_resnet_gru if "ResNet" in model_choice else load_tinyvit

with st.spinner(f"Loading {model_choice}..."):
    try:
        encoder, decoder, vocab, eval_tf, config, device = load_fn(model_path)
        nn_model, classes_list, nepali_list = load_features(features_path)
        st.success(f"Device: {device} | Vocab: {len(vocab)}")
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    idx = predict_class(encoder, image, eval_tf, device, nn_model)
    predicted_class = classes_list[idx]
    display_class = CLASS_LABELS.get(predicted_class, predicted_class.replace("_", " ").title())
    st.subheader("Predicted Dress Class")
    st.success(f"**{display_class}**")

    if st.button("Generate Nepali Caption"):
        ne_caption = nepali_list[idx]
        st.subheader("Nepali Caption")
        st.info(ne_caption)
