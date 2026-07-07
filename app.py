import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os, re, pickle, random
from sklearn.neighbors import NearestNeighbors
import threading
from huggingface_hub import hf_hub_download
HF_MODEL_REPO = "dkodee/cultural-dress-models"

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
    ck = torch.load(model_path, map_location=device, weights_only=False)
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
    ck = torch.load(model_path, map_location=device, weights_only=False)
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
def ensure_downloaded(path, repo_filename):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {repo_filename}..."):
            hf_hub_download(HF_MODEL_REPO, repo_filename, local_dir=BASE_DIR, local_dir_use_symlinks=False)
    return path

@st.cache_resource
def load_features(features_path):
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    feat = np.array(data["features"])
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(feat)
    return nn_model, data["classes"], data["nepali_captions"]

def auto_generate_features(model_key, encoder, eval_tf, device):
    """Generate missing features file if dataset is available."""
    fpath = os.path.join(BASE_DIR, f"features_{model_key}.pkl")
    if os.path.exists(fpath):
        return True
    dataset_path = os.path.join(BASE_DIR, "nepali_dresses_augmented")
    csv_path = os.path.join(BASE_DIR, "captions.csv")
    if not os.path.isdir(dataset_path) or not os.path.exists(csv_path):
        return False
    import pandas as pd
    from tqdm import tqdm
    df = pd.read_csv(csv_path)
    encoder.eval()
    features, classes, caps = [], [], []
    paths = df["image_path"].tolist()
    st.info(f"Precomputing {model_key} features ({len(paths)} images)...")
    for p in tqdm(paths, desc=f"Extracting {model_key}"):
        full = os.path.join(BASE_DIR, p)
        if not os.path.exists(full):
            continue
        img = Image.open(full).convert("RGB")
        img_t = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = encoder(img_t).cpu().numpy().flatten()
        features.append(feat)
        classes.append(p.split("/")[-2])
        row = df[df["image_path"] == p].iloc[0]
        caps.append(row["caption_ne"])
    out = {"features": features, "classes": classes, "nepali_captions": caps}
    with open(fpath, "wb") as f:
        pickle.dump(out, f)
    return True

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

def expand_caption(base_caption, predicted_class, model_type, lang="en"):
    kb = NEPALI_KNOWLEDGE_BASE if lang == "ne" else KNOWLEDGE_BASE
    if predicted_class not in kb:
        return base_caption
    entry = kb[predicted_class]
    extra = []
    if model_type == "resnet_gru":
        for key in ("visual", "significance"):
            if key in entry:
                extra.append(entry[key])
    else:
        for key in ("history", "community"):
            if key in entry:
                extra.append(entry[key])
    if extra:
        return base_caption + " " + " ".join(extra)
    return base_caption

# ========== RAG Chat System ==========

@st.cache_resource(show_spinner="Loading RAG knowledge engine...")
def init_rag_engine():
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def build_rag_index(knowledge_base, embedder):
    chunks, chunk_dress, chunk_cat = [], [], []
    for dress, cats in knowledge_base.items():
        for cat, text in cats.items():
            chunks.append(text)
            chunk_dress.append(dress)
            chunk_cat.append(cat)
    if chunks:
        embs = embedder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    else:
        embs = None
    return chunks, chunk_dress, chunk_cat, embs

def retrieve_rag(query, embedder, chunks, chunk_dress, chunk_cat, embeddings, dress_filter=None, top_k=3):
    from sentence_transformers import util
    q_emb = embedder.encode([query], convert_to_tensor=True, show_progress_bar=False)
    scores = util.cos_sim(q_emb, embeddings)[0]
    if dress_filter:
        mask = torch.tensor([d == dress_filter for d in chunk_dress], dtype=torch.float, device=scores.device)
        scores = scores * mask
    top_idx = scores.argsort(descending=True)
    results = []
    for i in top_idx:
        if len(results) >= top_k:
            break
        if scores[i].item() > 0.15:
            results.append((chunks[i], chunk_dress[i], chunk_cat[i], scores[i].item()))
    return results

def detect_category(query):
    q = query.lower()
    for key, keywords in {
        "history": ["history", "origin", "start", "begin", "come from", "historical", "traditionally", "evolve"],
        "community": ["who wears", "community", "people", "ethnic", "group", "demographic"],
        "occasions": ["festival", "occasion", "wear", "when", "celebration", "event", "ceremony"],
        "visual": ["look", "appearance", "design", "color", "pattern", "style", "fabric", "visual"],
        "significance": ["significance", "meaning", "symbolize", "represent", "important", "value"]
    }.items():
        if any(kw in q for kw in keywords):
            return key
    return None

CATEGORY_LABELS = {
    "history": ("History", "📜"),
    "community": ("Community", "👥"),
    "occasions": ("When it's worn", "🎉"),
    "visual": ("Appearance", "👗"),
    "significance": ("Cultural Significance", "✨"),
}

BOT_EMOJIS = ["🧐", "✨", "🌟", "💫", "🎯", "🔥", "💪", "🎪", "🏔️", "🌸"]
GREETINGS = [
    "Namaste! I'm **DAJU**, your cultural dress guide! Ask me anything about this attire!",
    "Hey there! DAJU here — ready to explore Nepali culture with you! 🏔️",
    "Namaste! DAJU reporting for duty! What would you like to know about this beautiful dress?",
    "Welcome, culture explorer! I'm DAJU, your guide to Nepal's rich textile heritage!",
]

QUIPS = {
    "curious": ["Great question! You're really digging deep into Nepali culture!", "Ooh, nice curiosity!", "Love this question!", "Now we're talking!", "Excellent question!"],
    "celebrate": ["Knowledge +1 for you!", "You're becoming a true culture connoisseur!", "Nepali culture thanks you!", "Your cultural IQ just went up!", "Another mystery of Nepali fashion unveiled!"],
    "newbie": ["Let's start our journey! Every expert was once a beginner.", "First question — exciting! Let's dive in!", "The first step on your cultural adventure!"],
}

KNOWLEDGE_BASE = {
    "daura_suruwal": {
        "history": "Daura-Suruwal is the national dress of Nepal, historically worn by the Shah kings and nobility. It gained prominence during the 19th century under King Mahendra and was declared the national male attire in the 1960s. The design has origins in the traditional kurta worn across South Asia but was uniquely tailored with Nepali characteristics.",
        "community": "Daura-Suruwal is worn by Nepali men of all communities and ethnic backgrounds. It is especially prominent among the Khas and Bahun-Chhetri communities of the hilly region. Today, it is embraced by Nepali men across all castes and ethnicities as a symbol of national identity.",
        "occasions": "This attire is commonly worn during Dashain and Tihar festivals, wedding ceremonies (both as groom's attire and formal guest wear), government official functions, cultural programs, school graduation ceremonies, and formal political events. It is the go-to formal wear for Nepali men on all special occasions.",
        "visual": "The Daura is a closed-neck shirt with five pleats (kallyan) and eight strings (bandh) that tie the left side. It typically features a deep red or maroon color scheme with gold embroidery. The Suruwal is a tight-fitting trouser that is broad at the top and narrows at the ankles. A Patuka (cloth belt) wraps around the waist, and a vest or coat is often worn over it.",
        "significance": "Daura-Suruwal represents Nepali national identity, sovereignty, and cultural pride. The five pleats symbolize the Pancha Buddha or five elements. The eight strings represent the eight directions in Hindu cosmology. Wearing it signifies respect for Nepali tradition and is a statement of cultural heritage."
    },
    "gunyo_cholo": {
        "history": "Gunyo Cholo is the traditional attire for Nepali women, consisting of a blouse (Cholo) and a long wrapped skirt (Gunyo). It has been worn for centuries and is considered the female counterpart to the male Daura-Suruwal. The garment has evolved from simple rural wear to an important cultural symbol across Nepal.",
        "community": "Gunyo Cholo is worn by Nepali women across all ethnic groups and regions, including the Khas, Newar, and other communities. It is particularly significant as a coming-of-age garment for young girls. Women of all ages wear it, from young girls during ceremonies to elderly women on formal occasions.",
        "occasions": "It is worn during the Gunyo Cholo ceremony (a traditional coming-of-age ritual for girls), wedding ceremonies, Dashain and Tihar festivals, religious pujas, and formal cultural events. Many schools in Nepal require female students to wear it for special cultural programs and celebrations.",
        "visual": "The Gunyo is a long, ankle-length skirt made from richly colored fabric, often red, maroon, or green with gold or silver embroidery. The Cholo is a fitted blouse that covers the torso, often with intricate patterns. A shawl or scarf (Hijra) is commonly draped over the shoulders. Gold jewelry including earrings, necklaces, and bangles completes the ensemble.",
        "significance": "Gunyo Cholo symbolizes femininity, cultural identity, and the transition from childhood to womanhood in Nepali society. The Gunyo Cholo ceremony (also known as Bhaitika among some communities) is one of the most important rites of passage for Nepali girls, marking their entry into womanhood and their cultural education."
    },
    "gurung_dress": {
        "history": "The Gurung dress originates from the Gurung ethnic community, indigenous to the Annapurna and Lamjung regions of central Nepal. The Gurungs have a long history as warriors and traders, and their traditional dress reflects their proud martial heritage. The attire has been preserved for generations and is still handmade using traditional techniques.",
        "community": "The Gurung people primarily inhabit the Lamjung, Kaski, Gorkha, and Manang districts of Nepal. Gurung women are known for their weaving skills, and traditional Gurung dress is handwoven by community members. The dress is worn by both men and women of the Gurung ethnicity during important occasions.",
        "occasions": "Gurung dress is worn during Lhosar (the Gurung New Year), wedding ceremonies, cultural festivals, dance performances, and community gatherings. It is also worn during the annual Gurung festivals like Toh Lhosar and during cultural exchange programs showcasing Nepal's ethnic diversity.",
        "visual": "The Gurung women's dress consists of a long-sleeved blouse (Cholo) and a vibrant, striped ankle-length skirt (Gunyo), typically in red, black, and white geometric patterns. Women also wear a shawl and elaborate silver jewelry including the distinctive large circular earrings. Men wear a traditional shirt, trouser, and coat with a distinctive cap.",
        "significance": "Gurung dress is a powerful symbol of Gurung ethnic identity and cultural preservation. The distinctive striped patterns and color combinations are unique to Gurung weaving traditions. Wearing the dress honors Gurung ancestry and keeps alive the rich cultural heritage of this warrior community."
    },
    "haku_patasi": {
        "history": "Haku Patasi is a traditional black cotton saree worn by Newar women of the Kathmandu Valley, particularly among the Jyapu (farming) community. The name comes from Newari language: 'Haku' meaning black and 'Patasi' meaning saree. This attire has been worn for centuries and is one of the most recognizable garments of the Newar community.",
        "community": "Haku Patasi is exclusively associated with the Newar community of the Kathmandu Valley, particularly women from the Jyapu, Maharjan, and Shrestha sub-communities. It is an essential garment for Newar women across all social strata and is a key identifier of Newar cultural identity.",
        "occasions": "It is worn during major Newar festivals such as Bisket Jatra, Indra Jatra, and Gai Jatra, as well as during wedding ceremonies, religious rituals, and cultural events. Newar women traditionally wear Haku Patasi for their wedding and during important life cycle ceremonies.",
        "visual": "Haku Patasi is a pure black cotton saree with a distinctive red border (often with a thin yellow or green line). The fabric is typically coarse cotton, handwoven on traditional looms. Women pair it with a red blouse (Cholo) and a red shawl (Hijra). The contrast of black and red is striking and deeply symbolic.",
        "significance": "The black color symbolizes the protective goddess Kali, while the red border represents fertility and marital status. Haku Patasi is more than clothing - it is a symbol of Newar identity, resilience, and connection to ancestral traditions. Its continued use represents the preservation of Newar culture in the face of modernization."
    },
    "limbu_dress_mekhli_and_chaubandi": {
        "history": "The Limbu traditional attire, known as Mekhli (for women) and Chaubandi (for men), belongs to the Limbu ethnic community of eastern Nepal, particularly the Koshi Province. The Limbus are indigenous to the land between the Arun and Mechi rivers. Their traditional dress has remained largely unchanged for centuries, reflecting their strong cultural preservation.",
        "community": "The Limbu people (also known as Yakthung) inhabit the hilly regions of eastern Nepal, including Taplejung, Panchthar, Ilam, and Dhankuta districts. The dress is worn by both male and female members of the Limbu community. It is an important marker of Limbu ethnic identity.",
        "occasions": "Limbu dress is worn during Chasok Tangnam (the Limbu harvest festival), wedding ceremonies, birth celebrations, cultural performances, and community gatherings. It is also worn during political and social events where Limbu representation is significant. The dress is a must for all traditional Limbu rituals.",
        "visual": "The women's Mekhli is a two-piece wrap-around skirt, typically in dark blue, black, or green with intricate geometric patterns. The Chaubandi for men is a long-sleeved shirt with a distinctive closed neck with four ties, paired with loose trousers. Both men and women wear a shawl. Silver coins and elaborate jewelry are essential accessories.",
        "significance": "Limbu dress represents the rich cultural heritage of the Kirati people and their connection to their ancestral lands. The geometric patterns on the Mekhli are said to represent the Limbu script and ancient symbols. Wearing the dress is an act of cultural assertion and pride for the Limbu community."
    },
    "magar_dress": {
        "history": "The Magar traditional dress originates from the Magar community, one of the largest ethnic groups in Nepal, primarily inhabiting the mid-western hills. The Magars have a long martial history and were renowned as fierce warriors alongside the British Gurkha regiments. Their traditional dress reflects their rugged lifestyle and cultural identity.",
        "community": "Magar people are concentrated in Gulmi, Palpa, Pyuthan, Rolpa, and other districts of western Nepal. The dress is worn by both men and women of the Magar community. Among the Magar sub-groups (including the Ale, Pun, Rana, and Thapa), the dress styles have subtle variations.",
        "occasions": "Magar dress is traditionally worn during Maghe Sankranti, wedding ceremonies, birth celebrations, and community festivals. It is also worn during cultural programs and by Magar representatives at national events. The dress is especially important during the annual Magar cultural festivals.",
        "visual": "Magar women wear a long-sleeved blouse (Chaubandi Cholo) with a full-length skirt (Fariya) in bold colors, primarily red, black, and green with distinctive horizontal or vertical stripes. They wear heavy silver jewelry including necklaces, bangles, and anklets. Men wear a traditional vest (Bhoto) over a white shirt with loose trousers and a white cap.",
        "significance": "Magar dress symbolizes the community's ethnic pride, warrior heritage, and connection to their ancestral lands. The distinctive patterns and jewelry designs have been passed down through generations. The dress serves as an important cultural identifier and a way of preserving Magar traditions in the modern era."
    },
    "sherpa_dress_chuba_bakkhu": {
        "history": "The Sherpa traditional dress, Chuba (for women) and Bakkhu (for men), originates from the Sherpa community of the Solukhumbu region, the homeland of Mount Everest. The Sherpas are renowned worldwide as mountaineers and guides. Their traditional dress is well-adapted to the cold Himalayan climate and has Tibetan cultural influences.",
        "community": "The Sherpa community primarily lives in the high-altitude regions of Solukhumbu, Sankhuwasabha, and Dolakha districts. Sherpa dress is unique to this community and reflects their Tibetan Buddhist cultural heritage. Both men and women of the Sherpa community wear variations of this attire.",
        "occasions": "Chuba and Bakkhu are worn during Loshar (Tibetan New Year), Dumje festival, Mani Rimdu festival, wedding ceremonies, and religious ceremonies at Buddhist monasteries. The dress is also worn for cultural performances and when welcoming visitors to the Everest region.",
        "visual": "The women's Chuba is a long, thick woolen dress (Tibetan-style) worn over a blouse, tied with a colorful sash at the waist. It often features elaborate geometric patterns and vibrant colors. The men's Bakkhu is a similar long robe tied with a sash. Both wear the traditional Tengma (soft boots) and a variety of Tibetan Buddhist-influenced jewelry including coral and turquoise beads.",
        "significance": "Sherpa dress represents the community's adaptation to high-altitude living and their Tibetan Buddhist heritage. The traditional woolen garments are practical for cold weather while being culturally significant. Wearing Chuba and Bakkhu connects Sherpa people to their mountaineering legacy and spiritual traditions."
    },
    "tamang_dress": {
        "history": "Tamang traditional dress belongs to the Tamang community, one of the largest ethnic groups in Nepal, primarily living in the hills surrounding the Kathmandu Valley. The Tamangs are of Tibeto-Burman origin and their traditional dress reflects this heritage. The dress has been preserved through centuries and is an important part of Tamang cultural identity.",
        "community": "Tamang people inhabit the Rasuwa, Nuwakot, Dhading, Sindhupalchok, and Kavrepalanchok districts surrounding the Kathmandu Valley. The dress is worn by Tamang men and women across all age groups. It is particularly well-preserved among rural Tamang communities.",
        "occasions": "Tamang dress is worn during Sonam Lhosar (Tamang New Year), wedding ceremonies, birth celebrations, and cultural festivals. It is also worn during religious ceremonies at Buddhist gumbas and during community gatherings. The dress is an essential part of Tamang cultural identity and is worn with pride during all important events.",
        "visual": "Tamang women wear a long dark-colored skirt (Gunyo) with a velvet or cotton blouse, often in black or deep red. They wear distinctive heavy silver jewelry including large necklace pieces (Potey) and multiple bangles. The most recognizable feature is the Tamang headdress (Gagar) worn by women on special occasions. Men wear traditional shirt, vest, and trousers with a white or black cap.",
        "significance": "Tamang dress symbolizes the community's Tibeto-Burman roots and their distinct cultural identity within Nepal's diverse ethnic landscape. The jewelry, particularly the Potey necklace made of silver and precious stones, is considered a marker of Tamang identity and is passed down through generations as family heirlooms."
    },
    "tharu_dress": {
        "history": "Tharu traditional dress belongs to the Tharu community, the indigenous people of the Terai region of southern Nepal. The Tharus have lived in the inner Terai for centuries and have developed a unique culture adapted to the tropical climate and forest environment. Their traditional dress is colorful, lightweight, and practical for the warm Terai climate.",
        "community": "The Tharu community is spread across the Terai belt from western to eastern Nepal, including Dang, Chitwan, Bardiya, Kailali, and Jhapa districts. The dress varies slightly among Tharu subgroups including the Rana Tharu, Dangaura Tharu, and Kathariya Tharu. It is worn by both men and women of the community.",
        "occasions": "Tharu dress is traditionally worn during Maghi (Tharu New Year), wedding ceremonies, harvest festivals, and cultural performances. It is also worn during the annual Tharu cultural festivals and community celebrations. The dress is especially vibrant during the Jitiya festival celebrated by Tharu women.",
        "visual": "Tharu women wear a short-sleeved blouse (Cholo) and a full-length cotton skirt (Lahanga) in bright, vibrant colors with mirror work and intricate embroidery. They are famous for their heavy silver jewelry, including the distinctive large nose ring (Bula), thick silver necklace (Har), and anklets that cover the entire lower leg. Men wear a simple dhoti and shirt with a traditional vest.",
        "significance": "Tharu dress represents the community's deep connection to the Terai land and forest. The distinctive jewelry, particularly the heavy silver ornaments, is not just decorative but serves as a form of savings and wealth for Tharu women. The bright colors and mirror work reflect the Tharu's joyful spirit and their celebration of nature and life."
    }
}

NEPALI_KNOWLEDGE_BASE = {
    "daura_suruwal": {
        "history": "दौरा-सुरुवाल नेपालको राष्ट्रिय पोशाक हो, जुन ऐतिहासिक रूपमा शाह राजाहरू र कुलीन वर्गले लगाउने गर्थे। यो १९औं शताब्दीमा राजा महेन्द्रको पालामा प्रसिद्ध भयो र १९६० को दशकमा राष्ट्रिय पुरुष पोशाक घोषित गरियो। यो दक्षिण एसियाभर लगाइने परम्परागत कुर्ताको डिजाइनमा आधारित छ तर नेपाली विशेषताहरूसहित अद्वितीय रूपमा तयार गरिएको छ।",
        "community": "दौरा-सुरुवाल सबै समुदाय र जातीय पृष्ठभूमिका नेपाली पुरुषहरूले लगाउँछन्। यो विशेष गरी पहाडी क्षेत्रका खस र बाहुन-क्षेत्री समुदायमा प्रमुख छ। आज, यो राष्ट्रिय पहिचानको प्रतीकको रूपमा सबै जाति र जातीयताका नेपाली पुरुषहरूले अँगालेका छन्।",
        "visual": "दौरा एक बन्द-घाँटीको कमिज हो जसमा पाँच कल्यान र आठ बन्ध हुन्छन् जुन देब्रेपट्टि बाँधिन्छन्। यो सामान्यतया गहिरो रातो वा मरुन रंगको हुन्छ र सुनको कढाईले सजिएको हुन्छ। सुरुवाल एक फितलो पाइन्ट हो जो माथि फराकिलो र खुट्टामा साँघुरो हुन्छ। पटुकाले कम्मर वरिपरि बेरिन्छ र प्रायः भेस्ट वा कोट लगाइन्छ।",
        "significance": "दौरा-सुरुवालले नेपाली राष्ट्रिय पहिचान, सार्वभौमिकता र सांस्कृतिक गौरवलाई प्रतिनिधित्व गर्दछ। पाँच कल्यानले पञ्चबुद्ध वा पाँच तत्वलाई प्रतीक गर्दछ। आठ बन्धले हिन्दू ब्रह्माण्डविज्ञानमा आठ दिशाहरूलाई प्रतिनिधित्व गर्दछ। यसलाई लगाउनु नेपाली परम्पराप्रति सम्मान र सांस्कृतिक सम्पदाको अभिव्यक्ति हो।"
    },
    "gunyo_cholo": {
        "history": "गुन्यू-चोलो नेपाली महिलाको परम्परागत पोशाक हो, जसमा चोलो (ब्लाउज) र गुन्यू (लामो बेरिएको स्कर्ट) हुन्छ। यो शताब्दीयौंदेखि लगाइँदै आएको छ र पुरुषको दौरा-सुरुवालको महिला संस्करण मानिन्छ। यो पोशाक साधारण ग्रामीण पहिरनबाट नेपालभरिको महत्त्वपूर्ण सांस्कृतिक प्रतीकमा विकसित भएको छ।",
        "community": "गुन्यू-चोलो खस, नेवार र अन्य जातीय समूहहरू सहित सबै जातीय समूह र क्षेत्रका नेपाली महिलाहरूले लगाउँछन्। यो विशेष गरी युवतीहरूको लागि वयस्कता आउने पोशाकको रूपमा महत्त्वपूर्ण छ। सबै उमेरका महिलाहरू यो लगाउँछन् - साना केटीहरूदेखि समारोहमा देखि वृद्ध महिलाहरू औपचारिक अवसरहरूमा।",
        "visual": "गुन्यू एक लामो, खुट्टासम्मको स्कर्ट हो जुन धनी रंगको कपडाबाट बनेको हुन्छ, प्रायः रातो, मरुन वा हरियो सुन वा चाँदीको कढाईसहित। चोलो एक फिट गरिएको ब्लाउज हो जुन शरीरलाई ढाक्छ, प्रायः जटिल ढाँचासहित। हिज्रा (सल) सामान्यतया काँधमा बेरिन्छ। सुनका गरगहना जस्तै मुन्द्रा, माला र चुराले पोशाक पूरा हुन्छ।",
        "significance": "गुन्यू-चोलोले नेपाली समाजमा नारीत्व, सांस्कृतिक पहिचान र बाल्यकालबाट महिलात्वमा संक्रमणलाई प्रतिनिधित्व गर्दछ। गुन्यू-चोलो समारोह नेपाली केटीहरूको लागि सबैभन्दा महत्त्वपूर्ण संस्कार हो, जसले महिलात्वमा उनीहरूको प्रवेश र सांस्कृतिक शिक्षालाई चिन्ह लगाउँछ।"
    },
    "gurung_dress": {
        "history": "गुरुङ पोशाक गुरुङ जातीय समुदायबाट उत्पत्ति भएको हो, जो मध्य नेपालको अन्नपूर्ण र लमजुङ क्षेत्रका मूल निवासी हुन्। गुरुङहरूको योद्धा र व्यापारीको रूपमा लामो इतिहास छ, र उनीहरूको परम्परागत पोशाकले उनीहरूको गौरवपूर्ण योद्धा सम्पदा झल्काउँछ। यो पोशाक पुस्तौंदेखि संरक्षित गरिएको छ र अझै पनि परम्परागत प्रविधिहरू प्रयोग गरेर हातले बनाइन्छ।",
        "community": "गुरुङ मानिसहरू मुख्यतया नेपालको लमजुङ, कास्की, गोरखा र मनाङ जिल्लाहरूमा बस्छन्। गुरुङ महिलाहरू आफ्नो बुनाई सीपको लागि परिचित छन्, र परम्परागत गुरुङ पोशाक समुदायका सदस्यहरूद्वारा हातले बनाइन्छ। यो पोशाक महत्त्वपूर्ण अवसरहरूमा गुरुङ जातिका पुरुष र महिला दुवैले लगाउँछन्।",
        "visual": "गुरुङ महिलाको पोशाकमा लामो बाहुला भएको चोलो र जीवन्त, धर्सादार खुट्टासम्मको गुन्यू हुन्छ, सामान्यतया रातो, कालो र सेतो ज्यामितीय ढाँचामा। महिलाहरू सल र विशिष्ट ठूला गोलाकार मुन्द्रा सहितको चाँदीका गरगहना लगाउँछन्। पुरुषहरूले परम्परागत कमिज, पाइन्ट र कोट विशिष्ट टोपीसहित लगाउँछन्।",
        "significance": "गुरुङ पोशाक गुरुङ जातीय पहिचान र सांस्कृतिक संरक्षणको शक्तिशाली प्रतीक हो। विशिष्ट धर्सादार ढाँचा र रंग संयोजन गुरुङ बुनाई परम्पराका लागि अद्वितीय छन्। यो पोशाक लगाउनु गुरुङ पुर्खाहरूको सम्मान र यो योद्धा समुदायको धनी सांस्कृतिक सम्पदा जीवित राख्नु हो।"
    },
    "haku_patasi": {
        "history": "हाकु पतासी काठमाडौं उपत्यकाका नेवार महिलाहरू, विशेष गरी ज्यापु समुदायमा लगाइने परम्परागत कालो सुती सारी हो। नाम नेवारी भाषाबाट आएको हो: 'हाकु' अर्थ कालो र 'पतासी' अर्थ सारी। यो पोशाक शताब्दीयौंदेखि लगाइँदै आएको छ र नेवार समुदायको सबैभन्दा चिन्न सकिने पोशाक हो।",
        "community": "हाकु पतासी विशेष रूपमा काठमाडौं उपत्यकाको नेवार समुदाय, विशेष गरी ज्यापु, महर्जन र श्रेष्ठ उप-समुदायका महिलाहरूसँग सम्बन्धित छ। यो सबै सामाजिक स्तरका नेवार महिलाहरूको लागि आवश्यक पोशाक हो र नेवार सांस्कृतिक पहिचानको मुख्य पहिचानकर्ता हो।",
        "visual": "हाकु पतासी एक शुद्ध कालो सुती सारी हो जसमा विशिष्ट रातो किनारा हुन्छ। कपडा सामान्यतया मोटो सुती हुन्छ, परम्परागत करघामा हातले बनाइएको। महिलाहरू यसलाई रातो चोलो र रातो हिज्रासँग जोडेर लगाउँछन्। कालो र रातोको विरोधाभास आकर्षक र गहिरो रूपमा प्रतीकात्मक छ।",
        "significance": "कालो रंगले रक्षक देवी कालीलाई प्रतिनिधित्व गर्दछ, जबकि रातो किनाराले उर्वरता र वैवाहिक स्थिति जनाउँदछ। हाकु पतासी केवल कपडा भन्दा बढी हो - यो नेवार पहिचान, लचिलोपन र पुर्खाको परम्परासँग जोडिएको प्रतीक हो। यसको निरन्तर प्रयोगले आधुनिकीकरणको सामना गर्दै नेवार संस्कृतिको संरक्षणलाई प्रतिनिधित्व गर्दछ।"
    },
    "limbu_dress_mekhli_and_chaubandi": {
        "history": "लिम्बू परम्परागत पोशाक, महिलाको लागि मेख्ली र पुरुषको लागि चौबन्धी, पूर्वी नेपालको लिम्बू जातीय समुदायसँग सम्बन्धित छ। लिम्बूहरू अरुण र मेची नदीहरूको बीचको भूमिका मूल निवासी हुन्। उनीहरूको परम्परागत पोशाक शताब्दीयौंदेखि अपरिवर्तित रहेको छ, जसले उनीहरूको बलियो सांस्कृतिक संरक्षणलाई प्रतिबिम्बित गर्दछ।",
        "community": "लिम्बू मानिसहरू पूर्वी नेपालको ताप्लेजुङ, पाँचथर, इलाम र धनकुटा जिल्लाहरूमा बस्छन्। यो पोशाक लिम्बू समुदायका पुरुष र महिला दुवैले लगाउँछन्। यो लिम्बू जातीय पहिचानको महत्त्वपूर्ण चिन्हकर्ता हो।",
        "visual": "महिलाको मेख्ली दुई टुक्राको बेरिएको स्कर्ट हो, सामान्यतया गाढा नीलो, कालो वा हरियो जटिल ज्यामितीय ढाँचासहित। पुरुषको चौबन्धी लामो बाहुला भएको कमिज हो जसमा चार बन्धसहित विशिष्ट बन्द घाँटी हुन्छ, फितलो पाइन्टसँग जोडिएको। चाँदीका सिक्का र विस्तृत गरगहना आवश्यक सहायक वस्तु हुन्।",
        "significance": "लिम्बू पोशाकले किराँती मानिसहरूको धनी सांस्कृतिक सम्पदा र उनीहरूको पुर्खाको भूमिसँग सम्बन्धलाई प्रतिनिधित्व गर्दछ। मेख्लीमा ज्यामितीय ढाँचाले लिम्बू लिपि र प्राचीन प्रतीकहरू प्रतिनिधित्व गर्ने भनिन्छ। यो पोशाक लगाउनु लिम्बू समुदायको लागि सांस्कृतिक अभिव्यक्ति र गौरवको कार्य हो।"
    },
    "magar_dress": {
        "history": "मगर परम्परागत पोशाक नेपालको सबैभन्दा ठूलो जातीय समूह मध्ये एक मगर समुदायबाट उत्पत्ति भएको हो, जो मुख्यतया मध्य-पश्चिमी पहाडहरूमा बस्छन्। मगरहरूको लामो युद्ध इतिहास छ र ब्रिटिश गोर्खा रेजिमेन्टसँगै भयंकर योद्धाको रूपमा प्रसिद्ध थिए। उनीहरूको परम्परागत पोशाकले उनीहरूको कठोर जीवनशैली र सांस्कृतिक पहिचान झल्काउँछ।",
        "community": "मगर मानिसहरू गुल्मी, पाल्पा, प्युठान, रोल्पा र अन्य पश्चिमी नेपालका जिल्लाहरूमा केन्द्रित छन्। यो पोशाक मगर समुदायका पुरुष र महिला दुवैले लगाउँछन्। मगर उप-समूहहरूमा पोशाक शैलीमा सूक्ष्म भिन्नताहरू छन्।",
        "visual": "मगर महिलाहरूले लामो बाहुलाको चौबन्धी चोलो सँग पूर्ण-लम्बाइको फरिया लगाउँछन्, मुख्य रूपमा रातो, कालो र हरियो विशिष्ट तेर्सो वा ठाडो धर्सासहित। तिनीहरू चाँदीका गहना जस्तै माला, चुरा र पाउँजेली लगाउँछन्। पुरुषहरूले सेतो कमिजमाथि परम्परागत भोटो, फितलो पाइन्ट र सेतो टोपी लगाउँछन्।",
        "significance": "मगर पोशाकले समुदायको जातीय गौरव, योद्धा सम्पदा र पुर्खाको भूमिसँग जोडिएको प्रतीक हो। विशिष्ट ढाँचा र गरगहना डिजाइनहरू पुस्तौंदेखि हस्तान्तरण गरिएका छन्। यो पोशाकले आधुनिक युगमा मगर परम्पराहरू संरक्षण गर्ने महत्त्वपूर्ण सांस्कृतिक पहिचानकर्ता र माध्यमको रूपमा काम गर्दछ।"
    },
    "sherpa_dress_chuba_bakkhu": {
        "history": "शेर्पा परम्परागत पोशाक, महिलाको लागि चुबा र पुरुषको लागि बक्खु, सगरमाथाको घर सोलुखुम्बु क्षेत्रको शेर्पा समुदायबाट उत्पत्ति भएको हो। शेर्पाहरू विश्वभर पर्वतारोही र गाइडको रूपमा प्रसिद्ध छन्। उनीहरूको परम्परागत पोशाक चिसो हिमाली मौसममा राम्रोसँग अनुकूलित छ र तिब्बती सांस्कृतिक प्रभावहरू छन्।",
        "community": "शेर्पा समुदाय मुख्यतया सोलुखुम्बु, सिन्धुपाल्चोक र दोलखाका उच्च-उचाइ क्षेत्रहरूमा बस्छ। शेर्पा पोशाक यस समुदायको लागि अद्वितीय छ र उनीहरूको तिब्बती बौद्ध सांस्कृतिक सम्पदा झल्काउँछ। शेर्पा समुदायका पुरुष र महिला दुवै यस पोशाकका भिन्नताहरू लगाउँछन्।",
        "visual": "महिलाको चुबा एक लामो, बाक्लो ऊनी पोशाक हो जुन ब्लाउजमाथि लगाइन्छ र कम्मरमा रंगीन स्याँठले बाँधिन्छ। यो प्रायः जटिल ज्यामितीय ढाँचा र जीवन्त रंगहरूले सजिएको हुन्छ। पुरुषको बक्खु पनि स्याँठले बाँधिएको लामो लबाडा हो। दुवैले मूंगा र फिरोजा मोती सहित तिब्बती बौद्ध-प्रभावित गरगहना लगाउँछन्।",
        "significance": "शेर्पा पोशाकले उच्च-उचाइ जीवनमा समुदायको अनुकूलन र उनीहरूको तिब्बती बौद्ध सम्पदालाई प्रतिनिधित्व गर्दछ। परम्परागत ऊनी वस्त्र चिसो मौसमको लागि व्यावहारिक हुँदै सांस्कृतिक रूपमा महत्त्वपूर्ण छन्। चुबा र बक्खु लगाउनुले शेर्पा मानिसहरूलाई उनीहरूको पर्वतारोहण सम्पदा र आध्यात्मिक परम्परासँग जोड्दछ।"
    },
    "tamang_dress": {
        "history": "तामाङ परम्परागत पोशाक नेपालको सबैभन्दा ठूलो जातीय समूह मध्ये एक तामाङ समुदायको हो, जो मुख्यतया काठमाडौं उपत्यका वरपरका पहाडहरूमा बस्छन्। तामाङहरू तिब्बती-बर्मी मूलका हुन् र उनीहरूको परम्परागत पोशाकले यो सम्पदा झल्काउँछ। यो पोशाक शताब्दीयौंदेखि संरक्षित गरिएको छ र तामाङ सांस्कृतिक पहिचानको महत्त्वपूर्ण भाग हो।",
        "community": "तामाङ मानिसहरू काठमाडौं उपत्यका वरपर रसुवा, नुवाकोट, धादिङ, सिन्धुपाल्चोक र काभ्रेपलाञ्चोक जिल्लाहरूमा बस्छन्। यो पोशाक सबै उमेर समूहका तामाङ पुरुष र महिलाले लगाउँछन्। यो ग्रामीण तामाङ समुदायहरूमा विशेष रूपमा राम्रोसँग संरक्षित गरिएको छ।",
        "visual": "तामाङ महिलाहरूले लामो गाढा रंगको गुन्यू मखमली वा सुती चोलोसँग लगाउँछन्, प्रायः कालो वा गाढा रातो। तिनीहरू विशिष्ट चाँदीका गरगहना जस्तै ठूलो पोते र धेरै चुरा लगाउँछन्। सबैभन्दा चिन्न सकिने विशेषता विशेष अवसरमा महिलाहरूले लगाउने तामाङ शिरवस्त्र गागर हो।",
        "significance": "तामाङ पोशाकले समुदायको तिब्बती-बर्मी जरा र नेपालको विविध जातीय परिदृश्य भित्र उनीहरूको विशिष्ट सांस्कृतिक पहिचानलाई प्रतीक गर्दछ। चाँदी र बहुमूल्य पत्थरले बनेको पोते माला तामाङ पहिचानको चिन्हकर्ता मानिन्छ र पारिवारिक सम्पत्तिको रूपमा पुस्तौं हस्तान्तरण गरिन्छ।"
    },
    "tharu_dress": {
        "history": "थारू परम्परागत पोशाक नेपालको दक्षिणी तराई क्षेत्रका आदिवासी थारू समुदायको हो। थारूहरू शताब्दीयौंदेखि भित्री तराईमा बसोबास गर्दै आएका छन् र उष्णकटिबंधीय मौसम र वन वातावरणमा अनुकूलित अद्वितीय संस्कृति विकसित गरेका छन्। उनीहरूको परम्परागत पोशाक रंगीन, हल्का र न्यानो तराई मौसमको लागि व्यावहारिक छ।",
        "community": "थारू समुदाय पश्चिमी नेपालदेखि पूर्वी नेपालसम्म तराई पेटीमा फैलिएको छ, जसमा दाङ, चितवन, बर्दिया, कैलाली र झापा जिल्लाहरू समावेश छन्। थारू उप-समूहहरूमा पोशाक थोरै फरक हुन्छ। यो समुदायका पुरुष र महिला दुवैले लगाउँछन्।",
        "visual": "थारू महिलाहरूले छोटो बाहुलाको चोलो र पूर्ण-लम्बाइको लहंगा चम्किलो, जीवन्त रंगहरूमा ऐना र जटिल कढाईसहित लगाउँछन्। तिनीहरू चाँदीका गरगहना जस्तै विशिष्ट ठूलो नाकको बुला, बाक्लो चाँदीको हार र पूरै खुट्टा ढाक्ने पाउँजेलीको लागि प्रसिद्ध छन्। पुरुषहरूले साधारण धोती र कमिज परम्परागत भोटोसहित लगाउँछन्।",
        "significance": "थारू पोशाकले तराई भूमि र वनसँग समुदायको गहिरो सम्बन्धलाई प्रतिनिधित्व गर्दछ। विशिष्ट गरगहना, विशेष गरी भारी चाँदीका आभूषण, सजावटी मात्र नभई थारू महिलाको लागि बचत र धनको रूपमा पनि काम गर्दछ। चम्किलो रंग र ऐनाको कामले थारूको आनन्दित आत्मा र प्रकृति र जीवनको उत्सवलाई प्रतिबिम्बित गर्दछ।"
    }
}

# ========== Streamlit UI ==========

st.set_page_config(page_title="Nepali Cultural Dress Captioning", layout="centered")

st.markdown("""
<style>
.bot-avatar { font-size: 2rem; }
.user-avatar { font-size: 1.5rem; }

.quick-chip {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.3);
    padding: 6px 14px;
    border-radius: 20px;
    margin: 3px;
    cursor: pointer;
    font-size: 0.8rem;
    transition: all 0.2s;
}
.quick-chip:hover { background: rgba(255,255,255,0.2); }
.progress-text { font-size: 0.8rem; color: #aaa; }
div[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: rgba(77, 150, 255, 0.1) !important;
    border: 1px solid rgba(77, 150, 255, 0.2) !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(107, 203, 119, 0.08) !important;
    border: 1px solid rgba(107, 203, 119, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏔️ Nepali Cultural Dress Explorer")
st.markdown("Upload a dress image, get a Nepali caption, then **chat with DAJU** — your cultural guide!")

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
    "ResNet-50 + GRU",
    "Tiny ViT + Transformer"
])

model_key = "resnet_gru" if "ResNet" in model_choice else "tinyvit"
model_path = os.path.join(BASE_DIR, "1_out" if "ResNet" in model_choice else "2_out")
features_path = os.path.join(BASE_DIR, f"features_{model_key}.pkl")
load_fn = load_resnet_gru if "ResNet" in model_choice else load_tinyvit

with st.spinner(f"Loading {model_choice}..."):
    try:
        encoder, decoder, vocab, eval_tf, config, device = load_fn(model_path)
        if not os.path.exists(features_path):
            if model_key == "resnet_gru":
                ensure_downloaded(features_path, "features_resnet_gru.pkl")
            if not os.path.exists(features_path) and auto_generate_features(model_key, encoder, eval_tf, device):
                st.rerun()
            if not os.path.exists(features_path):
                st.error(f"Features file for {model_choice} is missing. Run: python precompute_features.py")
                st.stop()
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

    model_key = "resnet_gru" if "ResNet" in model_choice else "tinyvit"

    st.subheader("Nepali Caption")
    ne_expanded = expand_caption(nepali_list[idx], predicted_class, model_key, lang="ne")
    st.info(ne_expanded)

    # ========== RAG Chat Support ==========

    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False

    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.chat_open:
            if st.button("🎭 Chat with **DAJU**", use_container_width=True, type="primary"):
                st.session_state.chat_open = True
                st.rerun()
        else:
            if st.button("✕ Close Chat", use_container_width=True):
                st.session_state.chat_open = False
                st.rerun()

    if st.session_state.chat_open:
        st.markdown(f"### 🎮 Chat with **DAJU** — Your Cultural Dress Guide")

        # Init RAG engine
        if "rag_initialized" not in st.session_state:
            with st.spinner("Loading RAG knowledge engine..."):
                embedder = init_rag_engine()
                chunks, chunk_dress, chunk_cat, embeddings = build_rag_index(KNOWLEDGE_BASE, embedder)
            st.session_state.rag_embedder = embedder
            st.session_state.rag_chunks = chunks
            st.session_state.rag_chunk_dress = chunk_dress
            st.session_state.rag_chunk_cat = chunk_cat
            st.session_state.rag_embeddings = embeddings
            st.session_state.rag_initialized = True

        # Session state initialisation
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "context_history" not in st.session_state:
            st.session_state.context_history = []


        # Quick action chips
        action_topics = ["History", "Who wears it", "Festivals", "Appearance", "Significance", "Tell me everything"]
        topic_keys_map = {"History": "history", "Who wears it": "community", "Festivals": "occasions",
                          "Appearance": "visual", "Significance": "significance", "Tell me everything": "all"}

        st.markdown("**Quick explore:**")
        chip_cols = st.columns(6)
        for i, topic in enumerate(action_topics):
            if chip_cols[i % 6].button(topic, key=f"chip_{topic}", use_container_width=True):
                question_map = {
                    "History": f"Tell me about the history of {display_class}",
                    "Who wears it": f"Who wears {display_class}?",
                    "Festivals": f"When and for what occasions is {display_class} worn?",
                    "Appearance": f"What does {display_class} look like?",
                    "Significance": f"What is the cultural significance of {display_class}?",
                    "Tell me everything": f"Tell me everything about {display_class}"
                }
                prompt = question_map[topic]

                st.session_state.chat_history.append({"role": "user", "content": prompt})

                query = prompt
                dress_key = predicted_class
                context_history = st.session_state.context_history
                retrieved = retrieve_rag(
                    query,
                    st.session_state.rag_embedder,
                    st.session_state.rag_chunks,
                    st.session_state.rag_chunk_dress,
                    st.session_state.rag_chunk_cat,
                    st.session_state.rag_embeddings,
                    dress_filter=dress_key,
                    top_k=3
                )

                if len(retrieved) < 2:
                    retrieved = retrieve_rag(
                        query,
                        st.session_state.rag_embedder,
                        st.session_state.rag_chunks,
                        st.session_state.rag_chunk_dress,
                        st.session_state.rag_chunk_cat,
                        st.session_state.rag_embeddings,
                        dress_filter=dress_key,
                        top_k=5
                    )[:3]

                if not retrieved:
                    cat_idx = list(KNOWLEDGE_BASE[dress_key].keys())
                    random.shuffle(cat_idx)
                    for c in cat_idx[:3]:
                        retrieved.append((KNOWLEDGE_BASE[dress_key][c], dress_key, c, 0.5))

                # Build response
                wanted_cat = topic_keys_map.get(topic)
                response_parts = []
                seen_cats = set()
                for chunk, src, cat, score in retrieved:
                    if wanted_cat != "all" and cat != wanted_cat:
                        continue
                    seen_cats.add(cat)
                    label, emoji = CATEGORY_LABELS.get(cat, (cat.capitalize(), "📌"))
                    response_parts.append(f"{emoji} **{label}:**\n{chunk}")

                if not response_parts and wanted_cat and wanted_cat != "all":
                    chunk = KNOWLEDGE_BASE[dress_key][wanted_cat]
                    label, emoji = CATEGORY_LABELS.get(wanted_cat, (wanted_cat.capitalize(), "📌"))
                    response_parts.append(f"{emoji} **{label}:**\n{chunk}")

                response = (
                    random.choice(QUIPS["curious"]) + "\n\n"
                    + "\n\n".join(response_parts) + "\n\n"
                    + random.choice(QUIPS["celebrate"])
                )

                ctx = {"question": query, "categories": list(seen_cats)[:3], "dress": dress_key}

                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.session_state.context_history.append(ctx)
                st.rerun()

        # Chat container
        with st.container(height=400):
            if not st.session_state.chat_history:
                greeting = random.choice(GREETINGS)
                st.markdown(f"🤖 **DAJU:** {greeting}")
                st.markdown("""
                <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:12px; border:1px dashed rgba(255,255,255,0.2); margin-top:8px;">
                <span style="font-size:0.85rem; color:#aaa;">💡 <b>Try asking:</b> "What's the history?", "Who wears it?", "What does it look like?", "Cultural significance?"</span>
                </div>
                """, unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎭"):
                    st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input(f"Ask DAJU about {display_class}..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🎭"):
                with st.spinner("DAJU is thinking..."):
                    query = prompt
                    dress_key = predicted_class

                    retrieved = retrieve_rag(
                        query,
                        st.session_state.rag_embedder,
                        st.session_state.rag_chunks,
                        st.session_state.rag_chunk_dress,
                        st.session_state.rag_chunk_cat,
                        st.session_state.rag_embeddings,
                        dress_filter=dress_key,
                        top_k=3
                    )

                    if len(retrieved) < 2:
                        retrieved = retrieve_rag(
                            query,
                            st.session_state.rag_embedder,
                            st.session_state.rag_chunks,
                            st.session_state.rag_chunk_dress,
                            st.session_state.rag_chunk_cat,
                            st.session_state.rag_embeddings,
                            dress_filter=dress_key,
                            top_k=5
                        )[:3]

                    if not retrieved:
                        cat_idx = list(KNOWLEDGE_BASE[dress_key].keys())
                        random.shuffle(cat_idx)
                        for c in cat_idx[:3]:
                            retrieved.append((KNOWLEDGE_BASE[dress_key][c], dress_key, c, 0.5))

                    wanted_cat = detect_category(query)
                    response_parts = []
                    seen_cats = set()
                    for chunk, src, cat, score in retrieved:
                        if wanted_cat and cat != wanted_cat:
                            continue
                        seen_cats.add(cat)
                        label, emoji = CATEGORY_LABELS.get(cat, (cat.capitalize(), "📌"))
                        response_parts.append(f"{emoji} **{label}:**\n{chunk}")

                    if not response_parts and wanted_cat:
                        chunk = KNOWLEDGE_BASE[dress_key][wanted_cat]
                        label, emoji = CATEGORY_LABELS.get(wanted_cat, (wanted_cat.capitalize(), "📌"))
                        response_parts.append(f"{emoji} **{label}:**\n{chunk}")

                    response = (
                        random.choice(QUIPS["curious"]) + "\n\n"
                        + "\n\n".join(response_parts) + "\n\n"
                        + random.choice(QUIPS["celebrate"])
                    )

                    ctx = {"question": query, "categories": list(seen_cats) if wanted_cat else ["all"], "dress": dress_key}

                    st.markdown(response)
                    st.session_state.context_history.append(ctx)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
