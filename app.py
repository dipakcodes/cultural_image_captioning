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

def expand_caption(base_caption, predicted_class, model_type):
    if predicted_class not in KNOWLEDGE_BASE:
        return base_caption
    kb = KNOWLEDGE_BASE[predicted_class]
    extra = []
    if model_type == "resnet_gru":
        for key in ("visual", "significance"):
            if key in kb:
                extra.append(kb[key])
    else:
        for key in ("history", "community"):
            if key in kb:
                extra.append(kb[key])
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

    with st.spinner("Generating caption..."):
        caption_fn = caption_resnet_gru if "ResNet" in model_choice else caption_tinyvit
        model_key = "resnet_gru" if "ResNet" in model_choice else "tinyvit"
        base = caption_fn(image, encoder, decoder, vocab, eval_tf, config, device)
        expanded = expand_caption(base, predicted_class, model_key)

    label = "✨ Creative Description" if model_key == "resnet_gru" else "📜 History & Origin"
    st.subheader("English Caption")
    st.success(expanded)

    st.subheader("Nepali Caption")
    st.info(nepali_list[idx])

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
                response_parts = []
                seen_cats = set()
                for chunk, src, cat, score in retrieved:
                    seen_cats.add(cat)
                    label, emoji = CATEGORY_LABELS.get(cat, (cat.capitalize(), "📌"))
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

                    response_parts = []
                    seen_cats = set()
                    for chunk, src, cat, score in retrieved:
                        seen_cats.add(cat)
                        label, emoji = CATEGORY_LABELS.get(cat, (cat.capitalize(), "📌"))
                        response_parts.append(f"{emoji} **{label}:**\n{chunk}")

                    response = (
                        random.choice(QUIPS["curious"]) + "\n\n"
                        + "\n\n".join(response_parts) + "\n\n"
                        + random.choice(QUIPS["celebrate"])
                    )

                    ctx = {"question": query, "categories": list(seen_cats)[:3], "dress": dress_key}

                    st.markdown(response)
                    st.session_state.context_history.append(ctx)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
