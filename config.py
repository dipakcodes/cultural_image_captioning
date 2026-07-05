import os


class CFG:
    # ---------------- Reproducibility ----------------
    SEED = 42

    # ---------------- Paths ----------------
    DATA_DIR   = "/kaggle/input/datasets/dipaksah00/dataset-one"
    CSV_DIR    = "/kaggle/input/datasets/dipaksah00/captions-csv"
    CSV_NAME   = "captions.csv"
    CAPTION_COL = "caption_en"
    OUTPUT_DIR = "/kaggle/working"

    # ---------------- Data & split ----------------
    VAL_FRACTION    = 0.10
    TEST_FRACTION   = 0.10
    SPLIT_BY_SOURCE = True
    IMAGE_SIZE      = 224
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]
    USE_TRAIN_AUG = True

    # ---------------- Vocabulary ----------------
    MIN_WORD_FREQ   = 2
    MAX_CAPTION_LEN = 22

    # ---------------- Tiny ViT Encoder ----------------
    PATCH_SIZE      = 16
    HIDDEN_DIM      = 256
    ENC_NHEAD       = 8
    ENC_LAYERS      = 4
    ENC_DIM_FEED    = 512
    ENC_DROPOUT     = 0.1

    # ---------------- Transformer Decoder ----------------
    DEC_NHEAD       = 8
    DEC_LAYERS      = 4
    DEC_DIM_FEED    = 512
    DEC_DROPOUT     = 0.1

    # ---------------- Training ----------------
    EPOCHS      = 50
    BATCH_SIZE  = 32
    ENCODER_LR  = 1e-4
    DECODER_LR  = 4e-4
    GRAD_CLIP   = 5.0
    NUM_WORKERS = 2

    # ---------------- Early stopping & checkpoints ----------------
    EARLY_STOPPING = True
    PATIENCE       = 8
    MONITOR        = "bleu4"
    CKPT_BEST      = "best_model.pth"
    CKPT_LAST      = "last_model.pth"

    # ---------------- Resume ----------------
    RESUME      = False
    RESUME_PATH = "/kaggle/working/last_model.pth"

    # ---------------- Inference ----------------
    BEAM_SIZE = 3
