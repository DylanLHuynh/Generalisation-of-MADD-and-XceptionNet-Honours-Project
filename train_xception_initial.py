import os, sys, random, time
from datetime import datetime
from typing import List, Tuple, Set
import warnings

import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# CHANGE HERE PATHING
LOG_DIR  = r"E:\Honours Project\Dataset Face Forensics++\Xception_Logs" #insert own directory for logs
CKPT_DIR = r"E:\Honours Project\Dataset Face Forensics++\Xception_checkpoints" # insert checkpoint directory path
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

#Config settings for parameters
class Cfg:
    IO_PRESET = "A"

    TRAIN_DIR = r"E:\Honours Project\Dataset Face Forensics++\dataset_split\train"
    VAL_DIR   = r"E:\Honours Project\Dataset Face Forensics++\dataset_split\val"
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    IMG_SIZE = 224
    MAX_PER_VIDEO_TRAIN = 50
    MAX_PER_VIDEO_VAL   = 100

    EPOCHS = 5
    BATCH_SIZE = 28
    LR = 2.0e-4
    WEIGHT_DECAY = 1e-4
    USE_AMP = True
    LABEL_SMOOTH = 0.1 

    # Preset A 
    A_NUM_WORKERS = 8
    A_PIN_MEMORY = True
    A_PREFETCH_FACTOR = 8
    A_PERSISTENT_VAL_WORKERS = True
    A_CV_THREADS_PER_WORKER = 1

    EARLY_STOP_PATIENCE = 2

    SEED = 42
    DETERMINISTIC = False

    OUT_LAST = os.path.join(CKPT_DIR, "xception_ffpp_last_v3-1-28.pth")
    OUT_BEST = os.path.join(CKPT_DIR, "xception_ffpp_best_v3-1-28.pth")

#reproducibility
def set_seed(seed=Cfg.SEED, deterministic=Cfg.DETERMINISTIC):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()

#fast transforms
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _resize_shorter_to(img: np.ndarray, target_short: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) == target_short:
        return img
    scale = target_short / float(min(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    return img[y0:y0+size, x0:x0+size]

def preprocess_np(img_rgb: np.ndarray, train: bool) -> torch.Tensor:
    img = _resize_shorter_to(img_rgb, Cfg.IMG_SIZE)
    img = _center_crop(img, Cfg.IMG_SIZE)
    if train and random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1, :])
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  
    return torch.from_numpy(img)  

# ===== dataset =====
class _BaseFaceFolder(Dataset):
    def __init__(self, root_dir: str, train: bool, max_per_video: int, seed: int = Cfg.SEED):
        self.root_dir = root_dir
        self.train = train
        self.max_per_video = max_per_video
        self.base_seed = seed
        self.bad_paths: Set[str] = set()
        self.videos: List[Tuple[List[str], int]] = []

        for label, cls in enumerate(["real", "fake"]):
            class_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_dir): continue
            for video_subdir in os.listdir(class_dir):
                if video_subdir.startswith(".") or video_subdir.startswith("._"): continue
                vdir = os.path.join(class_dir, video_subdir)
                if not os.path.isdir(vdir): continue
                imgs = [os.path.join(vdir, f) for f in os.listdir(vdir)
                        if (not f.startswith(".")) and f.lower().endswith(Cfg.IMAGE_EXTS)]
                if imgs: self.videos.append((imgs, label))

        self.samples: List[Tuple[str, int]] = []
        self.resample(epoch=0)

    def resample(self, epoch: int = 0):
        rng = random.Random(self.base_seed + epoch)
        new_samples: List[Tuple[str, int]] = []
        k = self.max_per_video
        for imgs, label in self.videos:
            chosen = imgs if len(imgs) <= k else rng.sample(imgs, k)
            new_samples.extend((p, label) for p in chosen)
        rng.shuffle(new_samples)
        self.samples = new_samples

    def __len__(self): return len(self.samples)

class FaceFolderFast(_BaseFaceFolder):
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if path in self.bad_paths: return None, None
        for _ in range(2):
            try:
                with open(path, "rb") as f:
                    buf = f.read()
                if USE_TURBOJPEG and (path.lower().endswith(".jpg") or path.lower().endswith(".jpeg")):
                    img = _JPEG.decode(buf, pixel_format=TJPF_RGB)  
                else:
                    data = np.frombuffer(buf, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None: raise ValueError("cv2.imdecode failed")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = preprocess_np(img, train=self.train) 
                return tensor, label
            except Exception:
                pass
        self.bad_paths.add(path); return None, None

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

#model Xception
import timm
def build_model():
    model = timm.create_model("legacy_xception", pretrained=True, num_classes=2)
    return model

#helpers
def _apply_io_preset():
    if Cfg.IO_PRESET.upper() == "A":
        cv2.setNumThreads(Cfg.A_CV_THREADS_PER_WORKER)
        io = dict(
            num_workers=Cfg.A_NUM_WORKERS,
            pin_memory=Cfg.A_PIN_MEMORY,
            prefetch_factor=Cfg.A_PREFETCH_FACTOR,
            persistent_val_workers=Cfg.A_PERSISTENT_VAL_WORKERS
        )
    else:
        cv2.setNumThreads(Cfg.B_CV_THREADS)
        io = dict(
            num_workers=Cfg.B_NUM_WORKERS,
            pin_memory=Cfg.B_PIN_MEMORY,
            prefetch_factor=Cfg.B_PREFETCH_FACTOR,
            persistent_val_workers=Cfg.B_PERSISTENT_VAL_WORKERS
        )
    return io

@torch.no_grad()
def validate(model, val_loader, device, criterion):
    model.eval()
    running_loss, seen, correct = 0.0, 0, 0
    for batch in tqdm(val_loader, desc="Validating", dynamic_ncols=True, leave=False):
        if batch is None: continue
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        bs = y.size(0)
        running_loss += loss.item() * bs
        seen += bs
    avg_loss = running_loss / max(1, seen) if seen else float('nan')
    acc = (correct / seen) if seen else float('nan')
    print(f"[VAL] val_loss={avg_loss:.4f} acc={acc:.4f}")
    return avg_loss, acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("[INFO] TurboJPEG:", "ON" if USE_TURBOJPEG else "OFF (OpenCV)")
    print(f"[INFO] channels_last: False, cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"[INFO] I/O preset: {Cfg.IO_PRESET}")

    model = build_model().to(device)

    # Optimizer
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.LR, weight_decay=Cfg.WEIGHT_DECAY, fused=True)
        print("[OPT] Using fused AdamW")
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.LR, weight_decay=Cfg.WEIGHT_DECAY)
        print("[OPT] Using AdamW")

    scaler = torch.amp.GradScaler("cuda", enabled=(Cfg.USE_AMP and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()  

    # Datasets
    train_ds = FaceFolderFast(Cfg.TRAIN_DIR, train=True,  max_per_video=Cfg.MAX_PER_VIDEO_TRAIN)
    val_ds   = FaceFolderFast(Cfg.VAL_DIR,   train=False, max_per_video=Cfg.MAX_PER_VIDEO_VAL)

    io = _apply_io_preset()

    # VAL loader
    dl_val = dict(
        batch_size=Cfg.BATCH_SIZE,
        pin_memory=io["pin_memory"],
        collate_fn=collate_skip_none,
        num_workers=io["num_workers"],
        drop_last=True,
        prefetch_factor=io["prefetch_factor"],
        persistent_workers=io["persistent_val_workers"],
    )
    val_loader = DataLoader(val_ds, shuffle=False, **dl_val)

    print(f"[LOADER] workers={io['num_workers']} pin_memory={io['pin_memory']} prefetch_factor={io['prefetch_factor']}")
    print(f"[RUN PARAMS] {{'epochs': {Cfg.EPOCHS}, 'batch_size': {Cfg.BATCH_SIZE}, 'lr': {Cfg.LR}, "
          f"'weight_decay': {Cfg.WEIGHT_DECAY}, 'label_smooth': {Cfg.LABEL_SMOOTH}, 'img_size': {Cfg.IMG_SIZE}, "
          f"'max_per_video_train': {Cfg.MAX_PER_VIDEO_TRAIN}, 'max_per_video_val': {Cfg.MAX_PER_VIDEO_VAL}, "
          f"'device': '{device}'}}")
    print("[TRAIN START] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    best_val_acc = -1.0
    epochs_since_improve = 0

    # Build initial train loader 
    train_ds.resample(epoch=1)
    dl_train = dict(
        batch_size=Cfg.BATCH_SIZE,
        pin_memory=io["pin_memory"],
        collate_fn=collate_skip_none,
        num_workers=io["num_workers"],
        drop_last=True,
        prefetch_factor=io["prefetch_factor"],
        persistent_workers=False, 
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_train)
    steps_per_epoch0 = max(len(train_loader), 1)
    total_steps = Cfg.EPOCHS * steps_per_epoch0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=Cfg.LR * 0.01
    )

    for epoch in range(1, Cfg.EPOCHS + 1):
        train_ds.resample(epoch)
        train_loader = DataLoader(train_ds, shuffle=True, **dl_train)

        print(f"[EPOCH START] {epoch}/{Cfg.EPOCHS} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.train()
        running_loss, seen, correct = 0.0, 0, 0

        timing_left = 12
        prev_end = time.time()

        for batch in tqdm(train_loader, desc=f"Training E{epoch}/{Cfg.EPOCHS}", dynamic_ncols=True, leave=False):
            fetch_time = time.time() - prev_end

            if batch is None:
                scheduler.step()
                prev_end = time.time()
                continue

            t0 = time.time()
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            h2d_time = time.time() - t0

            t1 = time.time()
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.amp.autocast("cuda", enabled=Cfg.USE_AMP):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            gpu_time = time.time() - t1

            scheduler.step()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            bs = y.size(0)
            running_loss += loss.item() * bs
            seen += bs

            if timing_left > 0:
                total = fetch_time + h2d_time + gpu_time
                print(f"[TIMING] fetch={fetch_time:.3f}s  h2d={h2d_time:.3f}s  gpu={gpu_time:.3f}s  total={total:.3f}s")
                timing_left -= 1

            prev_end = time.time()

        train_avg = running_loss / max(1, seen) if seen else float('nan')
        train_acc = (correct / seen) if seen else float('nan')
        print(f"[TRAIN] epoch={epoch} avg_loss={train_avg:.5f} acc={train_acc:.4f}")

        # Validation
        val_loss, val_acc = validate(model, val_loader, device, nn.CrossEntropyLoss())
        torch.save(model.state_dict(), Cfg.OUT_LAST)
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            epochs_since_improve = 0
            torch.save(model.state_dict(), Cfg.OUT_BEST)
            print(f"[SAVE] New BEST (val_acc={best_val_acc:.4f}) -> {Cfg.OUT_BEST}")
        else:
            epochs_since_improve += 1
            print(f"[EARLY-STOP] no improvement for {epochs_since_improve} epoch(s)")
            if epochs_since_improve >= Cfg.EARLY_STOP_PATIENCE:
                print("[EARLY-STOP] stopping training")
                break

        print(f"[EPOCH END]   {epoch}/{Cfg.EPOCHS} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("[TRAIN END] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
