import os, sys, random, time
from datetime import datetime
from typing import List, Tuple, Set
import warnings
import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1) 

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#stability & speed toggles
USE_CHANNELS_LAST = False
CUDNN_BENCHMARK  = False          

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

#File Paths 
LOG_DIR  = r"E:\Honours Project\Dataset Face Forensics++\MADD_Logs" #CHANGE DIRECTORY
CKPT_DIR = r"E:\Honours Project\Dataset Face Forensics++\MADD_checkpoints" #CHANGE DIRECTORY
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

#Configuration Parameters here
class Cfg:
    PRETRAINED_CKPT = r"E:\Honours Project\Dataset Face Forensics++\weights\xception-43020ad28.pth"
    TRAIN_DIR = r"E:\Honours Project\Dataset Face Forensics++\dataset_split\train"
    VAL_DIR   = r"E:\Honours Project\Dataset Face Forensics++\dataset_split\val"
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    IMG_SIZE = 224
    MAX_PER_VIDEO_TRAIN = 50
    MAX_PER_VIDEO_VAL   = 100

    BATCH_SIZE = 28              
    LR = 2.0e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTH = 0.1
    USE_AMP = True

    NUM_WORKERS = 8
    PIN_MEMORY = True
    PREFETCH_FACTOR = 8

    USE_AUX = False
    AUX_WEIGHT = 0.10

    EARLY_STOP_PATIENCE = 2
    EPOCHS = 10

    SEED = 42
    DETERMINISTIC = False

    OUT_LAST = os.path.join(CKPT_DIR, "mat_ffpp_last.pth")
    OUT_BEST = os.path.join(CKPT_DIR, "mat_ffpp_best.pth")

# reproducibility 
def set_seed(seed=Cfg.SEED, deterministic=Cfg.DETERMINISTIC):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()

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

# datasets
class _BaseFaceFolder(Dataset):
    def __init__(self, root_dir: str, train: bool, max_per_video: int, seed: int = Cfg.SEED):
        self.root_dir = root_dir; self.train = train; self.max_per_video = max_per_video; self.base_seed = seed
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

# MADD model import from repository
sys.path.append(r"E:\Honours Project\MADD MODEL\multiple-attention-master")
from models.MAT import MAT  

def _count_matches(target_module: nn.Module, src_keys: set) -> int:
    tgt_keys = set(target_module.state_dict().keys())
    return sum(1 for k in src_keys if k in tgt_keys)

def load_imagenet_xception_weights(model: nn.Module, ckpt_path: str) -> int:
    if not (isinstance(ckpt_path, str) and os.path.isfile(ckpt_path)):
        print(f"[WARN] Pretrained checkpoint not found at {ckpt_path}; using random init."); return 0
    print(f"[INFO] Loading ImageNet weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')
    src = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
    if not isinstance(src, dict):
        print("[WARN] Loaded object is not a state_dict; skipping."); return 0
    src_keys = set(src.keys())
    candidates: List[Tuple[str, nn.Module]] = [('model', model)]
    for attr in ['net', 'backbone', 'xception', 'model']:
        if hasattr(model, attr): candidates.append((attr, getattr(model, attr)))
    best_name, best_target, best_matches = None, None, -1
    seen = set()
    for name, tgt in candidates:
        if name in seen: continue
        seen.add(name)
        try: matches = _count_matches(tgt, src_keys)
        except Exception: matches = -1
        if matches > best_matches:
            best_matches, best_target, best_name = matches, tgt, name
    loaded = 0
    if best_target is not None and best_matches > 0:
        missing, unexpected = best_target.load_state_dict(src, strict=False)
        loaded = max(best_matches - len(missing), 0)
        print(f"[INFO] Loaded ImageNet weights into '{best_name}' (strict=False). "
              f"matched={best_matches}, loaded~={loaded}, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:    print(f"[DEBUG] Missing keys (first 10): {missing[:10]}")
        if unexpected: print(f"[DEBUG] Unexpected keys (first 10): {unexpected[:10]}")
    else:
        print("[WARN] Could not find a good target to load weights; proceeding with random init.")
    return loaded

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
        running_loss += loss.item() * y.size(0)
        seen += y.size(0)
    avg_loss = running_loss / max(1, seen) if seen else float('nan')
    acc = (correct / seen) if seen else float('nan')
    print(f"[VAL] val_loss={avg_loss:.4f} acc={acc:.4f}")
    return avg_loss, acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[INFO] TurboJPEG: {'ON' if USE_TURBOJPEG else 'OFF (OpenCV)'}")
    print(f"[INFO] channels_last: {USE_CHANNELS_LAST}, cudnn.benchmark: {CUDNN_BENCHMARK}")

    model = MAT(net='xception', num_classes=2, pretrained=False).to(device)

    loaded = load_imagenet_xception_weights(model, Cfg.PRETRAINED_CKPT)
    pretrained_used = 1 if loaded > 0 else 0

    # Optimizer
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.LR, weight_decay=Cfg.WEIGHT_DECAY, fused=True)
        print("[OPT] Using fused AdamW")
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.LR, weight_decay=Cfg.WEIGHT_DECAY)
        print("[OPT] Using AdamW")

    scaler = torch.amp.GradScaler("cuda", enabled=(Cfg.USE_AMP and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(label_smoothing=Cfg.LABEL_SMOOTH)

    # Data
    train_ds = FaceFolderFast(Cfg.TRAIN_DIR, train=True,  max_per_video=Cfg.MAX_PER_VIDEO_TRAIN)
    val_ds   = FaceFolderFast(Cfg.VAL_DIR,   train=False, max_per_video=Cfg.MAX_PER_VIDEO_VAL)

    dl_common = dict(
        batch_size=Cfg.BATCH_SIZE,
        pin_memory=Cfg.PIN_MEMORY,
        collate_fn=collate_skip_none,
    )
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=Cfg.NUM_WORKERS,
                            drop_last=True, prefetch_factor=Cfg.PREFETCH_FACTOR,
                            persistent_workers=True, **dl_common)

    print(f"[LOADER] workers={Cfg.NUM_WORKERS} pin_memory={Cfg.PIN_MEMORY} prefetch_factor={Cfg.PREFETCH_FACTOR}")
    print(f"[RUN PARAMS] {{'epochs': {Cfg.EPOCHS}, 'batch_size': {Cfg.BATCH_SIZE}, 'lr': {Cfg.LR}, "
          f"'weight_decay': {Cfg.WEIGHT_DECAY}, 'label_smooth': {Cfg.LABEL_SMOOTH}, "
          f"'img_size': {Cfg.IMG_SIZE}, 'max_per_video_train': {Cfg.MAX_PER_VIDEO_TRAIN}, "
          f"'max_per_video_val': {Cfg.MAX_PER_VIDEO_VAL}, 'device': '{device}', "
          f"'pretrained_used': {pretrained_used}}}")
    print("[TRAIN START] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    best_val_acc = -1.0
    epochs_since_improve = 0

    # initial scheduler sizing
    train_ds.resample(epoch=1)
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=Cfg.NUM_WORKERS,
                              drop_last=True, prefetch_factor=Cfg.PREFETCH_FACTOR,
                              persistent_workers=False, **dl_common)
    steps_per_epoch0 = max(len(train_loader), 1)
    total_steps = Cfg.EPOCHS * steps_per_epoch0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=Cfg.LR * 0.01
    )

    for epoch in range(1, Cfg.EPOCHS + 1):
        train_ds.resample(epoch)
        train_loader = DataLoader(train_ds, shuffle=True, num_workers=Cfg.NUM_WORKERS,
                                  drop_last=True, prefetch_factor=Cfg.PREFETCH_FACTOR,
                                  persistent_workers=False, **dl_common)

        print(f"[EPOCH START] {epoch}/{Cfg.EPOCHS} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.train()
        running_loss, seen, correct = 0.0, 0, 0

        timing_left = 12
        prev_end = time.time()

        for batch in tqdm(train_loader, desc=f"Training E{epoch}/{Cfg.EPOCHS}", dynamic_ncols=True, leave=False):
            fetch_time = time.time() - prev_end
            if batch is None:
                scheduler.step(); prev_end = time.time(); continue

            inputs, labels = batch 
            t0 = time.time()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            h2d_time = time.time() - t0

            t1 = time.time()
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.amp.autocast("cuda", enabled=Cfg.USE_AMP):
                    if Cfg.USE_AUX and hasattr(model, "train_batch"):
                        pack = model.train_batch(inputs, labels, jump_aux=False, drop_final=False)
                        outputs = pack['ensemble_logit']
                        loss = pack['ensemble_loss'] + Cfg.AUX_WEIGHT * pack['aux_loss']
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            gpu_time = time.time() - t1

            scheduler.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

            if timing_left > 0:
                total = fetch_time + h2d_time + gpu_time
                print(f"[TIMING] fetch={fetch_time:.3f}s  h2d={h2d_time:.3f}s  gpu={gpu_time:.3f}s  total={total:.3f}s")
                timing_left -= 1
            prev_end = time.time()

        train_avg = running_loss / max(1, seen) if seen else float('nan')
        train_acc = (correct / seen) if seen else float('nan')
        print(f"[TRAIN] epoch={epoch} avg_loss={train_avg:.5f} acc={train_acc:.4f}")

        # Validation + checkpoints + early stop
        val_loss, val_acc = validate(model, val_loader, device, nn.CrossEntropyLoss())
        torch.save(model.state_dict(), Cfg.OUT_LAST)
        print(f"[SAVE] Last checkpoint -> {Cfg.OUT_LAST}")

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc; epochs_since_improve = 0
            torch.save(model.state_dict(), Cfg.OUT_BEST)
            print(f"[SAVE] New BEST (val_acc={best_val_acc:.4f}) -> {Cfg.OUT_BEST}")
        else:
            epochs_since_improve += 1
            print(f"[EARLY-STOP] no improvement for {epochs_since_improve} epoch(s)")
            if epochs_since_improve >= Cfg.EARLY_STOP_PATIENCE:
                print("[EARLY-STOP] stopping training"); break

        print(f"[EPOCH END]   {epoch}/{Cfg.EPOCHS} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("[TRAIN END] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
