import os
import sys
import argparse
import logging
import random
import time
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

# DF40 dataset
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


# simple logging
def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger("xception_df40")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# dataset/augmentation
def ensure_dataset_defaults(config: dict) -> dict:
    config.setdefault('frame_num', {'train': 50, 'test': 100})
    config.setdefault('compression', 'raw')  
    config.setdefault('resolution', config.get('img_size', 224))
    config.setdefault('with_landmark', False)
    config.setdefault('with_mask', False)
    config.setdefault('use_data_augmentation', True)
    config.setdefault('video_mode', False)

    # normalization defaults 
    config.setdefault('mean', [0.485, 0.456, 0.406])
    config.setdefault('std',  [0.229, 0.224, 0.225])

    # augmentation defaults
    aug = dict(config.get('data_aug', {}))

    # flips/rotation
    aug.setdefault('flip_prob', 0.5)
    aug.setdefault('rotate_limit', 10)
    aug.setdefault('rotate_prob', 0.2)

    # brightness/contrast
    aug.setdefault('brightness_limit', 0.2)
    aug.setdefault('contrast_limit', 0.2)

    aug.setdefault('color_jitter', 0.0)
    aug.setdefault('grayscale_prob', 0.0)

    # blur
    aug.setdefault('blur_prob', 0.0)
    aug.setdefault('blur_limit', 3)
    aug.setdefault('sigma_limit', (0.1, 2.0))

    # JPEG/ImageCompression range
    aug.setdefault('quality_lower', 60)
    aug.setdefault('quality_upper', 100)

    # aliases
    aug.setdefault('jpeg_quality_lower', aug['quality_lower'])
    aug.setdefault('jpeg_quality_upper', aug['quality_upper'])
    aug.setdefault('jpeg_prob', 0.0)

    config['data_aug'] = aug
    return config

# label mapping
LABEL_DICT = {
    "DFD_fake": 1, "DFD_real": 0,
    "FF-SH": 1, "FF-F2F": 1, "FF-DF": 1, "FF-FS": 1, "FF-NT": 1, "FF-FH": 1, "FF-real": 0,
    "CelebDFv1_real": 0, "CelebDFv1_fake": 1, "CelebDFv2_real": 0, "CelebDFv2_fake": 1,
    "DFDCP_Real": 0, "DFDCP_FakeA": 1, "DFDCP_FakeB": 1,
    "DFDC_Fake": 1, "DFDC_Real": 0,
    "DF_fake": 1, "DF_real": 0,
    "UADFV_Fake": 1, "UADFV_Real": 0,
    "roop_Real": 0, "roop_Fake": 1,
    "FSAll_Fake": 1, "FSAll_Real": 0,
    "FRAll_Fake": 1, "FRAll_Real": 0,
    "EFSAll_Fake": 1, "EFSAll_Real": 0,
    "DF40_train_Fake": 1, "DF40_train_Real": 0,
    "e4s_Fake": 1, "e4s_Real": 0, "danet_Fake": 1, "danet_Real": 0, "fomm_Fake": 1, "fomm_Real": 0,
    "Collaborative_Diffusion_Fake": 1, "Collaborative_Diffusion_Real": 0,
    "e4e_Fake": 1, "e4e_Real": 0, "hyperreenact_Fake": 1, "hyperreenact_Real": 0,
    "MRAA_Fake": 1, "MRAA_Real": 0, "one_shot_free_Fake": 1, "one_shot_free_Real": 0,
    "pirender_Fake": 1, "pirender_Real": 0, "tpsm_Fake": 1, "tpsm_Real": 0,
    "facedancer_Fake": 1, "facedancer_Real": 0, "facevid2vid_Fake": 1, "facevid2vid_Real": 0,
    "mcnet_Fake": 1, "mcnet_Real": 0, "mraa_Fake": 1, "mraa_Real": 0,
    "fsgan_Fake": 1, "fsgan_Real": 0, "lia_Fake": 1, "lia_Real": 0,
    "inswap_Fake": 1, "inswap_Real": 0, "simswap_Fake": 1, "simswap_Real": 0,
    "sadtalker_Fake": 1, "sadtalker_Real": 0, "wav2lip_Fake": 1, "wav2lip_Real": 0,
    "uniface_Fake": 1, "uniface_Real": 0, "blendface_Fake": 1, "blendface_Real": 0,
    "mobileswap_Fake": 1, "mobileswap_Real": 0, "faceswap_Fake": 1, "faceswap_Real": 0,
    "dalle2_face_Fake": 1, "dalle2_face_Real": 0, "MidJourney_Fake": 1, "MidJourney_Real": 0,
    "heygen_Fake": 1, "heygen_Real": 0, "whichisreal_Fake": 1, "whichisreal_Real": 0,
    "StyleGAN2_Fake": 1, "StyleGAN2_Real": 0, "StyleGAN3_Fake": 1, "StyleGAN3_Real": 0,
    "StyleGANXL_Fake": 1, "StyleGANXL_Real": 0, "ddim_Fake": 1, "ddim_Real": 0,
    "DiT_Fake": 1, "DiT_Real": 0, "pixart_Fake": 1, "pixart_Real": 0,
    "SiT_Fake": 1, "SiT_Real": 0, "sd1.5_Fake": 1, "sd1.5_Real": 0,
    "sd2.1_Fake": 1, "sd2.1_Real": 0, "VQGAN_Fake": 1, "VQGAN_Real": 0,
    "rddm_Fake": 1, "rddm_Real": 0, "stargan_Fake": 1, "stargan_Real": 0,
    "starganv2_Fake": 1, "starganv2_Real": 0, "styleclip_Fake": 1, "styleclip_Real": 0,
    "deepfacelab_Fake": 1, "deepfacelab_Real": 0, "CollabDiff_Fake": 1, "CollabDiff_Real": 0,
    "adm_Fake": 1, "adm_Real": 0, "biggan_Fake": 1, "biggan_Real": 0,
    "glide_Fake": 1, "glide_Real": 0, "midjourney_Fake": 1, "midjourney_Real": 0,
    "sdv4_Fake": 1, "sdv4_Real": 0, "sdv5_Fake": 1, "sdv5_Real": 0,
    "vqdm_Fake": 1, "vqdm_Real": 0, "wukong_Fake": 1, "wukong_Real": 0,
}

# Xception (timm) builder 
def build_xception(config, logger):
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for this script. Install with: pip install timm"
        ) from e


    model = timm.create_model('legacy_xception', pretrained=False, num_classes=2, in_chans=3)
    logger.info("Initialized timm model: legacy_xception (num_classes=2).")

    # Load FF++ checkpoint 
    ckpt_path = config.get('pretrained_path', '')
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            if 'state_dict' in state and isinstance(state['state_dict'], dict):
                state = state['state_dict']
            elif 'model' in state and isinstance(state['model'], dict):
                state = state['model']
            elif 'net' in state and isinstance(state['net'], dict):
                state = state['net']

        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                if k.startswith("module."):
                    new_state[k[len("module."):]] = v
                else:
                    new_state[k] = v
            state = new_state

        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded pretrained weights from {ckpt_path}. "
            f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )
    else:
        logger.info("No pretrained_path found or file missing; training from scratch.")
    return model



def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_guess(batch):
    if isinstance(batch, dict):
        return batch.get('image', None), batch.get('label', None)
    return None, None


def build_optimizer(params, lr, weight_decay, fused=True):
    try:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)
    except TypeError:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

import numpy as np

def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]
    P = labels.sum()
    N = len(labels) - P
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = np.concatenate([[0.0], tps / max(P, 1)])
    fpr = np.concatenate([[0.0], fps / max(N, 1)])
    thresholds = np.concatenate([[scores[0] + 1e-12], scores])
    return fpr, tpr, thresholds

def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) == 0:
        return float("nan")
    fpr, tpr, _ = _roc_curve(np.asarray(scores, np.float64), np.asarray(labels, np.int64))
    return float(np.trapz(tpr, fpr))

def precision_recall_f1(TP: int, FP: int, FN: int):
    precision = float(TP / (TP + FP)) if (TP + FP) > 0 else float("nan")
    recall    = float(TP / (TP + FN)) if (TP + FN) > 0 else float("nan")
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision > 0 and recall > 0) else float("nan")
    return precision, recall, f1

def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, amp=False, logger=None, epoch=1, num_epochs=1):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    t0 = time.time()

    all_probs = []   
    all_labels = []  
    TP = FP = FN = TN = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", dynamic_ncols=True, leave=True)
    for it, batch in enumerate(pbar):
        x, y = collate_guess(batch)
        if x is None or y is None:
            raise RuntimeError("Could not parse batch from dataset (image/label missing).")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        running_loss += float(loss.item()) * bs
        total += bs

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1]  
            preds = logits.argmax(dim=1)

            correct += int((preds == y).sum().item())

            TP += int(((preds == 1) & (y == 1)).sum().item())
            FP += int(((preds == 1) & (y == 0)).sum().item())
            FN += int(((preds == 0) & (y == 1)).sum().item())
            TN += int(((preds == 0) & (y == 0)).sum().item())

            all_probs.extend(probs.detach().float().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

        avg_loss = running_loss / max(1, total)
        avg_acc  = correct / max(1, total)
        lr_val   = optimizer.param_groups[0]['lr']
        elapsed  = time.time() - t0
        ips      = total / max(1e-6, elapsed)  
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}", ips=f"{ips:.1f}", lr=f"{lr_val:.2e}", it=it)

    epoch_loss = running_loss / max(1, total)
    epoch_acc  = correct / max(1, total)
    auc        = roc_auc(all_probs, all_labels) if total > 0 else float("nan")
    precision, recall, f1 = precision_recall_f1(TP, FP, FN)

    if logger:
        logger.info(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={epoch_loss:.4f}  "
            f"acc={epoch_acc:.4f}  "
            f"auc={auc:.4f}  "
            f"precision={precision:.4f}  "
            f"recall={recall:.4f}  "
            f"f1={f1:.4f}"
        )
    return epoch_loss, epoch_acc, auc, precision, recall, f1



def parse_args():
    p = argparse.ArgumentParser("Train Xception (timm) on DF40", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--dataset_root_rgb", type=str, required=True,
                   help="Root folder containing datasets (e.g., D:\\...\\deepfakes_detection_datasets)")
    p.add_argument("--dataset_json_folder", type=str, required=True,
                   help="Folder containing DF40 dataset_json")
    p.add_argument("--pretrained_path", type=str,
                   default=r"E:\Honours Project\Dataset Face Forensics++\Xception_checkpoints\xception_ffpp_best_v3-1-28.pth",
                   help="Path to Xception FF++ checkpoint (.pth) for fine-tuning")
    p.add_argument("--log_dir", type=str,
                   default=r"D:\UNI STUFF\DF40\Train_XCEPTION",
                   help="Where to save logs/checkpoints")
    p.add_argument("--train_dataset", type=str, nargs="+", default=["DF40_all"],
                   help="One or more dataset keys found in dataset_json")
    p.add_argument("--test_dataset", type=str, nargs="*", default=[], help="(unused)")
    p.add_argument("--compression", type=str, default="raw", help="raw / c23 / c40")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs_train", type=int, default=28)
    p.add_argument("--bs_val", type=int, default=28)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=299,
                   help="Input size (dataset resize)")
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    p.add_argument("--max_samples_per_epoch", type=int, default=20000,
                   help="If >0, randomly sample this many items per epoch to cap ETA.")
    p.add_argument("--subset_seed", type=int, default=1234, help="Random seed for subsetting")

    p.add_argument("--train_only", action="store_true", help="Skip evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    datasets_root_parent = os.path.dirname(os.path.normpath(args.dataset_root_rgb))
    os.chdir(datasets_root_parent)

    config = {
        'mode': 'train',
        'lmdb': False,
        'dry_run': False,
        'dataset_root_rgb': args.dataset_root_rgb,
        'dataset_json_folder': args.dataset_json_folder,
        'cuda': torch.cuda.is_available(),
        'cudnn': True,
        'manualSeed': 1337,

        'model_name': 'xception',
        'pretrained_path': args.pretrained_path,

        'nEpochs': args.epochs,
        'start_epoch': 1,

        'train_batchSize': args.bs_train,
        'test_batchSize': args.bs_val,
        'workers': args.workers,
        'pin_memory': True,

        'img_size': args.img_size,
        'label_smooth': args.label_smooth,

        'max_per_video_train': 50,
        'max_per_video_val': 100,

        'amp': args.amp,

        'compression': args.compression,

        'optimizer': {
            'type': 'adamw',
            'adamw': {
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'beta1': 0.9, 'beta2': 0.999,
                'eps': 1e-08,
                'fused': True
            }
        },
        'lr_scheduler': None,

        'log_dir': args.log_dir,
        'save_ckpt': True,
        'save_feat': False,

        'train_dataset': args.train_dataset,
        'test_dataset': args.test_dataset,

        'label_dict': LABEL_DICT,
    }

    config = ensure_dataset_defaults(config)

    os.makedirs(config['log_dir'], exist_ok=True)
    logger = setup_logger(config['log_dir'])

    # print config
    logger.info("--------------- Configuration ---------------")
    logger.info(f"Working dir set to: {os.getcwd()}")
    for k, v in config.items():
        if k == 'label_dict':
            logger.info(f"{k}: (len={len(v)})")
        else:
            logger.info(f"{k}: {v}")

    #seed / device
    set_seed(config.get('manualSeed', 1337))
    device = get_device()
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # build dataset
    logger.info(f"frame_num: {config['frame_num']}")
    full_train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')
    orig_collate = full_train_set.collate_fn

    if args.max_samples_per_epoch and args.max_samples_per_epoch > 0:
        n = min(args.max_samples_per_epoch, len(full_train_set))
        rng = random.Random(args.subset_seed)
        indices = list(range(len(full_train_set)))
        rng.shuffle(indices)
        indices = indices[:n]
        train_set = Subset(full_train_set, indices)
        logger.info(f"Using a subset: {len(train_set)}/{len(full_train_set)} samples per epoch.")
    else:
        train_set = full_train_set
        logger.info(f"Using full dataset: {len(train_set)} samples.")

    # DataLoader tuning for Windows
    dl_kwargs = dict(
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=config['workers'],
        pin_memory=config['pin_memory'],
        collate_fn=orig_collate,
        drop_last=True,
        persistent_workers=(config['workers'] > 0)
    )
    if config['workers'] > 0:
        dl_kwargs['prefetch_factor'] = 2  

    train_loader = DataLoader(train_set, **dl_kwargs)
    logger.info(f"Batches/epoch (approx): {len(train_loader)}")

    #build model
    model = build_xception(config, logger).to(device)

    #loss / optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smooth']).to(device)

    opt_cfg = config['optimizer']['adamw']
    optimizer = build_optimizer(model.parameters(), lr=opt_cfg['lr'],
                                weight_decay=opt_cfg['weight_decay'],
                                fused=opt_cfg.get('fused', True))

    scaler = torch.cuda.amp.GradScaler() if (config['amp'] and device.type == 'cuda') else None

    # train loop 
    best_loss = float('inf')
    ckpt_dir = os.path.join(config['log_dir'], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        loss, acc, auc, prec, rec, f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, amp=config['amp'], logger=logger,
            epoch=epoch, num_epochs=config['nEpochs']
        )
        logger.info(
            f"[SUMMARY] epoch={epoch}  loss={loss:.4f}  acc={acc:.4f}  "
            f"auc={auc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}"
        )

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        #save last checkpoint
        last_path = os.path.join(ckpt_dir, "last.pth")
        torch.save(state, last_path)

        # save a per-epoch snapshot and alias
        epoch_path = os.path.join(ckpt_dir, f"epoch{epoch}.pth")
        torch.save(state, epoch_path)

        alias_path = os.path.join(ckpt_dir, f"2nd_xception_epoch{epoch}.pth")
        shutil.copyfile(epoch_path, alias_path)
        logger.info(f"Saved per-epoch checkpoint: {epoch_path} (alias: {alias_path})")

        # save best when loss improves
        if config['save_ckpt'] and loss < best_loss:
            best_loss = loss
            best_path = os.path.join(ckpt_dir, f"best_xception_epoch{epoch}.pth")
            state['best_loss'] = best_loss
            torch.save(state, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
