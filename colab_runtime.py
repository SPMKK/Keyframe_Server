# colab_runtime.py
import os, gc, contextlib
import torch
import cv2
from PIL import Image

def is_colab_env():
    if os.environ.get("COLAB_MODE") == "1":
        return True
    try:
        import google.colab  # noqa
        return True
    except Exception:
        return False

class RuntimeProfile:
    def __init__(self,
                 ocr_batch_size=6,
                 postproc_workers=12,
                 shot_workers=4,
                 clip_fp16=False,
                 keep_all_frames=True,      # giữ list all_frames trong RAM (local)
                 buffer_paths_only=False,   # chỉ đệm path thay vì giữ PIL.Image (colab)
                 webp_quality=90,
                 caption_sequential=False):
        self.ocr_batch_size = ocr_batch_size
        self.postproc_workers = postproc_workers
        self.shot_workers = shot_workers
        self.clip_fp16 = clip_fp16
        self.keep_all_frames = keep_all_frames
        self.buffer_paths_only = buffer_paths_only
        self.webp_quality = webp_quality
        self.caption_sequential = caption_sequential

def get_profile(force_colab=None):
    colab = is_colab_env() if force_colab is None else bool(force_colab)
    if colab:
        return RuntimeProfile(
            ocr_batch_size=int(os.getenv("OCR_BATCH_SIZE", "3")),
            postproc_workers=int(os.getenv("POSTPROC_WORKERS", "2")),
            shot_workers=min(4, os.cpu_count() or 1),
            clip_fp16=True,
            keep_all_frames=False,
            buffer_paths_only=True,
            webp_quality=85,
            caption_sequential=True,
        )
    return RuntimeProfile()

def maybe_half(model, device="cuda"):
    if isinstance(model, torch.nn.Module) and device.startswith("cuda"):
        try: model.half()
        except Exception: pass
    return model

@contextlib.contextmanager
def open_image(path):
    img = Image.open(path)
    try:
        img.load()
        yield img
    finally:
        try: img.close()
        except Exception: pass

def save_webp_from_array(rgb_array, path, quality=90):
    im = Image.fromarray(rgb_array)
    im.save(path, 'WEBP', quality=quality)
    im.close()

def read_frame_by_index(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def sampled_indices(start_idx, end_idx, sample_rate, max_frames_per_shot):
    idxs = list(range(start_idx, end_idx + 1, sample_rate))
    if len(idxs) > max_frames_per_shot:
        step = max(1, len(idxs) // max_frames_per_shot)
        idxs = idxs[::step]
    return idxs
