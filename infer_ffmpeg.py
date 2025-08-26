import os
import sys
import math
import numpy as np
# import tensorflow as tf
import torch
import clip
import cv2
import subprocess # <- ADDED
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
from PIL import Image
import shutil
import json
from typing import List, Tuple
import source.RAM as RAM
# import easyocr
import re
# import google.generativeai as genai
from PIL import Image
import time
# New imports for concurrency
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading
from transformers import SiglipProcessor, SiglipModel
from source.gemini_mistral_server import GeminiApiKeyManager, MistralApiKeyManager
from source.Generator import Generator
import torch
import torch.nn as nn
import torch.nn.functional as functional
import random
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import tempfile
from typing import List, IO, Optional
# from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor, Siglip2Model, Siglip2Processor
# (All class definitions like TransNetV2 and other helper methods remain the same)
# ... [TransNetV2 class and other methods from previous answer go here] ...
# NOTE: To keep the answer focused, I am omitting the unchanged code.
# Assume the TransNetV2 class and helper methods from the previous answer are present.
OCR_BATCH_SIZE = 6
MISTRAL_SHEET_URL = os.getenv("MISTRAL_SHEET_URL", "https://docs.google.com/spreadsheets/d/1NAlj7OiD9apH3U47RLJK0en1wLSW78X5zqmf6NmVUA4/export?format=csv&gid=0")
GEMINI_SHEET_URL = os.getenv("GEMINI_SHEET_URL", "https://docs.google.com/spreadsheets/d/1gqlLToS3OXPA-CvfgXRnZ1A6n32eXMTkXz4ghqZxe2I/export?format=csv&gid=0")

API_KEY = GeminiApiKeyManager(
    sheet_url=GEMINI_SHEET_URL
).get_active_key_count()
#prompt
API_KEY_MISTRAL = MistralApiKeyManager(
    sheet_url=MISTRAL_SHEET_URL
).get_active_key_count()
OCR_PROMPT = '''
Step 1: Perform OCR on the provided image. Extract all visible characters exactly as they appear, even if they are small, low-contrast, partially occluded, distorted, or angled.  
- Prioritize text from signboards, banners, posters, or other high-importance areas.  
- For each character, use only what is clearly visible—do NOT guess or reconstruct missing parts.  
- Preserve the original spatial reading order: top-to-bottom, left-to-right.  
- Return the raw text as a single continuous string with words separated by exactly one space.  
- Do not insert line breaks, punctuation, or extra commentary.

Step 2: Using the OCR output from Step 1, produce a corrected version:  
- Fix spelling errors, grammar issues, and incorrect word forms.  
- Preserve the meaning and intent of the original text.  
- Maintain proper Vietnamese diacritics where applicable.  
- Do not add extra words or rephrase beyond necessary corrections.

Final Output Format:  
{"raw_text": "<raw OCR text>", "corrected_text": "<corrected version>"}
'''
try:
    from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
except ImportError:
    raise ImportError("Không thể import 'TransNetV2Supernet'. Vui lòng đảm bảo file model 'supernet_...' tồn tại.")

class AutoShot:
    """
    AutoShot model for shot boundary detection, structured as a drop-in 
    replacement for the TransNetV2 class.
    This class detects scene transitions in videos using a PyTorch supernet.
    """
    def __init__(self, checkpoint_path=None):
        """
        Khởi tạo và tải model AutoShot từ checkpoint.
        Args:
            checkpoint_path (str): Đường dẫn bắt buộc tới file checkpoint của AutoShot (ví dụ: 'ckpt.pth').
        """
        checkpoint_path = os.path.join(os.path.dirname(__file__), "ckpt_0_200_0.pth") if checkpoint_path is None else checkpoint_path
        print(f"[AutoShot] Using checkpoint path: {checkpoint_path}")
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[AutoShot] ERROR: Checkpoint file is required and was not found at '{checkpoint_path}'.")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[AutoShot] Using device: {self.device}")

        # Tải Supernet model
        self.model = TransNetV2Supernet().eval().to(self.device)
        self._input_size = (27, 48, 3) # Giữ lại để tương thích nếu có code nào đó check

        print(f'[AutoShot] Loading checkpoint from {checkpoint_path}')
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get('net', checkpoint)
            
            model_dict = self.model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            print(f"[AutoShot] Loading {len(filtered_dict)}/{len(state_dict)} parameters from checkpoint.")
            if len(filtered_dict) == 0:
                print("[AutoShot] WARNING: No matching parameters found in the checkpoint.")

            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)
            print("[AutoShot] Model loaded successfully!")
        except Exception as e:
            raise IOError(f"[AutoShot] Failed to load checkpoint file '{checkpoint_path}'. Error: {e}")

    def predict_raw(self, frames: np.ndarray):
        """
        Dự đoán trên một batch duy nhất. Tương đương `predict_raw` của TransNetV2.
        Args:
            frames (np.ndarray): Một batch duy nhất có shape [1, 100, 27, 48, 3].
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (single_frame_pred, all_frames_pred).
                                           Đối với AutoShot, hai giá trị này là giống nhau.
        """
        # (N, T, H, W, C) -> (N, C, T, H, W)
        batch_tensor = torch.from_numpy(frames.transpose((0, 4, 1, 2, 3))).float()
        batch_tensor = batch_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(batch_tensor)
            if isinstance(output, tuple):
                output = output[0]  # Lấy output đầu tiên nếu là tuple
            
            # Áp dụng sigmoid và loại bỏ chiều batch
            predictions = torch.sigmoid(output).squeeze(0).cpu().numpy()
        
        # AutoShot chỉ có một luồng dự đoán, ta trả về nó cho cả hai vị trí
        # để giữ nguyên chữ ký hàm của TransNetV2
        return predictions, predictions

    def predict_frames(self, frames: np.ndarray):
        """
        Dự đoán trên một chuỗi các frame. Tương đương `predict_frames` của TransNetV2.
        Args:
            frames (np.ndarray): Mảng các frame có shape [frames, height, width, 3].
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (single_frame_pred, all_frames_pred).
        """
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            f"[AutoShot] Input shape must be [frames, height, width, 3], but got {frames.shape}."

        # Trình vòng lặp input (input_iterator) được tái tạo từ TransNetV2
        def input_iterator():
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis] # Thêm chiều batch

        predictions = []
        for inp in input_iterator():
            # inp có shape [1, 100, 27, 48, 3]
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            
            # Lấy 50 frame ở giữa, loại bỏ padding
            predictions.append((single_frame_pred[25:75], all_frames_pred[25:75]))

            print("\r[AutoShot] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        # Cắt bớt phần thừa để khớp với số lượng frame gốc
        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]

    def predict_video(self, video_fn: str):
        """
        Trích xuất frame và dự đoán trên toàn bộ video. Giao diện chính.
        Args:
            video_fn (str): Đường dẫn tới file video.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (video, single_frame_pred, all_frames_pred).
        """
        """
        Extracts frames from a video for TransNetV2 and runs predictions,
        using GPU hardware acceleration for decoding where possible.
        """
        print(f"[TransNetV2-PyTorch] Extracting frames from {video_fn} using ffmpeg CLI")
        codec = FFmpegReader.get_codec(video_fn)
        print(f"[FFmpeg] Detected video codec: {codec}")
        command = ['ffmpeg']

        # Add hardware acceleration arguments if enabled and supported
        if FFmpegReader._hw_accel_enabled:
            codec = FFmpegReader.get_codec(video_fn)
            decoder_map = {
                'h264': 'h264_cuvid', 'hevc': 'hevc_cuvid', 'av1': 'av1_cuvid',
                'vp9': 'vp9_cuvid', 'mpeg2video': 'mpeg2_cuvid', 'vc1': 'vc1_cuvid',
            }
            if codec in decoder_map:
                print(f"[TransNetV2] Using GPU acceleration for '{codec}' codec.")
                command.extend(['-hwaccel', 'cuda', '-c:v', decoder_map[codec]])
            else:
                print(f"[TransNetV2] Codec '{codec}' not supported for GPU acceleration. Falling back to CPU.")

        # Add the rest of the ffmpeg arguments
        command.extend([
            '-i', video_fn,
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-sws_flags', 'bilinear',
            '-s', '48x27',
            'pipe:1'
        ])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=True # Use check=True to automatically raise CalledProcessError on failure
            )
            video_stream = result.stdout
        except FileNotFoundError:
            raise RuntimeError("[FFmpeg] 'ffmpeg' command not found. Please ensure ffmpeg is installed.")
        except subprocess.CalledProcessError as e:
            error_details = e.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"[FFmpeg] Failed to decode {video_fn} for TransNetV2.\n"
                                f"FFmpeg Command: {' '.join(command)}\n"
                                f"Exit Code: {e.returncode}\nSTDERR:\n{error_details}")
            
        video = np.frombuffer(video_stream, np.uint8).reshape([-1, *self._input_size])
        
        if len(video) == 0:
            print(f"[AutoShot] WARNING: No frames were extracted from {video_fn}. Check the video file.")
            return np.array([]), np.array([]), np.array([])


        # Gọi predict_frames để xử lý các frame đã trích xuất
        predictions = self.predict_frames(video)

        print(f"Length of predictions{len(predictions)}")
        print(f"Type of predictions{type(predictions)}")
        print(f"Predictions: {predictions}")
        return (video, *predictions)

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.3):
        """
        Chuyển đổi mảng dự đoán thành các đoạn cảnh.
        *** Phương thức này được sao chép nguyên vẹn từ TransNetV2 vì logic là chung. ***
        """
        predictions = (predictions.flatten() > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
        
        return np.array(scenes, dtype=np.int32)
class TransNetV2(nn.Module):

    def __init__(self,
                 F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5,
                 use_convex_comb_reg=False,  # not supported
                 use_resnet_features=False,  # not supported
                 use_resnet_like_top=False,  # not supported
                 frame_similarity_on_last_layer=False):  # not supported
        super(TransNetV2, self).__init__()

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplemented("Some options not implemented in Pytorch version of Transnet!")

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True
        ) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(
            lookup_window=101, output_dim=128
        ) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity: output_dim += 128
        if use_color_histograms: output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, \
            "incorrect input type and/or shape"
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot

class StackedDDCNNV2(nn.Module):

    def __init__(self,
                 in_filters,
                 n_blocks,
                 filters,
                 shortcut=True,
                 use_octave_conv=False,  # not supported
                 pool_type="avg",
                 stochastic_depth_drop_prob=0.0):
        super(StackedDDCNNV2, self).__init__()

        if use_octave_conv:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")

        assert pool_type == "max" or pool_type == "avg"
        if use_octave_conv and pool_type == "max":
            print("WARN: Octave convolution was designed with average pooling, not max pooling.")

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, octave_conv=use_octave_conv,
                          activation=functional.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)
        ])
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):

    def __init__(self,
                 in_filters,
                 filters,
                 batch_norm=True,
                 activation=None,
                 octave_conv=False):  # not supported
        super(DilatedDCNNV2, self).__init__()

        if octave_conv:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")

        assert not (octave_conv and batch_norm)

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):

    def __init__(self,
                 in_filters,
                 filters,
                 dilation_rate,
                 separable=True,
                 octave=False,  # not supported
                 use_bias=True,
                 kernel_initializer=None):  # not supported
        super(Conv3DConfigurable, self).__init__()

        if octave:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")
        if kernel_initializer is not None:
            raise NotImplemented("Kernel initializers are not implemented in Pytorch version of Transnet!")

        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                              dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1),
                              dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias)
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(in_filters, filters, kernel_size=3,
                             dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias)
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):

    def __init__(self,
                 in_filters,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 stop_gradient=False,  # not supported
                 use_bias=False):
        super(FrameSimilarity, self).__init__()

        if stop_gradient:
            raise NotImplemented("Stop gradient not implemented in Pytorch version of Transnet!")

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):

    def __init__(self,
                 lookup_window=101,
                 output_dim=None):
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 384, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))

        histograms = histograms.view(batch_size, time_window, 384).float()
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities
    
class FFmpegReader:
    """
    A robust, single-pass, and THREAD-SAFE helper class to get video metadata.

    This class ensures that for any given video file path, the `ffprobe` command
    is executed only once, even when called concurrently from multiple threads.
    """
    _metadata_cache = {}
    _cache_lock = threading.Lock() # <-- Add a lock for the cache
    _gpu_checked = False      # <-- Add this class attribute
    _hw_accel_enabled = False # <-- And this one

    @classmethod
    def _check_hw_accel(cls):
        """
        Checks for an NVIDIA GPU and nvidia-smi command.
        This check is performed only once.
        """
        if cls._gpu_checked:
            return

        # Use shutil.which to find nvidia-smi in the system's PATH
        if shutil.which('nvidia-smi') is not None:
            try:
                # Run nvidia-smi to confirm it works
                subprocess.run(['nvidia-smi'], capture_output=True, check=True)
                print("[FFmpeg] NVIDIA GPU detected. Hardware acceleration will be enabled.")
                cls._hw_accel_enabled = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("[FFmpeg] nvidia-smi found but failed to run. Disabling hardware acceleration.")
                cls._hw_accel_enabled = False
        else:
            print("[FFmpeg] NVIDIA GPU not detected (nvidia-smi not found). Using CPU decoding.")
            cls._hw_accel_enabled = False
        
        cls._gpu_checked = True


    @classmethod
    def get_codec(cls, video_path: str) -> str:
        """Gets the video codec name using ffprobe."""
        # This can reuse the metadata logic for efficiency
        metadata = cls.get_metadata(video_path)
        
        # We need to run a specific ffprobe command if the codec isn't in the cache
        if 'codec_name' in metadata:
            return metadata['codec_name']

        with cls._cache_lock:
            # Double-check after acquiring lock
            if 'codec_name' in cls._metadata_cache.get(video_path, {}):
                return cls._metadata_cache[video_path]['codec_name']

            try:
                command = [
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1'
                ]
                result = subprocess.run(command + [video_path], capture_output=True, check=True, text=True)
                codec = result.stdout.strip()
                
                # Store it back in the cache
                cls._metadata_cache[video_path]['codec_name'] = codec
                return codec
            except Exception as e:
                print(f"[WARNING] Could not determine codec for {video_path}. Error: {e}")
                return "unknown"
    @classmethod
    def get_metadata(cls, video_path: str) -> dict:
        """
        Gets video metadata using a single ffprobe call and caches the result safely.
        """
        # First, check the cache without a lock for maximum performance.
        # This is safe because dictionary reads are atomic in Python.
        # If the key is already there, we avoid locking altogether.
        if video_path in cls._metadata_cache:
            return cls._metadata_cache[video_path]

        # If not in cache, acquire the lock to prevent race conditions.
        with cls._cache_lock:
            # Double-check if another thread populated the cache while we were waiting for the lock.
            if video_path in cls._metadata_cache:
                return cls._metadata_cache[video_path]

            # --- If we are here, we are the ONLY thread running ffprobe for this video path ---
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at: {video_path}")

            print(f"[FFprobe] Cache miss for {os.path.basename(video_path)}. Running ffprobe...")

            command = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration,avg_frame_rate',
                '-of', 'json'
            ]

            try:
                result = subprocess.run(command + [video_path], capture_output=True, check=True, text=True)
                info = json.loads(result.stdout)
                
                if not info.get('streams'):
                    raise RuntimeError("No video streams found in the file.")
                
                stream_info = info['streams'][0]

                # ... (rest of the parsing logic is identical to the previous answer) ...
                try:
                    fr_num, fr_den = map(int, stream_info.get('r_frame_rate', '0/1').split('/'))
                    fps = float(fr_num) / float(fr_den) if fr_den != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    fps = 0.0

                duration = float(stream_info.get('duration', 0))
                nb_frames_str = stream_info.get('nb_frames', '0')

                if nb_frames_str and nb_frames_str.isdigit() and int(nb_frames_str) > 0:
                    total_frames = int(nb_frames_str)
                elif duration > 0 and fps > 0:
                    total_frames = int(duration * fps)
                else:
                    count_command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', video_path]
                    count_result = subprocess.run(count_command, capture_output=True, check=True, text=True)
                    total_frames = int(count_result.stdout.strip())

                metadata = {
                    'width': int(stream_info['width']),
                    'height': int(stream_info['height']),
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration': duration,
                }
                
                # Store the result in the cache before releasing the lock.
                cls._metadata_cache[video_path] = metadata
                return metadata

            except FileNotFoundError:
                 raise RuntimeError("`ffprobe` command not found. Please ensure ffmpeg is installed.")
            except subprocess.CalledProcessError as e:
                error_details = e.stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"ffprobe failed for {video_path}. Error: {error_details}")
            except (KeyError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Failed to parse ffprobe JSON output for {video_path}. Error: {e}")
            
class VideoKeyframeExtractor:
    def __init__(self, transnet_weights=None, output_dir="keyframes", 
                 sample_rate=1, max_frames_per_shot=50, model="google/siglip2-base-patch16-384", base_url = "http://192.168.20.170:6660"):
        self.device = "cuda"

        print("[TransNetV2-PyTorch] Initializing model.")
        self.transnet = AutoShot()
         # Kiểm tra và tải trọng số
        if transnet_weights is None or not os.path.exists(transnet_weights):
            raise FileNotFoundError(f"PyTorch weights for TransNetV2 not found at: {transnet_weights}")
        
        # print(f"[TransNetV2-PyTorch] Loading weights from {transnet_weights}")
        # state_dict = torch.load(transnet_weights, map_location=self.device)
        # self.transnet.load_state_dict(state_dict)
        # self.transnet.to(self.device) # Chuyển model lên GPU/CPU
        # self.transnet.eval() # Đặt model ở chế độ đánh giá
        # print("[TransNetV2-PyTorch] Model loaded successfully.")

        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        # self.model = SiglipModel.from_pretrained(model, device_map="auto", attn_implementation="sdpa").eval()
        # self.processor = SiglipProcessor.from_pretrained(model)
        print("[CLIP] Model loaded successfully.")
        self.ram = RAM.load_tag_model()
        self.ocr_model = Generator(
            base_url= base_url,
            api_key=API_KEY
        )
        print("[OCR] Model loaded successfully.")
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.max_frames_per_shot = max_frames_per_shot
        # print(f"[CLIP] Using device: {self.device}")
        os.makedirs(output_dir, exist_ok=True)
        # Thread-safe counter for frame filenames
        self._frame_counter = 1
        self._counter_lock = threading.Lock()
        # THÊM KHÓA GPU NÀY:
        self._gpu_lock = threading.Lock()
        self._ffmpeg_lock = threading.Lock()
    def get_video_fps(self, video_path: str) -> float:
        """Gets video FPS using ffprobe."""
        try:
            return FFmpegReader.get_metadata(video_path)['fps']
        except RuntimeError as e:
            print(f"[ERROR] Could not get FPS for {video_path}: {e}. Defaulting to 25.0 fps.")
            return 25.0


# Giảm Ram
    @staticmethod
    def _yuv420p_to_rgb(yuv_frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Converts a raw YUV420p frame buffer to an RGB NumPy array using OpenCV.
        This is the most robust and often the fastest method.
        """
        # For YUV420p, the total height of the stacked planes is 1.5 times the frame height.
        # OpenCV's cvtColor expects the frame in this stacked format.
        yuv_height_with_padding = math.ceil(height * 1.5)
        
        # Reshape the flat byte array into the YUV image format (stacked planes).
        yuv_img = yuv_frame.reshape((yuv_height_with_padding, width))
        
        # Use OpenCV to perform the conversion.
        # COLOR_YUV2RGB_I420 is the specific code for planar YUV 4:2:0 format.
        rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)
        
        return rgb_img
    
    def _read_frames_by_abs_indices(self, video_path: str, abs_indices: List[int], for_clip: bool=True) -> List[np.ndarray]:
        """
        Reads frames by absolute indices using ffmpeg and resizes them if needed.
        """
        frames_rgb = self._read_frames_with_ffmpeg(video_path, abs_indices)
        if not for_clip:
            return frames_rgb

        resized_frames = []
        for frame_rgb in frames_rgb:
            img = Image.fromarray(frame_rgb)
            h, w = img.height, img.width
            long_edge = max(h, w)
            target = 384
            if long_edge > target:
                scale = target / long_edge
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_frames.append(np.array(img))
        return resized_frames

    def _read_single_frame_fullres(self, video_path: str, idx: int) -> np.ndarray:
        """
        Reads a single full-resolution frame using ffmpeg.
        """
        frames = self._read_frames_with_ffmpeg(video_path, [idx])
        if not frames:
            raise ValueError(f"Cannot read frame {idx} from {video_path} using ffmpeg.")
        return frames[0]
    
    def _read_frames_by_streaming(self, video_path: str, indices: List[int]) -> dict:
            """
            Reads a specific set of frames by streaming, now with robust error handling
            and diagnostics by capturing ffmpeg's stderr.
            """
            if not indices:
                return {}

            frames_to_get = sorted(list(set(indices)))
            frame_results = {}

            with self._ffmpeg_lock:
                meta = FFmpegReader.get_metadata(video_path)
                height, width = meta['height'], meta['width']
                frame_size = height * width * 3

                command = [
                    'ffmpeg', '-hide_banner', # Allow error messages
                    '-i', video_path,
                    '-pix_fmt', 'bgr24',
                    '-f', 'rawvideo',
                    'pipe:1'
                ]

                proc = None
                try:
                    # We capture stderr to see ffmpeg's internal messages
                    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    current_frame_idx = 0
                    frames_found = 0

                    while frames_found < len(frames_to_get):
                        target_frame_idx = frames_to_get[frames_found]
                        
                        bytes_to_skip = (target_frame_idx - current_frame_idx) * frame_size
                        if bytes_to_skip > 0:
                            # This is a more robust way to skip bytes from a pipe
                            skipped_bytes = 0
                            while skipped_bytes < bytes_to_skip:
                                chunk_size = min(65536, bytes_to_skip - skipped_bytes)
                                chunk = proc.stdout.read(chunk_size)
                                if not chunk: break # Stream ended while skipping
                                skipped_bytes += len(chunk)
                            if not chunk: break # Break outer loop if stream ended
                        
                        raw_frame = proc.stdout.read(frame_size)
                        if not raw_frame or len(raw_frame) < frame_size:
                            # This is the primary failure point
                            print(f"[WARNING][STREAM] Stream ended prematurely while trying to read frame {target_frame_idx}.")
                            break

                        frame_data = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3).copy()
                        frame_results[target_frame_idx] = frame_data
                        
                        current_frame_idx = target_frame_idx + 1
                        frames_found += 1

                    # --- NEW DIAGNOSTIC LOGIC ---
                    # After the loop, whether it finished or broke, check the process.
                    # Use communicate() with a short timeout to get any final stderr messages.
                    _, stderr_data = proc.communicate(timeout=5)
                    if stderr_data:
                        error_message = stderr_data.decode('utf-8', errors='ignore').strip()
                        if error_message:
                            print(f"[DIAGNOSTIC][ffmpeg stderr] The ffmpeg process reported the following:\n---\n{error_message}\n---")

                except subprocess.TimeoutExpired:
                    print("[ERROR][STREAM] ffmpeg process timed out and was killed.")
                    if proc: proc.kill()

                except Exception as e:
                    print(f"[ERROR][STREAM] An error occurred during ffmpeg streaming: {e}")
                    if proc and proc.poll() is None:
                        proc.kill()
                        proc.communicate()
                
                finally:
                    # Final check to ensure the process is terminated
                    if proc and proc.poll() is None:
                        proc.kill()
                        proc.communicate()

                return frame_results
                
    def _read_frames_downscaled_ffmpeg(self, video_path: str, indices: List[int], size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """
        PASS 1 EXTRACTOR: Ultra-fast extraction for CLIP.
        - Extracts in native YUV420p to minimize FFmpeg's work.
        - Downscales inside FFmpeg (very fast).
        - Converts YUV->RGB in Python for final use.
        """
        if not indices:
            return []

        width, height = size
        # Frame size for YUV420p is 1.5 bytes per pixel
        frame_size = int(width * height * 1.5)
        select_filter = "+".join([f"eq(n,{i})" for i in indices])

        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', video_path,
            '-vf', f"select='{select_filter}',scale={width}:{height}",
            '-pix_fmt', 'yuv420p', # <-- FASTEST: Output native format
            '-sws_flags', 'bilinear', # Use the fast bilinear scaling algorithm
            '-f', 'rawvideo',
            'pipe:1'
        ]
        
        proc = None
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            frames_rgb = []
            for _ in range(len(indices)):
                raw_frame = proc.stdout.read(frame_size)
                if not raw_frame or len(raw_frame) < frame_size:
                    break
                # Convert the raw YUV buffer to an RGB numpy array
                rgb_frame = self._yuv420p_to_rgb(np.frombuffer(raw_frame, dtype=np.uint8), width, height)
                frames_rgb.append(rgb_frame)
            return frames_rgb
        finally:
            if proc:
                proc.kill()
                proc.communicate()

                

    def _read_frames_with_ffmpeg(self, video_path: str, indices: List[int], output_dir: str) -> List[str]:
        """
        Extracts specified frames from a video and saves them as PNG files in the given output directory.
        Returns a list of file paths to the extracted frames.
        """
        if not indices:
            return []

        os.makedirs(output_dir, exist_ok=True)
        select_filter = "+".join([f"eq(n,{i})" for i in indices])

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"select='{select_filter}'",
            "-vsync", "0",
            os.path.join(output_dir, "frame_%05d.png")
        ]

        try:
            subprocess.run(command, check=True)
            # Return the sorted list of created frame paths
            return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')])
        except subprocess.CalledProcessError as e:
            print(f"[ERROR][FFMPEG] Frame extraction failed: {e}")
            return []

    def extract_video_frames(self, video_path: str) -> Tuple[List[str], List[int]]:
        """
        Extracts frames from a video at a given sample rate using ffmpeg, saving them to a temporary directory.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            meta = FFmpegReader.get_metadata(video_path)
            frame_indices = list(range(0, meta['total_frames'], self.sample_rate))
            frame_paths = self._read_frames_with_ffmpeg(video_path, frame_indices, temp_dir)
            return frame_paths, frame_indices[:len(frame_paths)]

        
    def _get_next_frame_count(self):
        with self._counter_lock:
            count = self._frame_counter
            self._frame_counter += 1
            return count

    def extract_clip_features(self, frames: List[np.ndarray], shot_id: int = None) -> np.ndarray:
        features = []
        batch_size = 16
        progress_prefix = f"[CLIP][Shot {shot_id}]" if shot_id is not None else "[CLIP]"
        target_size = (224, 224)
        # Đảm bảo chỉ MỘT luồng truy cập GPU tại một thời điểm
        with self._gpu_lock: # <-- THÊM DÒNG NÀY
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i+batch_size]

                # Convert frames to PIL Images and preprocess for CLIP
                batch_inputs = torch.stack([
                    self.preprocess(
                        Image.fromarray(cv2.cvtColor(
                            cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2RGB  # ✅ convert to RGB here
                        ))
                    )
                    for frame in batch_frames
                ]).to(self.device)


                # Extract features
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_inputs)
                    batch_features /= batch_features.norm(dim=-1, keepdim=True)  # Normalize

                features.append(batch_features.cpu().numpy())
                print(f"\r[CLIP] Processing frames {i+len(batch_frames)}/{len(frames)}", end="")
                
                # Giải phóng bộ nhớ GPU sau mỗi batch nếu có thể
                del batch_features, batch_inputs
                torch.cuda.empty_cache() 
                # print(f"\r{progress_prefix} Processing frames {i + len(batch_frames)}/{len(frames)}", end="")
            
        print(f"\r{progress_prefix} Feature extraction complete.                ")
        return np.vstack(features)

    def adaptive_clustering(self, features: np.ndarray) -> List[int]:
        n_samples = features.shape[0]
        if n_samples <= 1: return [0] if n_samples == 1 else []
        if n_samples <= 3: return list(range(n_samples))

        k_max = min(int(np.sqrt(n_samples)), 10)
        best_score, best_k, best_labels = -1, 2, None

        for k in range(2, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(features)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score, best_k, best_labels = score, k, labels
                    best_centers = kmeans.cluster_centers_

        if best_labels is None:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
            best_labels = kmeans.fit_predict(features)
            best_centers = kmeans.cluster_centers_

        keyframe_indices = []
        for i in range(len(best_centers)):
            cluster_frames = np.where(best_labels == i)[0]
            if len(cluster_frames) > 0:
                dists = np.linalg.norm(features[cluster_frames] - best_centers[i], axis=1)
                keyframe_indices.append(cluster_frames[np.argmin(dists)])

        keyframe_features = features[keyframe_indices]
        n_keyframes = len(keyframe_indices)
        if n_keyframes <= 1: return keyframe_indices

        similarity_matrix = np.dot(keyframe_features, keyframe_features.T)
        to_keep = list(range(n_keyframes))
        threshold = 0.945
        i = 0
        while i < len(to_keep):
            j = i + 1
            while j < len(to_keep):
                if similarity_matrix[to_keep[i], to_keep[j]] > threshold:
                    to_keep.pop(j)
                else: j += 1
            i += 1
        print(f"[CLIP] Reduced {n_keyframes} keyframes to {len(to_keep)} after similarity filtering.")
        return [keyframe_indices[i] for i in to_keep]

    
    def _extract_json_from_response(self, response_str: str) -> dict | None:
        """
        Safely extracts a JSON object from a string that may be wrapped in markdown or other text.
        Finds the first '{' and the last '}' to delimit the JSON object.
        """
        try:
            # Find the starting position of the JSON object
            start_index = response_str.find('{')
            # Find the ending position of the JSON object
            end_index = response_str.rfind('}')

            if start_index == -1 or end_index == -1 or end_index < start_index:
                # If no valid JSON object is found, return None
                print(f"[JSON_HELPER] Could not find a valid JSON object in the response.")
                return None

            # Extract the JSON substring
            json_str = response_str[start_index : end_index + 1]

            # Parse the extracted string
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            print(f"[JSON_HELPER] Failed to decode the extracted JSON string: {e}")
            return None
        except Exception as e:
            print(f"[JSON_HELPER] An unexpected error occurred during JSON extraction: {e}")
        return None

    #----------------------------------------RE-OCR----------------------------------------------
    @staticmethod
    def _safe_write_json(path: str, obj: dict):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=3, ensure_ascii=False)
        os.replace(tmp, path)

    @staticmethod
    def _parse_frame_range(s: str) -> tuple[int, int]:
        # "2:4" -> (2,4), inclusive. Validate.
        m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", s or "")
        if not m:
            raise ValueError(f"Invalid --frame format: {s}. Expected 'start:end' (e.g. 2:4).")
        a, b = int(m.group(1)), int(m.group(2))
        if a <= 0 or b <= 0 or a > b:
            raise ValueError(f"Invalid range {a}:{b}. Must be 1-based and start<=end.")
        return a, b


    def reocr_specific_frames(self, output_path: str, video_folder: str, frame_range: str | None):
        vdir = os.path.join(output_path, video_folder)
        meta_path = os.path.join(vdir, "metadata.json")
        if not os.path.exists(vdir) or not os.path.exists(meta_path):
            raise FileNotFoundError("Video folder hoặc metadata.json không tồn tại.")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        video_key = next(iter(meta.keys()))

        targets = []
        if frame_range:
            a, b = self._parse_frame_range(frame_range)
            for i in range(a, b + 1):
                fname = f"frame_{i:03d}"
                p = os.path.join(vdir, fname + ".png")
                if os.path.exists(p) and fname in meta[video_key]:
                    targets.append((fname, p))
                else:
                    print(f"[SKIP] {fname}.png missing or not in metadata.")
        else:
            for fname in meta[video_key].keys():
                p = os.path.join(vdir, fname + ".png")
                if os.path.exists(p):
                    targets.append((fname, p))

        if not targets:
            print("[INFO] Không có frame nào để re-OCR."); return

        if self.ocr_model is None:
            print("[ERROR] self.ocr_model chưa được khởi tạo."); return

        print(f"[PROC] {video_folder}: re-OCR {len(targets)} frame(s).")
        for fname, p in targets:
            try:
                img = Image.open(p)
                resp_str, _ = self.ocr_model.generate(
                    model_name="gemini-2.0-flash",
                    images=[img],
                    prompt=OCR_PROMPT,
                    temperature=0.0,
                )
                js = self._extract_json_from_response(resp_str)
                meta[video_key][fname]["ocr"] = (js.get("corrected_text", "") if js else "") or ""
            except Exception as e:
                print(f"[WARN] OCR failed for {fname}.png: {e}")
                meta[video_key][fname]["ocr"] = ""
            self._safe_write_json(meta_path, meta)
        print(f"[DONE] {video_folder}: re-OCR hoàn tất.")


    #-----------------------------------------RE-OCR----------------------------------------------
        
    def _post_process_batch_concurrently(self, batch_data, tag_model, ocr_model, caption_model=None):
        """
        Xử lý đồng thời một lô (batch) các keyframe.
        - OCR được thực hiện trong một lệnh gọi API duy nhất cho cả lô.
        - Tagging được thực hiện tuần tự cho từng ảnh trong lô.
        """
        if not batch_data:
            return []

        keyframe_paths, images = zip(*batch_data)
        batch_size = len(keyframe_paths)
        
        tags_results = [[] for _ in range(batch_size)]
        ocr_results = ["" for _ in range(batch_size)]
        captions = ["" for _ in range(batch_size)]
        # 1. <<< SỬA LỖI >>>: Thực hiện OCR theo lô với định dạng input đúng và phân tích JSON output
        # 1. OCR: chạy TUẦN TỰ TỪNG ẢNH trong batch (không gửi cả batch một lần nữa)
        if ocr_model and images:
            try:
                print(f"[PostProc] Running OCR sequentially on {batch_size} frames...")
                for i, img in enumerate(images):
                    try:
                        # Giữ nguyên ảnh, KHÔNG crop
                        response_str, _ = ocr_model.generate(
                            model_name="gemini-2.0-flash",
                            images=[img],          # gửi 1 ảnh/lần
                            prompt=OCR_PROMPT,
                            temperature=0.0,
                        )
                        response_json = self._extract_json_from_response(response_str)
                        valid = response_json.get("corrected_text", "")
                        ocr_results[i] = valid or ""

                    except Exception as ee:
                        print(f"[WARNING] OCR failed for index {i}: {ee}")
                        ocr_results[i] = ""

                print(f"[PostProc] OCR sequential completed for {batch_size} frames.")

            except Exception as e:
                print(f"[WARNING] OCR batch wrapper failed: {e}. All OCR results for this batch will be empty.")
                # ocr_results đã khởi tạo rỗng

                
        if caption_model:
            try:
                print(f"[PostProc] Submitting batch of {batch_size} frames for captions...")
                # Convert tuple 'images' to a list for the API call
                with open('source/prompt_caption.txt', 'r') as f:
                    prompt_caption = f.read().strip()

                response_str, _ = caption_model.generate(
                    model_name="pixtral-large-latest", # Using a reliable model
                    images=list(images), 
                    prompt=prompt_caption,
                    temperature=0.2,
                )
                
                # Robustly parse the JSON response
                response_json = self._extract_json_from_response(response_str)

                if response_json and "captions" in response_json:
                    raw_captions = response_json.get("captions", [])
                    print(f"[PostProc] Successfully parsed {len(raw_captions)} captions from response.")
                    
                    # Ensure the final captions list matches the batch size
                    num_returned = len(raw_captions)
                    for i in range(batch_size):
                        if i < num_returned and raw_captions[i]:
                            captions[i] = raw_captions[i].strip()
                        else:
                            captions[i] = "" # Fill with empty string if not provided or empty

                    if num_returned != batch_size:
                        print(f"[WARNING] Captioning returned {num_returned} results for a batch of {batch_size}. Padding/truncating as needed.")
                
                else:
                    # This branch handles cases where parsing failed or the 'captions' key is missing
                    print(f"[WARNING] Captioning response could not be parsed or was invalid. Using empty captions. Full response: {response_str}")
                    # 'captions' is already initialized to empty strings, so no action needed

            except Exception as e:
                # This is a fallback for generate() call failures (e.g., network error)
                print(f"[WARNING] Captioning batch processing failed with an API/network error: {e}. All captions for this batch will be empty.")
        # 2. Thực hiện Tagging cho từng ảnh trong lô
        if tag_model:
            # with self._gpu_lock:
            for i in range(batch_size):
                try:
                    tags = RAM.get_tag(keyframe_paths[i], tag_model)
                    tags_results[i] = tags
                except Exception as e:
                    print(f"[WARNING] Tagging failed for {os.path.basename(keyframe_paths[i])}: {e}")

        # 3. Kết hợp kết quả và trả về
        final_results = []
        for i in range(batch_size):
            final_results.append((keyframe_paths[i], tags_results[i], ocr_results[i], captions[i]))
            
        return final_results

    def _process_shot_concurrently(self, video_path: str, shot_task_data: dict) -> Tuple:
        """
        Processes a single shot by extracting its downscaled frames and finding keyframes.
        """
        shot_idx = shot_task_data["shot_idx"]
        abs_indices = shot_task_data["abs_indices"]

        # 1. FETCH DOWNSCALED DATA
        try:
            # Use the new function to get small 224x224 frames
            downscaled_frames = self._read_frames_downscaled_ffmpeg(video_path, abs_indices, size=(384, 384))
        except Exception as e:
            print(f"[Worker][Shot {shot_idx+1}] Failed to extract downscaled frames: {e}")
            return shot_idx, []

        if not downscaled_frames:
            return shot_idx, []

        # 2. RUN CLIP & CLUSTERING (on small frames)
        # Note: The CLIP preprocessor will still run, but it won't have to do much resizing.
        shot_features = self.extract_clip_features(downscaled_frames, shot_id=shot_idx + 1)
        keyframe_indices_in_shot = self.adaptive_clustering(shot_features)

        # 3. RETURN FINAL INDICES
        # Map the indices from the small list back to the original absolute frame numbers.
        final_abs_indices = [abs_indices[i] for i in keyframe_indices_in_shot]

        return shot_idx, final_abs_indices
        #-------------------End of Ram Modified Code-------------------
    
    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.3): # Tôi đã đổi threshold mặc định thành 0.3 cho giống pipeline của bạn
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)

    def _predict_frames_pytorch(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Xử lý các frame đã được trích xuất bằng model TransNetV2 PyTorch.
            Hàm này mô phỏng lại logic cửa sổ trượt của phiên bản gốc.
            """
            # Chuyển model sang chế độ đánh giá
            self.transnet.eval()
            
            predictions = []
            
            # Logic cửa sổ trượt (sliding window) - được giữ nguyên từ phiên bản TensorFlow
            def input_iterator():
                no_padded_frames_start = 25
                no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
                start_frame = np.expand_dims(frames[0], 0)
                end_frame = np.expand_dims(frames[-1], 0)
                padded_inputs = np.concatenate(
                    [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
                )
                ptr = 0
                while ptr + 100 <= len(padded_inputs):
                    out = padded_inputs[ptr:ptr + 100]
                    ptr += 50
                    yield out[np.newaxis]

            for batch_np in input_iterator():
                # Chuyển đổi NumPy sang Tensor, đưa lên GPU
                batch_torch = torch.from_numpy(batch_np).to(self.device)
                
                with torch.no_grad(): # Quan trọng: Vô hiệu hóa tính toán gradient để tiết kiệm bộ nhớ và tăng tốc
                    # Forward pass
                    single_frame_pred_logits, all_frames_pred_logits_dict = self.transnet(batch_torch)
                    
                    # Áp dụng sigmoid để có được xác suất
                    single_frame_pred = torch.sigmoid(single_frame_pred_logits)
                    all_frames_pred = torch.sigmoid(all_frames_pred_logits_dict["many_hot"])

                # Chuyển kết quả về CPU, sang NumPy và cắt lấy phần dự đoán đáng tin cậy (25:75)
                # .detach() để tách khỏi computational graph
                predictions.append((
                    single_frame_pred.detach().cpu().numpy()[0, 25:75, 0],
                    all_frames_pred.detach().cpu().numpy()[0, 25:75, 0]
                ))

                print(f"\r[TransNetV2-PyTorch] Processing video frames {min(len(predictions) * 50, len(frames))}/{len(frames)}", end="")
            print("")

            single_frame_pred_np = np.concatenate([p[0] for p in predictions])
            all_frames_pred_np = np.concatenate([p[1] for p in predictions])

            return single_frame_pred_np[:len(frames)], all_frames_pred_np[:len(frames)]

    def _predict_video_pytorch(self, video_fn: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts frames from a video for TransNetV2 and runs predictions,
        using GPU hardware acceleration for decoding where possible.
        """
        print(f"[TransNetV2-PyTorch] Extracting frames from {video_fn} using ffmpeg CLI")
        codec = FFmpegReader.get_codec(video_fn)
        print(f"[FFmpeg] Detected video codec: {codec}")
        command = ['ffmpeg']

        # Add hardware acceleration arguments if enabled and supported
        if FFmpegReader._hw_accel_enabled:
            codec = FFmpegReader.get_codec(video_fn)
            decoder_map = {
                'h264': 'h264_cuvid', 'hevc': 'hevc_cuvid', 'av1': 'av1_cuvid',
                'vp9': 'vp9_cuvid', 'mpeg2video': 'mpeg2_cuvid', 'vc1': 'vc1_cuvid',
            }
            if codec in decoder_map:
                print(f"[TransNetV2] Using GPU acceleration for '{codec}' codec.")
                command.extend(['-hwaccel', 'cuda', '-c:v', decoder_map[codec]])
            else:
                print(f"[TransNetV2] Codec '{codec}' not supported for GPU acceleration. Falling back to CPU.")

        # Add the rest of the ffmpeg arguments
        command.extend([
            '-i', video_fn,
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-sws_flags', 'bilinear',
            '-s', '48x27',
            'pipe:1'
        ])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=True # Use check=True to automatically raise CalledProcessError on failure
            )
            video_stream = result.stdout
        except FileNotFoundError:
            raise RuntimeError("[FFmpeg] 'ffmpeg' command not found. Please ensure ffmpeg is installed.")
        except subprocess.CalledProcessError as e:
            error_details = e.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"[FFmpeg] Failed to decode {video_fn} for TransNetV2.\n"
                                f"FFmpeg Command: {' '.join(command)}\n"
                                f"Exit Code: {e.returncode}\nSTDERR:\n{error_details}")

        video_frames = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

        single_pred, all_pred = self._predict_frames_pytorch(video_frames)
        return video_frames, single_pred, all_pred
        
    ### THAY ĐỔI KẾT THÚC ###

    def _validate_and_rerun_missing_frames(
        self,
        video_output_dir: str,
        metadata: dict,
        # <<< CHANGE: `all_keyframes_info` is no longer needed as an argument
        # We pass fps, and the models for the re-run logic.
        fps: float,
        tag_model,
        ocr_model,
        caption_model
    ) -> dict:
        """
        Audits the output directory and metadata for consistency, fixing any issues found.
        - Deletes metadata entries that point to non-existent image files.
        - Identifies image files that are missing from the metadata.
        - Finds frames with incomplete metadata (e.g., empty OCR) and re-runs post-processing.
        """
        print("\n[VALIDATION] Starting comprehensive audit of output directory and metadata...")
        if not metadata or not list(metadata.keys()):
            print("[VALIDATION] Metadata is empty. Nothing to validate.")
            return metadata
            
        video_name = list(metadata.keys())[0]

        # --- Step 1: Gather Ground Truth from Disk and Metadata ---
        try:
            # Get all 'frame_xxx' names from saved .png files
            saved_files = {
                os.path.splitext(f)[0] for f in os.listdir(video_output_dir) if f.startswith('shot') and f.endswith('.png')
            }
        except FileNotFoundError:
            print("[VALIDATION] ERROR: Output directory not found. Aborting validation.")
            return metadata

        # Get all 'frame_xxx' names from the metadata dictionary
        frames_in_metadata = set(metadata[video_name].keys())

        # --- Step 2: Find and Fix Discrepancies ---

        # A. Find metadata entries without a corresponding image file ("zombie" entries)
        missing_images = frames_in_metadata - saved_files
        if missing_images:
            # <<< CHANGED: This block no longer deletes entries. It only warns you.
            print(f"[VALIDATION] WARNING: Found {len(missing_images)} metadata entries with no matching image file.")
            # We can print a few examples to help with debugging
            for i, frame_name in enumerate(list(missing_images)):
                if i < 5: # Print up to 5 examples
                    print(f"  - Example of missing file for metadata entry: {frame_name}.png")
                elif i == 5:
                    print("  - ... (and possibly more)")
                    break
            # Update our set of valid metadata keys after deletion
            frames_in_metadata -= missing_images

        # B. Find image files without metadata ("orphaned" images)
        missing_metadata = saved_files - frames_in_metadata
        if missing_metadata:
            print(f"[VALIDATION] WARNING: Found {len(missing_metadata)} orphaned image files with no metadata.")
            for frame_name in sorted(list(missing_metadata)):
                # We can't fix these as we don't know their original frame index, so we just warn.
                print(f"  - Orphaned file: {frame_name}.png")

        # --- Step 3: Find and Re-process Incomplete Frames ---
        frames_to_rerun = []
        # We only check frames that are confirmed to exist both on disk and in metadata
        valid_frames_to_check = sorted(list(frames_in_metadata.intersection(saved_files)))
        
        for frame_name in valid_frames_to_check:
            entry = metadata[video_name].get(frame_name, {})
            # Check if tags are missing/empty OR if ocr text is missing (None) or empty ("")
            is_incomplete = not entry.get("tags") or entry.get("ocr") is None
            # Add other checks as needed, e.g., or not entry.get("caption")
            
            if is_incomplete:
                frames_to_rerun.append(frame_name)
                
        if not frames_to_rerun:
            print("[VALIDATION] All valid frames have complete metadata. No re-run needed.")
            print("[VALIDATION] Audit complete.")
            return metadata

        print(f"[VALIDATION] Found {len(frames_to_rerun)} frames with incomplete data. Preparing re-run...")
        # print(f"  - Frames to fix: {frames_to_rerun}")

        # --- Step 4: Execute the Re-run ---
        rerun_batch_data = []
        for frame_name in frames_to_rerun:
            keyframe_path = os.path.join(video_output_dir, f"{frame_name}.png")
            try:
                # We must re-load the image for the post-processing function
                image = Image.open(keyframe_path)
                rerun_batch_data.append((keyframe_path, image))
            except Exception as e:
                print(f"[VALIDATION] WARNING: Could not load image {keyframe_path} for re-run: {e}")

        if not rerun_batch_data:
            print("[VALIDATION] No valid images could be loaded for re-run. Aborting.")
            print("[VALIDATION] Audit complete.")
            return metadata

        print(f"[VALIDATION] Submitting a batch of {len(rerun_batch_data)} frames for re-processing...")
        try:
            rerun_results = self._post_process_batch_concurrently(
                rerun_batch_data, tag_model, ocr_model, caption_model
            )
        except Exception as e:
            print(f"[VALIDATION] Rerun failed with a critical error: {e}. Returning original metadata.")
            return metadata

        # --- Step 5: Integrate the Fresh Results ---
        successful_updates = 0
        for keyframe_path, tags, ocr_text, caption in rerun_results:
            frame_name = os.path.splitext(os.path.basename(keyframe_path))[0]
            
            # We UPDATE the existing entry, which is much safer.
            if frame_name in metadata[video_name]:
                # Only update if the new data is actually better
                if not metadata[video_name][frame_name].get("tags") and tags:
                    metadata[video_name][frame_name]["tags"] = tags
                if not metadata[video_name][frame_name].get("ocr") and ocr_text:
                    metadata[video_name][frame_name]["ocr"] = ocr_text
                # Add caption logic if used
                # if not metadata[video_name][frame_name].get("caption") and caption:
                #      metadata[video_name][frame_name]["caption"] = caption
                successful_updates += 1
            else:
                print(f"[VALIDATION] WARNING: Re-processed frame '{frame_name}' not found in original metadata to update.")

        print(f"[VALIDATION] Successfully updated metadata for {successful_updates}/{len(rerun_results)} re-processed frames.")
        print("[VALIDATION] Audit complete.")
        
        return metadata
    @staticmethod
    def _build_shot_tasks_optimized(scenes, all_frame_indices, max_frames_per_shot):
        """
        Builds shot tasks in a single pass with O(N+M) complexity.
        N = number of frames, M = number of shots.
        """
        if not scenes.any() or not all_frame_indices:
            return [], set()

        shot_frames_map = [[] for _ in range(len(scenes))]
        all_required_abs_indices = set()
        shot_idx = 0

        for frame_pos_in_list, abs_frame_idx in enumerate(all_frame_indices):
            while shot_idx < len(scenes) and abs_frame_idx > scenes[shot_idx][1]:
                shot_idx += 1
            if shot_idx >= len(scenes):
                break
            
            start_idx, end_idx = scenes[shot_idx]
            if start_idx <= abs_frame_idx <= end_idx:
                shot_frames_map[shot_idx].append(frame_pos_in_list)

        all_shot_tasks = []
        for shot_idx, rel_positions in enumerate(shot_frames_map):
            if not rel_positions:
                continue
            
            if len(rel_positions) > max_frames_per_shot:
                step = max(1, len(rel_positions) // max_frames_per_shot)
                rel_positions = rel_positions[::step]
            
            shot_abs_indices = [all_frame_indices[i] for i in rel_positions]
            
            all_shot_tasks.append({
                "shot_idx": shot_idx,
                "abs_indices": shot_abs_indices
            })
            all_required_abs_indices.update(shot_abs_indices)
            
        return all_shot_tasks, all_required_abs_indices
    # <<< MAIN METHOD REFACTORED FOR STREAMING PIPELINE >>>
    def _process_shot_on_loaded_frames(self, shot_data: Tuple) -> Tuple[int, List[int]]:
        shot_idx, scene, all_frame_paths, all_frame_indices, is_last_shot = shot_data
        start_frame, end_frame = scene

        if is_last_shot:
            rel_positions = [
                i for i, fidx in enumerate(all_frame_indices) 
                if start_frame <= fidx <= end_frame
            ]
        else:
            rel_positions = [
                i for i, fidx in enumerate(all_frame_indices) 
                if start_frame <= fidx < end_frame
            ]

        if not rel_positions:
            return shot_idx, []

        if len(rel_positions) > self.max_frames_per_shot:
            step = max(1, len(rel_positions) // self.max_frames_per_shot)
            sampled_rel_positions = rel_positions[::step]
        else:
            sampled_rel_positions = rel_positions

        if not sampled_rel_positions:
            return shot_idx, []

        sampled_shot_frames = [cv2.imread(all_frame_paths[i]) for i in sampled_rel_positions]

        shot_features = self.extract_clip_features(sampled_shot_frames, shot_id=shot_idx + 1)
        keyframe_indices_in_shot = self.adaptive_clustering(shot_features) 

        final_absolute_indices = [
            all_frame_indices[sampled_rel_positions[local_idx]]
            for local_idx in keyframe_indices_in_shot
        ]

        return shot_idx, final_absolute_indices
    
    def _identify_and_extract_shot_keyframes(self, video_path: str, shot_task_data: dict) -> List[Tuple[np.ndarray, int]]:
        """
        Identifies keyframes for a single shot, then immediately extracts them at full resolution.
        This function is a self-contained task for a worker thread in the streaming pipeline.
        Returns a list of tuples, where each tuple is (full_res_frame_data, original_frame_index).
        """
        shot_idx = shot_task_data["shot_idx"]
        abs_indices = shot_task_data["abs_indices"]

        # --- Keyframe Identification Logic (formerly PASS 1) ---
        try:
            # Use the fast downscaled reader
            downscaled_frames = self._read_frames_downscaled_ffmpeg(video_path, abs_indices, size=(384, 384))
        except Exception as e:
            print(f"[Worker][Shot {shot_idx+1}] Failed to extract downscaled frames: {e}")
            return []

        if not downscaled_frames:
            return []

        # --- Run SigLIP & Clustering (on small frames) ---
        # This part will be serialized by the _gpu_lock, which is expected.
        shot_features = self.extract_clip_features(downscaled_frames, shot_id=shot_idx + 1)
        keyframe_indices_in_shot = self.adaptive_clustering(shot_features)
        
        # Map local indices back to the original absolute frame numbers
        final_abs_indices = sorted([abs_indices[i] for i in keyframe_indices_in_shot])

        if not final_abs_indices:
            print(f"[Worker][Shot {shot_idx+1}] No keyframes found after clustering.")
            return []

        # --- Full-Resolution Extraction Logic (formerly PASS 2) ---
        # Immediately fetch the high-quality frames now that we know which ones we need.
        try:
            full_res_frames = self._read_frames_with_ffmpeg(video_path, final_abs_indices)
            # Pair up the frame data with its original index to return
            return list(zip(full_res_frames, final_abs_indices))
        except Exception as e:
            print(f"[Worker][Shot {shot_idx+1}] Failed to extract full-resolution frames: {e}")
            return []
        

    def extract_keyframes(self, video_path: str) -> None:
        """
        A true producer-consumer pipeline for keyframe extraction with disk-based frame caching.
        """
        self._frame_counter = 1
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)

        if os.path.exists(video_output_dir):
            shutil.rmtree(video_output_dir)
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"\n[PIPELINE] Starting processing for: {video_path}")

        cache_dir = tempfile.mkdtemp()
        try:
            fps = self.get_video_fps(video_path)
            _, single_frame_predictions, _ = self.transnet.predict_video(video_path)
            scenes_in_prediction_space = self.transnet.predictions_to_scenes(single_frame_predictions)
            
            total_original_frames = FFmpegReader.get_metadata(video_path)['total_frames']
            total_prediction_frames = len(single_frame_predictions)
            scale_factor = total_original_frames / total_prediction_frames
            print(f"[PIPELINE] Video FPS: {fps:.2f}, Scale factor: {scale_factor:.4f}")
            scenes = []
            for start_pred, end_pred in scenes_in_prediction_space:
                original_start = int(start_pred * scale_factor)
                original_end = int(end_pred * scale_factor)
                if scenes and original_start < scenes[-1][1]:
                    original_start = scenes[-1][1]
                if original_end > original_start:
                    scenes.append([original_start, original_end])
            scenes = np.array(scenes, dtype=np.int32)
            print(f"[PIPELINE] Scaled to {len(scenes)} shots in original video space.")

            all_frame_indices = list(range(0, total_original_frames, self.sample_rate))
            all_frame_paths = self._read_frames_with_ffmpeg(video_path, all_frame_indices, cache_dir)
            all_frame_indices = all_frame_indices[:len(all_frame_paths)]
            print(f"[PIPELINE] Cached {len(all_frame_paths)} frames to disk.")

            frame_index_to_path = {frame_idx: path for frame_idx, path in zip(all_frame_indices, all_frame_paths)}

            all_shot_keyframe_indices = {}
            num_cpu_workers = 16
            print(f"[PIPELINE] Using {num_cpu_workers} workers to find all keyframe indices...")

            with ThreadPoolExecutor(max_workers=num_cpu_workers, thread_name_prefix='ShotProc') as shot_executor:
                num_scenes = len(scenes)
                shot_futures = {
                    shot_executor.submit(
                        self._process_shot_on_loaded_frames,
                        (shot_idx, scene, all_frame_paths, all_frame_indices, shot_idx == num_scenes - 1)
                    ): shot_idx
                    for shot_idx, scene in enumerate(scenes)
                }

                for future in as_completed(shot_futures):
                    shot_idx = shot_futures[future]
                    try:
                        s_idx, original_indices = future.result()
                        if original_indices:
                            all_shot_keyframe_indices[s_idx] = sorted(original_indices)
                            print(f"\r[PIPELINE] Producer for shot {s_idx+1} finished, found {len(original_indices)} keyframes.", end="")
                    except Exception as exc:
                        print(f'\n[ERROR] Shot producer {shot_idx + 1} generated an exception: {exc}')
            
            print("\n[PIPELINE] All shot producers finished.")
            print(f"[PIPELINE] Total keyframes identified across all shots: {sum(len(v) for v in all_shot_keyframe_indices.values())}")

            all_unique_indices_to_fetch = set()
            for shot_idx in sorted(all_shot_keyframe_indices.keys()):
                for frame_idx in all_shot_keyframe_indices[shot_idx]:
                    all_unique_indices_to_fetch.add(frame_idx)
            
            if not all_unique_indices_to_fetch:
                print("[PIPELINE] No keyframes found for the entire video. Exiting.")
                return

            tag_model = self.ram
            ocr_model = self.ocr_model
            caption_model = None
            
            metadata = {video_name: {}}
            metadata_stubs = {}
            all_keyframes_info = []
            num_post_proc_workers = 12
            
            with ThreadPoolExecutor(max_workers=num_post_proc_workers, thread_name_prefix='PostProc') as post_proc_executor:
                keyframe_batch_buffer = []
                post_processing_futures = []

                for shot_idx in sorted(all_shot_keyframe_indices.keys()):
                    for keyframe_idx_in_shot, original_frame_idx in enumerate(all_shot_keyframe_indices[shot_idx]):
                        
                        cached_frame_path = frame_index_to_path.get(original_frame_idx)
                        if cached_frame_path is None:
                            print(f"[WARNING] Could not find frame {original_frame_idx} in cache. Skipping.")
                            continue
                        
                        image = Image.open(cached_frame_path)
                        
                        frame_name_for_file = f"shot{shot_idx+1:03d}_frame{original_frame_idx:05d}"
                        keyframe_filename = f"{frame_name_for_file}.png"
                        keyframe_path = os.path.join(video_output_dir, keyframe_filename)
                        
                        timestamp = f"{(original_frame_idx / fps) // 60:02.0f}:{(original_frame_idx / fps) % 60:06.3f}"
                        metadata_stubs[keyframe_path] = {
                            "frame_name": frame_name_for_file,
                            "id": original_frame_idx,
                            "time-stamp": timestamp,
                            "shot": shot_idx + 1,
                        }

                        image.save(keyframe_path, 'PNG')
                        keyframe_batch_buffer.append((keyframe_path, image))
                        all_keyframes_info.append((keyframe_path, shot_idx, keyframe_idx_in_shot, original_frame_idx))

                        if len(keyframe_batch_buffer) >= OCR_BATCH_SIZE:
                            task_future = post_proc_executor.submit(
                                self._post_process_batch_concurrently, keyframe_batch_buffer, tag_model, ocr_model, caption_model)
                            post_processing_futures.append(task_future)
                            keyframe_batch_buffer = []

                if keyframe_batch_buffer:
                    task_future = post_proc_executor.submit(
                        self._post_process_batch_concurrently, keyframe_batch_buffer, tag_model, ocr_model, caption_model)
                    post_processing_futures.append(task_future)

                print(f"\n[PIPELINE] All shots processed. Aggregating results from {len(post_processing_futures)} post-processing batches...")
                for future in as_completed(post_processing_futures):
                    try:
                        batch_results = future.result()
                        if not batch_results: continue

                        for keyframe_path, tags, ocr_text, caption in batch_results:
                            if keyframe_path in metadata_stubs:
                                stub = metadata_stubs[keyframe_path]
                                frame_name = stub.pop("frame_name")
                                metadata[video_name][frame_name] = {
                                    **stub, "tags": tags, "ocr": ocr_text, "caption": caption
                                }
                            else:
                                print(f"[WARNING] Could not find metadata stub for processed keyframe: {keyframe_path}")
                    except Exception as exc:
                        print(f'[ERROR] A post-processing batch generated an exception: {exc}')

            metadata = self._validate_and_rerun_missing_frames(
                video_output_dir, metadata, fps, tag_model, ocr_model, caption_model
            )
            
            sorted_frames = sorted(
                metadata[video_name].items(), 
                key=lambda item: int(item[0].split('_frame')[1])
            )
            metadata[video_name] = dict(sorted_frames)

            metadata_path = os.path.join(video_output_dir, "metadata.json")
            with open(metadata_path, "w", encoding='utf-8') as f:
                json.dump(metadata, f, indent=3, ensure_ascii=False)
            print(f"[PIPELINE] Metadata saved to {metadata_path}")
            
            all_keyframes_info.sort(key=lambda x: (x[1], x[3]))
            with open(os.path.join(video_output_dir, "keyframes_summary.txt"), "w") as f:
                f.write(f"Video: {os.path.basename(video_path)}\nTotal shots: {len(scenes)}\nTotal keyframes: {len(all_keyframes_info)}\n\n")
                for path, s_idx, k_idx, o_idx in all_keyframes_info:
                    f.write(f"{os.path.basename(path)}\n")
        finally:
            shutil.rmtree(cache_dir)
            print(f"[PIPELINE] Removed temporary cache directory: {cache_dir}")
            
def re_ocr_subcommand(output_path: str,
                      video_folder: str,
                      frame_range: str | None,
                      ocr_base_url: str = "http://0.0.0.0:9600"):
    # OCR Generator giống pipeline
    GEMINI_SHEET_URL = os.getenv(
        "GEMINI_SHEET_URL",
        "https://docs.google.com/spreadsheets/d/1gqlLToS3OXPA-CvfgXRnZ1A6n32eXMTkXz4ghqZxe2I/export?format=csv&gid=0",
    )
    api_key = GeminiApiKeyManager(sheet_url=GEMINI_SHEET_URL).get_active_key_count()
    ocr_model = Generator(base_url=ocr_base_url, api_key=api_key)

    vdir = os.path.join(output_path, video_folder)
    meta_path = os.path.join(vdir, "metadata.json")
    if not os.path.exists(vdir):
        raise FileNotFoundError(f"Video folder not found: {vdir}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in: {vdir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    video_key = next(iter(meta.keys()))

    # chọn frame
    targets = []
    if frame_range:
        m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", frame_range or "")
        if not m:
            raise ValueError(f"Invalid --frame format: {frame_range}. Expected 'start:end'")
        a, b = int(m.group(1)), int(m.group(2))
        if a <= 0 or b <= 0 or a > b:
            raise ValueError(f"Invalid range {a}:{b}. Must be 1-based and start<=end.")
        for i in range(a, b + 1):
            fname = f"frame_{i:03d}"
            imgp = os.path.join(vdir, fname + ".png")
            if not os.path.exists(imgp):
                print(f"[SKIP] {video_folder}: {fname}.png không tồn tại.")
                continue
            if fname not in meta[video_key]:
                print(f"[WARN] {video_folder}: {fname} không có trong metadata.json, bỏ qua.")
                continue
            targets.append((fname, imgp))
    else:
        for fname in meta[video_key].keys():
            imgp = os.path.join(vdir, fname + ".png")
            if os.path.exists(imgp):
                targets.append((fname, imgp))
            else:
                print(f"[WARN] Missing image file for {fname}.png; skipping.")

    if not targets:
        print("[INFO] Không có frame nào để re-OCR.")
        return

    print(f"[PROC] {video_folder}: re-OCR {len(targets)} frame(s).")
    fixed, empty = 0, 0
    t0 = time.time()

    for fname, imgp in targets:
        try:
            img = Image.open(imgp)
            resp_str, _ = ocr_model.generate(
                model_name="gemini-2.0-flash",
                images=[img],
                prompt=OCR_PROMPT,
                temperature=0.0,
            )
            # tái dùng extractor JSON như trong class:
            # code nhanh gọn cục bộ:
            si, sj = resp_str.find('{'), resp_str.rfind('}')
            js = json.loads(resp_str[si:sj+1]) if (si != -1 and sj != -1 and sj > si) else None
            text = (js.get("corrected_text", "") if js else "") or ""
        except Exception as e:
            print(f"[WARN] OCR failed for {fname}.png: {e}")
            text = ""

        meta[video_key][fname]["ocr"] = text
        fixed += 1
        if text == "":
            empty += 1

        if fixed % 10 == 0:
            tmp = meta_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=3, ensure_ascii=False)
            os.replace(tmp, meta_path)

    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=3, ensure_ascii=False)
    os.replace(tmp, meta_path)

    print(f"[DONE] {video_folder}: fixed={fixed}, still_empty={empty}, time={time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Keyframe pipeline + utilities")
    subparsers = parser.add_subparsers(dest="cmd")

    # re-ocr subcommand
    p_reocr = subparsers.add_parser("re-ocr", help="Re-run OCR for frames in a video folder")
    p_reocr.add_argument("--output_path", required=True, help="Thư mục tổng (cha của các folder video)")
    p_reocr.add_argument("--video-folder", required=True, help="Tên folder video, ví dụ L01_V001")
    p_reocr.add_argument("--frame", help="Range 'start:end' 1-based, vd 2:4 (bỏ qua để re-OCR toàn bộ).")
    p_reocr.add_argument("--ocr-base-url", default="192.168.20.170:6660", help="Base URL cho OCR server")

    # extract subcommand (pipeline gốc)
    p_ext = subparsers.add_parser("extract", help="Run full keyframe extraction pipeline")
    p_ext.add_argument("video_path", type=str, help="Path to the video file or directory of videos")
    p_ext.add_argument("--output", type=str, default="keyframes", help="Output directory for keyframes")
    p_ext.add_argument("--sample-rate", type=int, default=3, help="Sample every N frames to reduce computation")
    p_ext.add_argument("--max-frames", type=int, default=55, help="Maximum number of frames to process per shot")

    # nếu muốn giữ tương thích cũ (không ghi 'extract'), map về extract:
    if len(sys.argv) > 1 and sys.argv[1] not in ("re-ocr", "extract"):
        # inject 'extract' trước argv để parse cũ vẫn chạy
        sys.argv.insert(1, "extract")

    args = parser.parse_args()

    if args.cmd == "re-ocr":
        re_ocr_subcommand(
            output_path=args.output_path,
            video_folder=args.video_folder,
            frame_range=args.frame,
            ocr_base_url=args.ocr_base_url,
        )
        return

    # extract (mặc định)
    start = time.time()
    extractor = VideoKeyframeExtractor(
        transnet_weights="transnetv2-pytorch-weights.pth",
        output_dir=args.output,
        sample_rate=args.sample_rate,
        max_frames_per_shot=args.max_frames
    )
    extractor.extract_keyframes(args.video_path)
    print(f"Total time taken for keyframe extraction: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()


    # python infer_concurent.py /workspace/WorkingSpace/Personal/chinhnm/data/L01_V002.mp4 --output /workspace/WorkingSpace/Personal/chinhnm/dmdm_demo
