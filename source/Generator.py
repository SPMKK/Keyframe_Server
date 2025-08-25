from openai import OpenAI
import re 
# from gemini_mistral_server import MISTRAL_SHEET_URL, GEMINI_SHEET_URL
import requests
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import base64
import mimetypes
from io import BytesIO
from typing_extensions import TypedDict, Literal    
# Define a type for the message structure used in the API
from typing import Union, List, Dict, Any, Optional, Tuple
import json
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[Dict[str, Any]]]
import contextlib
import m3u8
from pydub import AudioSegment # Add this new import
import subprocess
import tempfile
import os
import math
import time
import asyncio
import concurrent.futures
import functools
import mimetypes
import unicodedata
from pydub.silence import split_on_silence
from pydub.silence import detect_nonsilent
from io import BytesIO
def remove_vietnamese_tones(text):
    # Normalize Unicode (NFD = tách ký tự và dấu)
    text = unicodedata.normalize('NFD', text)
    # Loại bỏ các ký tự dạng dấu (combining diacritical marks)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    # Convert về dạng chuẩn NFC nếu cần (tùy use-case)
    return text
def _format_time(seconds: float) -> str:
    """Formats seconds into HH:MM:SS string."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _wav_bytes_per_sec(sr: int = 16000, ch: int = 1, bits: int = 16) -> int:
    return int(sr * ch * (bits // 8))

def _sec_from_mb_pcm(max_chunk_mb: float, bytes_per_sec: int, safety: float = 0.9) -> float:
    # Ước lượng thời lượng theo băng thông PCM (chuẩn hóa về 16k/mono/16-bit trước khi nén FLAC)
    return (max_chunk_mb * 1024 * 1024 * safety) / bytes_per_sec

def _export_flac_to_bytes(seg, name: str):
    buf = BytesIO()
    # pydub/ffmpeg sẽ nén FLAC lossless
    seg.export(buf, format="flac")
    return {"bytes": buf.getvalue(), "name": name}
def _parse_time_key_to_seconds(time_key: str) -> Tuple[float, float]:
    """
    Parses a 'HH:MM:SS-HH:MM:SS' string back into a tuple of (start_sec, end_sec).
    Returns (0.0, 0.0) on failure.
    """
    try:
        start_str, end_str = time_key.split('-')
        s_h, s_m, s_s = map(int, start_str.split(':'))
        e_h, e_m, e_s = map(int, end_str.split(':'))
        start_sec = float(s_h * 3600 + s_m * 60 + s_s)
        end_sec = float(e_h * 3600 + e_m * 60 + e_s)
        return start_sec, end_sec
    except Exception:
        return 0.0, 0.0
    
def chunk_audio_for_gemini_fast_flac(
    audio_path: str,
    max_chunk_mb: float = 18,
    min_chunk_sec: float = 1.0,
    min_loudness_dbfs: float = -50.0,
    min_silence_len_ms: int = 700,
    keep_silence_ms: int = 300,
    join_gap_ms: int = 250,
    target_sr: int = 16000,
    target_channels: int = 1,
    target_bits: int = 16,
    parallel_exports: int = 8
):
    """
    Chunk nhanh cho transcription:
    - Chuẩn hóa audio về mono/16k/16-bit (chuẩn tốt cho ASR)
    - Gom non-silent theo thời lượng mục tiêu suy ra từ giới hạn MB *trên PCM*
    - Chỉ export 1 lần/ chunk sang FLAC (lossless)
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(target_channels).set_frame_rate(target_sr).set_sample_width(target_bits // 8)

    non_silent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=audio.dBFS - 14,
        seek_step=1
    )
    if not non_silent_ranges:
        print("[INFO] No speech detected.")
        return []

    # Merge các khoảng gần nhau và thêm đệm
    merged = []
    cur_start, cur_end = non_silent_ranges[0]
    for s, e in non_silent_ranges[1:]:
        if s - cur_end <= join_gap_ms:
            cur_end = e
        else:
            merged.append([max(0, cur_start - keep_silence_ms), min(len(audio), cur_end + keep_silence_ms)])
            cur_start, cur_end = s, e
    merged.append([max(0, cur_start - keep_silence_ms), min(len(audio), cur_end + keep_silence_ms)])

    # Tính thời lượng mục tiêu dựa trên băng thông PCM chuẩn hóa
    bps = _wav_bytes_per_sec(sr=target_sr, ch=target_channels, bits=target_bits)
    target_sec = _sec_from_mb_pcm(max_chunk_mb, bps, safety=0.9)
    target_ms = int(target_sec * 1000)

    # Gom interval -> chunk theo target_ms
    chunks = []
    cur_chunk_start = None
    cur_chunk_end = None
    acc_ms = 0

    def flush_chunk(a_start, a_end):
        seg = audio[a_start:a_end]
        if len(seg) < int(min_chunk_sec * 1000):
            return None
        if seg.dBFS < min_loudness_dbfs:
            return None
        return seg

    for s, e in merged:
        seg_len = e - s
        if cur_chunk_start is None:
            cur_chunk_start, cur_chunk_end, acc_ms = s, e, seg_len
        else:
            if acc_ms + seg_len <= target_ms:
                cur_chunk_end = e
                acc_ms += seg_len
            else:
                seg = flush_chunk(cur_chunk_start, cur_chunk_end)
                if seg:
                    chunks.append(seg)
                cur_chunk_start, cur_chunk_end, acc_ms = s, e, seg_len

    seg = flush_chunk(cur_chunk_start, cur_chunk_end)
    if seg:
        chunks.append(seg)

    if not chunks:
        print("[INFO] No valid chunks after filtering.")
        return []

    # Export FLAC song song
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_exports) as ex:
        futs = [ex.submit(_export_flac_to_bytes, seg, f"chunk_{i}.flac") for i, seg in enumerate(chunks, 1)]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())

    results.sort(key=lambda x: int(x["name"].split("_")[1].split(".")[0]))
    print(f"[INFO] Final: {len(results)} safe FLAC chunks for Gemini.")
    return results

class Generator:
    """
    An optimized, universal client for the Unified AI Gateway server.
    Supports both text and multimodal (text and image) generation, intelligently
    leveraging the server's backend routing for Mistral and Gemini models.
    """
    def __init__(self,
                 base_url: str = "http://localhost:9501",
                 api_key: str = "dummy", # The gateway manages keys, so this can be a placeholder
                 model_name: str = "mistral-medium-latest",
                 temperature: float = 0.7,
                 max_new_tokens: int = 4096,
                 timeout: int = 900):
        """
        Initialize the Generator for the Unified AI Gateway.

        Args:
            base_url: URL of your AI Gateway server.
            api_key: API key (can be a dummy value for the gateway).
            model_name: Default model name to use for requests.
            temperature: Default temperature for generation.
            max_new_tokens: Default maximum number of new tokens to generate.
            timeout: Request timeout in seconds (increased for vision models).
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

        # Initialize OpenAI client pointed at your server's endpoint
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
            timeout=timeout
        )

        print(f"✅ Generator Initialized: Connected to Unified AI Gateway at {self.base_url}")
        self._test_connection()

    def _test_connection(self) -> None:
        """Tests the connection to the server and displays its capabilities."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"🏥 Server Health: {health_data.get('status', 'unknown').capitalize()}")

                if health_data.get('vision_support'):
                    print("📷 Vision Support: Enabled")
                    vision_models = health_data.get('supported_vision_models', [])
                    if vision_models:
                        print(f"🎯 Supported Vision Models: {', '.join(vision_models)}")
                else:
                    print("📷 Vision Support: Disabled")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not fetch server health info: {e}")
            print("🔄 Proceeding with basic configuration, but server may be offline.")

    def get_available_models(self) -> List[str]:
        """Gets the list of available models from the server."""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            print(f"⚠️ Could not fetch models from the gateway: {e}")
            return []

    def _encode_image_from_path(self, image_path: str) -> str:
        """Loads an image file, encodes it to Base64, and formats it as a data URI."""
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found at {image_path}")

            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith('image'):
                ext_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp'}
                mime_type = ext_map.get(path.suffix.lower())
                if not mime_type:
                    raise ValueError(f"Unsupported or unknown image format for file: {path.name}")

            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            raise IOError(f"Error processing image file {image_path}: {e}") from e

    def _encode_image_from_pil(self, image: Image.Image) -> str:
        """Encodes a PIL.Image object to a Base64 data URI."""
        try:
            buffered = BytesIO()
            # Save as PNG to preserve quality; it's a safe, widely supported format.
            image.save(buffered, format="PNG")
            base64_bytes = base64.b64encode(buffered.getvalue())
            return f"data:image/png;base64,{base64_bytes.decode('utf-8')}"
        except Exception as e:
            raise IOError(f"Error processing PIL image object: {e}") from e

    def _prepare_api_messages(self,
                            prompt: Optional[str],
                            messages: Optional[List[Message]],
                            images: Optional[List[Union[str, Image.Image]]]) -> Tuple[List[Message], List[Message], bool]:
        """Prepares messages for the API server, handling text and various image input types."""
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        full_history = list(messages) if messages else []

        if not full_history:
            full_history.append({"role": "user", "content": prompt or ""})

        if not images:
            return full_history, full_history, False

        # Images can only be attached to the last user message
        last_message = full_history[-1]
        if last_message["role"] != "user":
            raise ValueError("Images can only be added to the most recent 'user' message in the history.")

        # Ensure content is a string before modification
        text_content = last_message.get("content", "")
        if not isinstance(text_content, str):
            raise TypeError("The last message content must be a string when providing new images.")

        has_images = False
        final_user_content = [{"type": "text", "text": text_content}]

        for i, img_input in enumerate(images):
            try:
                if isinstance(img_input, str):
                    base64_image_uri = self._encode_image_from_path(img_input)
                elif isinstance(img_input, Image.Image):
                    base64_image_uri = self._encode_image_from_pil(img_input)
                else:
                    raise TypeError(f"Image input must be a file path (str) or a PIL.Image.Image object, but got {type(img_input)}")

                final_user_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_image_uri}
                })
                has_images = True
            except (IOError, FileNotFoundError, ValueError, TypeError) as e:
                error_msg = f"[Error processing image {i+1}: {e}]"
                # Prepend the error to the text content for the LLM to see
                final_user_content[0]["text"] = f"{error_msg}\n{final_user_content[0]['text']}"
                print(f"⚠️  {error_msg}")

        # Replace the last message with the new multimodal content
        api_messages = full_history[:-1]
        api_messages.append({"role": "user", "content": final_user_content})

        return api_messages, full_history, has_images

    def _handle_api_error(self, error: Exception) -> str:
        """Provides user-friendly interpretations of potential API errors."""
        error_str = str(error)
        if "Connection refused" in error_str:
            return f"Connection Failed. Is the AI Gateway server running at {self.base_url}?"
        if "All Gemini API keys have been exhausted" in error_str:
            return "Server reported that all Gemini keys are exhausted or have failed. Please check the server logs."
        if "Mistral request failed after" in error_str:
            return "Server reported that all retries for Mistral failed. The Mistral API might be down or all keys are invalid."
        return f"An unhandled error occurred: {error_str}"

    def generate(self,
                 prompt: Optional[str] = None,
                 messages: Optional[List[Message]] = None,
                 images: Optional[List[Union[str, Image.Image]]] = None,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_new_tokens: Optional[int] = None,
                 **kwargs) -> Tuple[str, List[Message]]:
        """
        Generates a response using the Unified AI Gateway.

        The gateway server automatically:
        - Routes requests to Mistral or Gemini based on the model name.
        - Manages API key rotation, rate limiting, and load balancing.
        - Provides real-time metrics and monitoring.

        Args:
            prompt: A single string prompt. Used if 'messages' is not provided.
            messages: A list of message dictionaries (OpenAI format).
            images: A list of image inputs. Items can be file paths (str) or PIL.Image.Image objects.
            model_name: The model to use (e.g., "gpt-4o", "mistral-medium-latest").
            temperature: The temperature for this request.
            max_new_tokens: The max tokens for this request.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A tuple containing:
            - The generated text response (str).
            - The complete conversation history including the new response (list).
        """
        try:
            api_messages, conversation_history, has_images = self._prepare_api_messages(
                prompt, messages, images
            )

            selected_model = model_name if model_name is not None else self.model_name

            if has_images:
                num_images = len([part for part in api_messages[-1]['content'] if part.get('type') == 'image_url'])
                print(f"📷 Sending multimodal request to model '{selected_model}' with {num_images} image(s)")

            api_params = {
                "model": selected_model,
                "messages": api_messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                **kwargs
            }

            response = self.client.chat.completions.create(**api_params)
            result = response.choices[0].message.content or ""

            if response.model and response.model != selected_model:
                print(f"🔄 Server auto-routed request to model: {response.model} (requested: {selected_model})")

        except Exception as e:
            error_message = self._handle_api_error(e)
            print(f"❌ Generation failed: {error_message}")
            raise RuntimeError(error_message) from e

        conversation_history.append({"role": "assistant", "content": result})
        return result, conversation_history
    
    def _handle_m3u8_playlist(self, file_path: str) -> bytes:
        """
        Parses and processes an M3U8 playlist, correctly handling both
        web URLs and local file paths for media segments.
        """
        # print(f"📄 Playlist file detected. Parsing and processing media from '{Path(file_path).name}'...")
        try:
            # The base path of the m3u8 file is used to resolve relative segment paths.
            base_path = Path(file_path).parent
            playlist = m3u8.load(file_path)
            
            if playlist.is_variant:
                # print("ℹ️ Master playlist found. Selecting first available stream.")
                # The URI in a master playlist can be another m3u8 file
                sub_playlist_uri = playlist.playlists[0].uri
                # Resolve its path relative to the master playlist's path
                playlist = m3u8.load(str(base_path / sub_playlist_uri))

            media_segments = []
            total_segments = len(playlist.segments)
            # print(f"⚙️ Processing {total_segments} media segments...")

            for i, segment in enumerate(playlist.segments):
                segment_uri = segment.uri
                
                # --- CORRECTED LOGIC: Differentiate between URL and local path ---
                if segment_uri.startswith(('http://', 'https://')):
                    # It's a full web URL, download it.
                    response = requests.get(segment_uri, timeout=30)
                    response.raise_for_status()
                    media_segments.append(response.content)
                else:
                    # It's a relative local file path. Construct the full path and read it.
                    local_segment_path = base_path / segment_uri
                    if not local_segment_path.exists():
                        raise FileNotFoundError(f"Media segment file not found: {local_segment_path}")
                    
                    with open(local_segment_path, 'rb') as f_segment:
                        media_segments.append(f_segment.read())
                # --- END OF CORRECTION ---

                print(f"   Processed segment {i+1}/{total_segments}", end='\r')

            # print("\n✅ Processing complete. Stitching segments together.")
            return b"".join(media_segments)

        except Exception as e:
            raise IOError(f"Failed to process M3U8 playlist '{file_path}': {e}") from e

    def _chunk_audio_by_time(self, audio: AudioSegment, chunk_duration_sec: int) -> List[Dict[str, Any]]:
        """
        MODIFIED: Slices an audio segment into fixed-duration chunks WITH an overlap.
        It stores both the actual audio segment and its logical, non-overlapping time range.
        """
        overlap_ms = int(5.0 * 1000)
        chunk_duration_ms = int(chunk_duration_sec * 1000)
        
        print(f"🔪 Slicing audio into {chunk_duration_sec}-second segments with a {5.0}-second overlap...")
        chunks = []
        duration_ms = len(audio)
        
        # The step is the chunk duration minus the overlap
        step = chunk_duration_ms - overlap_ms
        
        for i in range(0, duration_ms, step):
            # Define the logical, non-overlapping boundaries for this chunk
            logical_start_ms = i
            logical_end_ms = min(i + chunk_duration_ms, duration_ms)

            # Define the actual slice boundaries, adding the overlap to the end
            actual_end_ms = min(logical_end_ms + overlap_ms, duration_ms)
            
            # The start of the slice is just the logical start
            segment = audio[logical_start_ms:actual_end_ms]

            if len(segment) > 500: # Ignore tiny leftover segments
                chunks.append({
                    "segment": segment,
                    "start_sec": logical_start_ms / 1000.0,
                    "end_sec": logical_end_ms / 1000.0
                })
        print(f"✅ Created {len(chunks)} overlapping time-based chunks.")
        return chunks

    def _transcribe_chunk(self,
                          chunk_data: Dict[str, Any],
                          model: str,
                          prompt: Optional[str],
                          language: Optional[str]) -> Tuple[Dict[str, Any], str]:
        """
        MODIFIED: Now always returns the chunk_data and either the result or a formatted error string.
        It no longer raises an exception on its own.
        """
        transcription_url = f"{self.base_url}/v1/audio/transcriptions"
        
        data_payload = {'model': model}
        if prompt: data_payload['prompt'] = prompt
        if language: data_payload['language'] = language
        
        files_payload = {'file': (chunk_data["name"], chunk_data["bytes"])}
        
        try:
            response = requests.post(
                transcription_url, files=files_payload, data=data_payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            if "text" not in result:
                raise Exception(f"Server returned an unexpected JSON response: {result}")
            
            print(f"✅ Transcribed chunk '{chunk_data['name']}' ({_format_time(chunk_data['start_sec'])}) successfully.")
            return (chunk_data, result["text"])
        except Exception as e:
            error_message = f"Failed to transcribe chunk {chunk_data['name']} due to {type(e).__name__}"
            print(f"❌ {error_message}")
            # Return a formatted, recognizable error string instead of raising an exception
            return (chunk_data, f"[ERROR: {error_message}]")


    async def transcribe(self,
                         file: Union[str, bytes],
                         model: str = "gemini-1.5-pro-latest",
                         file_name: Optional[str] = "media.dat",
                         prompt: Optional[str] = None,
                         language: Optional[str] = None,
                         chunk_duration_minutes: float = 5.0,
                         max_concurrent_chunks: int = 10,
                         # --- NEW PARAMETER FOR TARGETED RETRIES ---
                         time_ranges_to_process: Optional[List[Tuple[float, float]]] = None
                        ) -> Dict[str, str]:
        """
        Transcribes an audio or video file into a timestamped JSON object.

        The file is split into fixed-duration chunks (e.g., 5 minutes), which are
        processed concurrently for high speed.

        Args:
            file: File path (str) or file content (bytes).
            model: The transcription model to use (e.g., "gemini-1.5-pro-latest").
            file_name: A name for the file if input is bytes.
            prompt: An optional prompt to guide the transcription model.
            language: The language of the audio (e.g., "Vietnamese").
            chunk_duration_minutes: The duration of each chunk in minutes.
            max_concurrent_chunks: The number of chunks to process in parallel.

        Returns:
            A dictionary where keys are time ranges ("HH:MM:SS-HH:MM:SS") and
            values are the transcribed text for that range.
        """
        print(f"🎤 Preparing concurrent transcription for model '{model}'...")
        temp_dir = tempfile.mkdtemp()
        
        try:
            input_path_str = None
            if isinstance(file, str):
                if not Path(file).exists(): raise FileNotFoundError(f"Input file not found: {file}")
                input_path_str = file
            elif isinstance(file, bytes):
                input_path = Path(temp_dir) / file_name
                with open(input_path, 'wb') as f: f.write(file)
                input_path_str = str(input_path)
            else:
                raise TypeError(f"File input must be a file path (str) or bytes, but got {type(file)}")

            # --- STAGE 1: Load and standardize audio ---
            print("🔊 Standardizing audio to 16kHz mono for optimal transcription...")
            audio = AudioSegment.from_file(input_path_str)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

            # --- STAGE 2: Chunk audio by time duration ---
            time_based_chunks = []
            if not time_ranges_to_process:
                # A) Full file processing
                print(f"🔪 Slicing entire audio into {chunk_duration_minutes}-minute segments...")
                chunk_duration_sec = int(chunk_duration_minutes * 60)
                time_based_chunks = self._chunk_audio_by_time(audio, chunk_duration_sec)
            else:
                # B) Targeted range processing for retries
                print(f"🎯 Processing {len(time_ranges_to_process)} specific time range(s)...")
                for start_sec, end_sec in time_ranges_to_process:
                    segment = audio[int(start_sec * 1000):int(end_sec * 1000)]
                    if len(segment) > 500: # Ignore empty segments
                        time_based_chunks.append({
                            "segment": segment,
                            "start_sec": start_sec,
                            "end_sec": end_sec
                        })

            if not time_based_chunks:
                print("[INFO] No audio chunks to process.")
                return {}

            # --- STAGE 3: Prepare chunks for concurrent upload (export to FLAC) ---
            chunks_to_process = []
            for i, chunk in enumerate(time_based_chunks, 1):
                flac_data = _export_flac_to_bytes(chunk["segment"], f"chunk_{i}.flac")
                chunks_to_process.append({
                    "bytes": flac_data["bytes"],
                    "name": flac_data["name"],
                    "start_sec": chunk["start_sec"],
                    "end_sec": chunk["end_sec"],
                    "prompt": prompt # Use the same prompt for all chunks
                })
            
            # --- STAGE 4: Transcribe all chunks concurrently ---
            start_time = time.time()
            final_transcript = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_chunks) as executor:
                loop = asyncio.get_running_loop()
                tasks = [
                    loop.run_in_executor(
                        executor, self._transcribe_chunk,
                        chunk_data, model, prompt, language
                    ) for chunk_data in chunks_to_process
                ]
                
                # `results` will be a list of tuples: [(chunk_data, text), ...]
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                print(f"✅ All {len(results)} parts transcribed in {end_time - start_time:.2f} seconds.")

            # --- STAGE 5: Assemble final timestamped JSON object ---
            print("📝 Assembling final transcript with timestamps and any errors...")
            final_transcript = {}
            for chunk_data, text_or_error in sorted(results, key=lambda item: item[0]['start_sec']):
                start_str = chunk_data['start_sec']
                end_str = chunk_data['end_sec']
                time_key = f"{start_str}-{end_str}"
                
                # Check if the result for this chunk is an error string
                if isinstance(text_or_error, str) and text_or_error.strip().startswith("[ERROR:"):
                    final_transcript[time_key] = text_or_error
                else:
                    # It's a successful transcription, process it
                    processed_text = text_or_error.strip()
                    final_transcript[time_key] = processed_text

            return final_transcript

        except Exception as e:
            # This will only catch critical errors like file not found, not individual chunk errors
            print(f"❌ A critical error occurred during the transcription pre-processing: {e}")
            raise
        finally:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temporary files.")
        #OCR mistral
    def ocr(self,
            image: Union[str, Image.Image],
            model_name: str = "mistral-ocr-latest",
            include_image_base64: bool = False,
            **kwargs) -> Dict[str, Any]:
        """
        Performs Optical Character Recognition (OCR) on an image using Mistral's OCR models.

        This method sends a request to the dedicated `/v1/ocr` endpoint on the Unified AI Gateway.

        Args:
            image: The image to process. Can be a file path (str) or a PIL.Image.Image object.
            model_name: The OCR model to use (e.g., "mistral-ocr-latest").
            include_image_base64: Whether to include the base64-encoded image in the response.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A dictionary containing the structured OCR response from the server.

        Raises:
            RuntimeError: If the OCR request fails after all retries on the server side
                          or if there is a connection issue.
        """
        print(f"📄 Sending OCR request for model '{model_name}'...")
        try:
            if isinstance(image, str):
                base64_image_uri = self._encode_image_from_path(image)
            elif isinstance(image, Image.Image):
                base64_image_uri = self._encode_image_from_pil(image)
            else:
                raise TypeError(f"Image input must be a file path (str) or a PIL.Image.Image object, but got {type(image)}")

            ocr_url = f"{self.base_url}/v1/ocr"
            
            payload = {
                "model": model_name,
                "image": base64_image_uri,
                "include_image_base64": include_image_base64,
                **kwargs
            }

            response = requests.post(ocr_url, json=payload, timeout=self.timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            # The server might return 200 OK but with an error message in the JSON
            if "error" in result and result.get("status") == "error":
                 error_message = result.get("error_message", "Unknown OCR error from server")
                 raise Exception(error_message)

            return result

        except requests.exceptions.HTTPError as http_err:
            try:
                # Try to parse the JSON error from the response body
                error_details = http_err.response.json()
                error_message = error_details.get("error", {}).get("message", http_err.response.text)
            except json.JSONDecodeError:
                error_message = str(http_err)
            
            print(f"❌ OCR failed: {error_message}")
            raise RuntimeError(f"OCR request failed: {error_message}") from http_err
        
        except Exception as e:
            error_message = self._handle_api_error(e)
            print(f"❌ OCR failed: {error_message}")
            raise RuntimeError(error_message) from e
        
    def process_video(self,
                      video: Union[str, bytes],
                      prompt: str,
                      model_name: str = "gemini-1.5-pro-latest",
                      file_name: str = "video.mp4",
                      **kwargs) -> Tuple[str, List[Message]]:
        """
        Processes a video with a given prompt using the AI Gateway's dedicated endpoint.

        This method is ideal for analyzing video content, describing scenes, or answering
        questions about a video.

        Args:
            video: The video source. Can be a local file path (str), a web URL (str),
                   or raw video data (bytes).
            prompt: The text prompt to guide the analysis of the video.
            model_name: The model to use for processing (e.g., "gemini-1.5-pro-latest").
            file_name: A name for the file, required if the input is bytes.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A tuple containing the generated text response and a simplified conversation history.
        """
        print(f"📹 Preparing to process video for model '{model_name}'...")
        try:
            video_bytes: Optional[bytes] = None
            final_file_name = file_name

            # --- Step 1: Handle different video input types ---
            if isinstance(video, bytes):
                video_bytes = video
                print(f"   -> Processing video from in-memory bytes ({len(video_bytes)/1024**2:.2f} MB).")
            elif isinstance(video, str):
                if video.startswith(('http://', 'https://')):
                    print(f"   -> Downloading video from URL: {video}")
                    try:
                        response = requests.get(video, stream=True, timeout=self.timeout)
                        response.raise_for_status()
                        video_bytes = response.content
                        final_file_name = Path(video).name or file_name
                    except requests.RequestException as e:
                        raise IOError(f"Failed to download video from URL: {e}") from e
                else:
                    video_path = Path(video)
                    if not video_path.exists():
                        raise FileNotFoundError(f"Video file not found at: {video_path}")
                    print(f"   -> Reading video from local path: {video_path}")
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    final_file_name = video_path.name
            else:
                raise TypeError(f"Unsupported video input type: {type(video)}. Must be str (path/URL) or bytes.")

            if not video_bytes:
                raise ValueError("Video content could not be loaded or is empty.")

            # --- Step 2: Make the API request to the video processing endpoint ---
            video_processing_url = f"{self.base_url}/v1/video/process"
            
            data_payload = {'model': model_name, 'prompt': prompt}
            data_payload.update(kwargs) # Pass any extra parameters

            files_payload = {'file': (final_file_name, video_bytes, mimetypes.guess_type(final_file_name)[0] or 'video/mp4')}

            print(f"   -> Sending '{final_file_name}' to the gateway. This may take a while...")
            # Use a significantly longer timeout for video processing
            response = requests.post(
                video_processing_url,
                files=files_payload,
                data=data_payload,
                timeout=self.timeout * 5 # 5x default timeout for video
            )
            response.raise_for_status()

            result_json = response.json()
            if "error" in result_json: # Handle server-side errors returned in a 200 OK
                raise Exception(result_json.get("error", {}).get("message", "Unknown server error"))

            # --- Step 3: Format and return the response ---
            response_text = result_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            conversation_history = [
                {"role": "user", "content": f"[Video processed: {final_file_name}]\n\n{prompt}"},
                {"role": "assistant", "content": response_text}
            ]

            return response_text, conversation_history

        except Exception as e:
            error_message = self._handle_api_error(e)
            print(f"❌ Video processing failed: {error_message}")
            raise RuntimeError(error_message) from e
        
    def get_server_metrics(self) -> Optional[Dict]:
        """Gets real-time metrics from the gateway's dashboard."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json().get('metrics', {})
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not fetch server metrics: {e}")
        return None

    def print_server_status(self) -> None:
        """Prints a comprehensive status report from the AI Gateway."""
        metrics = self.get_server_metrics()
        if not metrics:
            print("\n❌ Could not retrieve server status. Is the server online?")
            return

        print("\n" + "="*50)
        print("🚀 UNIFIED AI GATEWAY STATUS")
        print("="*50)
        print(f"📊 Total Requests: {metrics.get('total_requests', 0):,}")
        print(f"✅ Success Rate:   {metrics.get('success_rate', 0):.1f}%")
        print(f"⚡ Active Requests:  {metrics.get('active_count', 0)}")
        print(f"🔑 Active Keys - Mistral: {metrics.get('mistral_keys_available', 'N/A')} | Gemini: {metrics.get('gemini_keys_available', 'N/A')}")
        print(f"⏱️ Avg Response:   {metrics.get('avg_response_time', 0)*1000:.0f} ms")
        print(f"📈 Throughput:     {metrics.get('requests_per_minute', 0):.1f} req/min")
        print(f"⏰ Uptime:         {metrics.get('uptime_formatted', 'Unknown')}")
        print("="*50)

# Message = Dict[str, Union[str, List[Dict[str, str]]]]
