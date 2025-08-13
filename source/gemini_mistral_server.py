import asyncio
import os
import json
import time
import base64
import re
import io
import threading
import queue
import contextlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
import mimetypes
# --- Third-party libraries ---
import pandas as pd
import requests
from aiohttp import web, WSMsgType
from mistralai import Mistral
from mistralai import File  # Corrected import for batch operations
from PIL import Image
from rich.console import Console
import tempfile
# --- Gemini-Specific Imports (requires 'pip install google-generativeai') ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Rich Console for Better Logging ---
console = Console()


# ==============================================================================
# 1. Custom Exceptions
# ==============================================================================
class NoAvailableKeysError(Exception):
    """Raised when no API keys are available and none are expected to become available."""
    pass

class ApiKeyQuotaExceededError(Exception):
    """Custom exception raised when an API key hits its rate limit or quota."""
    pass

MISTRAL_RPM_RATE = int(os.getenv("MISTRAL_RPM_RATE", 3000))      # e.g., 50 RPS * 60
MISTRAL_RPM_CAPACITY = int(os.getenv("MISTRAL_RPM_CAPACITY", 100)) # Burst capacity
GEMINI_RPM_RATE = int(os.getenv("GEMINI_RPM_RATE", 3000))        # e.g., 50 RPS * 60
GEMINI_RPM_CAPACITY = int(os.getenv("GEMINI_RPM_CAPACITY", 100))  # Burst capacity
permanent_failure_keywords = ["api key not valid", "INVALID_ARGUMENT","suspended", "api key expired", "permission_denied","CONSUMER_SUSPENDED", "Permission denied", 'API_KEY_INVALID', "expired", 'UNAVAILABLE', "	PERMISSION_DENIED"]
# ==============================================================================
# 2. Reusable, Thread-Safe Key Manager Base (for Gemini and Mistral)
# ==============================================================================
class BaseKeyManager:
    """A thread-safe, resilient API key manager with individual key rate limiting."""
    def __init__(
        self,
        provider_name: str,
        sheet_url: str,
        revival_delay_seconds: int = 90,
        reload_interval: int = 1200,
        key_rpm: int = 10, # NEW: Requests per minute allowed for each individual key
        full_reset_interval: int = 1800, # NEW: Full reset interval in seconds
        penalty_box_threshold: int = 5,     # NEW: Failures before entering penalty box
        penalty_box_duration: int = 600,    # NEW: 10-minute (600s) penalty box duration
    ):
        self.provider_name = provider_name
        self.sheet_url = sheet_url
        self.revival_delay_seconds = revival_delay_seconds
        self.penalty_box_threshold = penalty_box_threshold
        self.penalty_box_duration = penalty_box_duration
        self.reload_interval = reload_interval
        # NEW: Calculate the required cooldown period from the RPM value.
        # e.g., 30 RPM -> 2s cooldown. 60 RPM -> 1s cooldown.
        self.key_cooldown_seconds = 60.0 / key_rpm if key_rpm > 0 else 0
        self.full_reset_interval = full_reset_interval # Store new interval
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._active_keys: List[Dict] = []
        self._retired_keys: List[tuple[float, Dict]] = []
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._current_key_index = 0
        self._all_known_tokens = set()
        self._in_use_tokens: set[str] = set()
        self._all_known_keys: Dict[str, Dict] = {} 
        # NEW: Dictionary to track the last time a key was successfully released.
        self._key_last_used_timestamps: Dict[str, float] = {}

        self._stop_event = threading.Event()
        self._load_and_initialize_keys()
        self._revival_thread = threading.Thread(target=self._revive_keys_periodically, daemon=True)
        self._revival_thread.start()
        self._reload_thread = threading.Thread(target=self._reload_keys_periodically, daemon=True)
        self._reload_thread.start()
        if self.full_reset_interval > 0:
            self._full_reset_thread = threading.Thread(target=self._full_state_reset_periodically, daemon=True)
            self._full_reset_thread.start()
            console.print(f"[{self.provider_name} KeyManager] Full state reset configured every [bold green]{self.full_reset_interval / 60:.0f} minutes[/bold green].")
        
        console.print(f"[{self.provider_name} KeyManager] Initialized with [bold green]{len(self._active_keys)}[/bold green] keys. Cooldown: [cyan]{self.key_cooldown_seconds:.2f}s[/cyan] ({key_rpm} RPM/key).")

    def _load_api_keys_from_sheet(self) -> List[Dict]: raise NotImplementedError
    def _load_and_initialize_keys(self):
        try:
            new_keys_info = self._load_api_keys_from_sheet()
            with self._lock:
                newly_added = [info for info in new_keys_info if info['token'] not in self._all_known_tokens]
                for info in newly_added:
                    self._active_keys.append(info)
                    self._all_known_tokens.add(info['token'])
                    self._all_known_keys[info['token']] = info
                if newly_added:
                    console.print(f"[{self.provider_name} KeyManager] Loaded [green]{len(newly_added)}[/green] new keys.")
                    self._condition.notify_all()
                    
            if not self._all_known_tokens:
                raise ValueError(f"No {self.provider_name} API keys loaded.")
        except Exception as e:
            console.print(f"[{self.provider_name} KeyManager] [red]Key loading failed:[/red] {e}")

    def _revive_keys_periodically(self):
        while not self._stop_event.wait(5):
            with self._lock:
                now = time.time()
                keys_to_revive, remaining_retired = [], []
                # MODIFIED: Logic is simpler now. It just checks if the revival time has been reached.
                for revival_time, key_info in self._retired_keys:
                    if now >= revival_time:
                        keys_to_revive.append(key_info)
                        self._key_last_used_timestamps.pop(key_info['token'], None)
                    else:
                        remaining_retired.append((revival_time, key_info))
                if keys_to_revive:
                    self._retired_keys = remaining_retired
                    for key_info in keys_to_revive:
                        log_key_display = f"{key_info['name']}-{key_info['token'][:5]}...{key_info['token'][-7:]}" if self.provider_name == "Gemini" else key_info['token']
                        self._active_keys.append(key_info)
                        failure_count = self._failure_counts.pop(key_info['token'], 0)
                        console.print(f"[{self.provider_name} KeyManager] ✅ Revived key [cyan]{log_key_display}[/cyan]. Failure:{failure_count}. Active keys: [green]{len(self._active_keys)}[/green].")
                    self._condition.notify_all()
    def _reload_keys_periodically(self):
        while not self._stop_event.wait(self.reload_interval):
            self._load_and_initialize_keys()
    def _perform_full_reset(self):
        """Helper function containing the logic for a full state reset."""
        console.print(f"[{self.provider_name} KeyManager] 🔄 Performing full key state reset...", style="bold magenta")
        
        for _, key_info in self._retired_keys:
            if key_info['token'] in self._all_known_tokens:
                self._active_keys.append(key_info)
        self._retired_keys.clear()
        
        self._failure_counts.clear()
        self._in_use_tokens.clear()
        self._key_last_used_timestamps.clear()
        
        console.print(f"[{self.provider_name} KeyManager] ✅ Full state reset complete. All keys marked active/available.", style="bold magenta")
        console.print(f"[{self.provider_name} KeyManager] Active keys: [magenta]{len(self._active_keys)}[/magenta].", style="bold green")
        self._condition.notify_all()
    def remove_key_permanently(self, key_info: Dict[str, Any]):
        """
        Permanently removes a key from all tracking and state management.
        This is used for suspended or invalid keys.
        """
        with self._lock:
            key_token = key_info['token']
            key_name = key_info.get('name', 'N/A')
            
            # Check if the key is known before proceeding
            if key_token not in self._all_known_tokens:
                return

            console.print(f"[{self.provider_name} KeyManager] 🔥 Permanently removing suspended/invalid key: [red]{key_name}[/red] ({key_token[:7]}...)", style="bold red")

            # Remove from all possible states
            self._active_keys = [k for k in self._active_keys if k['token'] != key_token]
            self._retired_keys = [(rt, k) for rt, k in self._retired_keys if k['token'] != key_token]
            self._in_use_tokens.discard(key_token)
            
            # Remove from all state-tracking dictionaries
            self._key_last_used_timestamps.pop(key_token, None)
            self._failure_counts.pop(key_token, None)
            self._all_known_tokens.discard(key_token)
            # self._all_known_keys.pop
            
            self._condition.notify_all()
    def _full_state_reset_periodically(self):
        """Periodically calls the full state reset logic."""
        while not self._stop_event.wait(self.full_reset_interval):
            with self._lock:
                self._perform_full_reset()

    def _emergency_revive_all_keys(self):
        """
        A less aggressive reset that only revives retired keys without
        affecting active keys' cooldowns or failure counts.
        """
        console.print(f"[{self.provider_name} KeyManager] 🚨 Performing emergency revival of all retired keys...", style="bold yellow")
        revived_count = 0
        for _, key_info in self._retired_keys:
            if key_info['token'] in self._all_known_tokens:
                self._active_keys.append(key_info)
                self._key_last_used_timestamps.pop(key_info['token'], None)
                self._failure_counts.pop(key_info['token'], 0)
                revived_count += 1
        self._retired_keys.clear()
        
        if revived_count > 0:
            console.print(f"[{self.provider_name} KeyManager] ✅ Emergency revival complete. Moved [green]{revived_count}[/green] keys to active pool.", style="bold yellow")
            console.print(f"[{self.provider_name} KeyManager] Active keys: [magenta]{len(self._active_keys)}[/magenta].", style="bold green")
            self._condition.notify_all()
                
    def select_key(self) -> Dict[str, Any]:
        with self._lock:
            start_wait_time = time.monotonic()
            while True:
                now = time.time()
                # MODIFIED: A key is free if it's not locked AND not on cooldown.
                free_keys = [
                    k for k in self._active_keys
                    if k['token'] not in self._in_use_tokens
                    and now - self._key_last_used_timestamps.get(k['token'], 0) > self.key_cooldown_seconds
                ]

                if free_keys:
                    self._current_key_index = (self._current_key_index + 1) % len(free_keys)
                    key_info = free_keys[self._current_key_index]
                    self._in_use_tokens.add(key_info['token']) # LOCK
                    return key_info
                if not self._active_keys and not self._retired_keys:
                    self._emergency_revive_all_keys() # The `select_key` method calls the emergency function.
                    # raise NoAvailableKeysError(f"All {self.provider_name} keys removed.")
                    continue
                if time.monotonic() - start_wait_time > 2:
                    # self._emergency_revive_all_keys()
                    console.print(f"[{self.provider_name} KeyManager] All active keys are busy or on cooldown. Waiting...", style="yellow")
                    start_wait_time = time.monotonic()

                # Wait for a key to be released or for a cooldown to expire.
                # The timeout allows it to re-check the timestamps periodically.
                self._condition.wait(timeout=self.key_cooldown_seconds or 1.0)

    def release_key(self, key_info: Dict[str, Any]):
        with self._lock:
            token = key_info['token']
            if token in self._in_use_tokens:
                self._in_use_tokens.remove(token) # UNLOCK
                # NEW: Update the timestamp to start the cooldown.
                self._key_last_used_timestamps[token] = time.time()
                self._condition.notify() # Wake up one waiting thread.
                # console.print(f"[{self.provider_name} KeyManager] Released key [cyan]{key_info['name']}-{token[:5]}...{token[-7:]}[/cyan]. Cooldown started.", style="green")

    def retire_key(self, key_info: Dict[str, Any]):
        with self._lock:
            key_token, key_name = key_info['token'], key_info['name']
            log_key_display = f"{key_info['name']}-{key_token[:5]}...{key_token[-7:]}" if self.provider_name == "Gemini" else key_name
            original_len = len(self._active_keys)
            self._active_keys = [k for k in self._active_keys if k['token'] != key_token]
            if len(self._active_keys) < original_len:
                self._in_use_tokens.discard(key_token) # Ensure it's unlocked
                # NEW: Clear the timestamp for the retired key.
                self._key_last_used_timestamps.pop(key_token, None)
                self._failure_counts[key_token] += 1
                fail_count = self._failure_counts[key_token]
                retirement_duration = self.revival_delay_seconds
                log_message_suffix = ""
                
                if fail_count >= self.penalty_box_threshold:
                    retirement_duration = self.penalty_box_duration
                    log_message_suffix = f" [bold yellow](Penalty Box)[/bold yellow]"
                
                # Calculate the absolute time when the key should be revived.
                revival_time = time.time() + retirement_duration
                
                self._retired_keys.append((revival_time, key_info))
                console.print(f"[{self.provider_name} KeyManager]  Retiring key [red]{log_key_display}[/red] for {retirement_duration}s. Total failures: {fail_count}.{log_message_suffix}")

    def get_active_key_count(self) -> int:
        with self._lock:
            return len(self._active_keys)

    def shutdown(self):
        console.print(f"[{self.provider_name} KeyManager] Shutting down...")
        self._stop_event.set()
        with self._lock: self._condition.notify_all()
        if self._revival_thread.is_alive(): self._revival_thread.join(timeout=1)
        if self._reload_thread.is_alive(): self._reload_thread.join(timeout=1)

class GeminiApiKeyManager(BaseKeyManager):
    def __init__(self, **kwargs):
        super().__init__(provider_name="Gemini", **kwargs)

    def _load_api_keys_from_sheet(self) -> List[Dict]:
        console.print(f"[{self.provider_name} KeyManager] Fetching API keys from Google Sheets...")
        try:
            response = requests.get(self.sheet_url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = df.columns.str.strip()
            if 'Token' not in df.columns or 'Name' not in df.columns:
                raise ValueError("Gemini CSV must contain 'Token' and 'Name' columns.")
            df = df.dropna(subset=['Token', 'Name']).drop_duplicates(subset=['Token'])
            console.print(f"[{self.provider_name} KeyManager] Loaded {len(df)} keys from Google Sheets.")
            return df[['Token', 'Name']].rename(columns={'Token': 'token', 'Name': 'name'}).to_dict('records')
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.provider_name} API keys: {e}")

class MistralApiKeyManager(BaseKeyManager):
    def __init__(self, **kwargs):
        super().__init__(provider_name="Mistral", **kwargs)

    def _load_api_keys_from_sheet(self) -> List[Dict]:
        console.print(f"[{self.provider_name} KeyManager] Fetching API keys from Google Sheets...")
        try:
            response = requests.get(self.sheet_url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), header=None, dtype=str)
            keys = df[0].dropna().str.strip().unique()
            keys_info = []
            for i, token in enumerate(keys):
                if token:
                    name = f"mistral-{token[:4]}...{token[-4:]}-{i}"
                    keys_info.append({'token': token, 'name': name})
            return keys_info
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.provider_name} API keys: {e}")

class AsyncTokenBucket:
    """
    An asynchronous token bucket rate limiter that works on a
    Requests Per Minute (RPM) basis.
    """
    def __init__(self, rate: int, capacity: int, loop: asyncio.AbstractEventLoop):
        """
        Initializes the token bucket for a given RPM.
        Args:
            rpm (int): The number of requests allowed per minute.
            capacity (int): The maximum number of tokens the bucket can hold (burst capacity).
            loop (asyncio.AbstractEventLoop): The asyncio event loop.
        """
        self.rpm = rate
        self.capacity = capacity
        self._loop = loop
        self._tokens = self.capacity
        self._lock = asyncio.Lock()
        self._waiters = deque()
        
        # Calculate the delay between refilling single tokens to match the RPM.
        # e.g., 120 RPM means a new token should be available every 60/120 = 0.5 seconds.
        self._refill_interval = 60.0 / self.rpm if self.rpm > 0 else float('inf')
        
        self._refill_task = self._loop.create_task(self._refill_periodically())

    async def acquire(self):
        """Acquires a token. If none are available, waits until one is."""
        async with self._lock:
            if self._tokens > 0:
                self._tokens -= 1
                return
            # No tokens, need to wait
            future = self._loop.create_future()
            self._waiters.append(future)
        # Wait outside the lock
        await future

    async def _refill_periodically(self):
        """Background task to refill tokens periodically to match the target RPM."""
        while True:
            await asyncio.sleep(self._refill_interval)
            async with self._lock:
                # Add one token, up to the burst capacity.
                if self._tokens < self.capacity:
                    self._tokens += 1
                
                # Wake up one waiter if any are waiting and we have a token.
                if self._tokens > 0 and self._waiters:
                    self._tokens -= 1
                    waiter = self._waiters.popleft()
                    waiter.set_result(True)

    async def stop(self):
        self._refill_task.cancel()

# ==============================================================================
# 3. Stateless Generator Clients
# ==============================================================================
class GeneratorGemini:
    def __init__(self, api_key_manager: 'GeminiApiKeyManager', model_name="gemini-1.5-pro-latest", temperature=0.0, max_new_tokens=8192):
        self.api_key_manager = api_key_manager
        self.model_name, self.temperature, self.max_new_tokens = model_name, temperature, max_new_tokens
        if not genai: raise ImportError("'google-generativeai' not installed.")
        self.genai = genai

    def _prepare_api_request(self, messages, images):
        history, last_message = messages[:-1], messages[-1]
        formatted_history = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [m.get("content", "")]} for m in history if isinstance(m.get("content", ""), str)]
        prompt_content = []
        if isinstance(last_message.get('content', ''), str): prompt_content.append(last_message['content'])
        if images: prompt_content.extend(images)
        return prompt_content, formatted_history

    async def generate(self, messages: list,
                       images: Optional[List[Image.Image]] = None,
                       model_name: str = None,
                       max_new_tokens: int = None, temperature: float = None) -> str:
        """
        Generates a response from Gemini. It first acquires a token from the global
        rate limiter, then locks a key for the duration of the API call.
        """
        # NEW: The generate method is now async because it awaits the rate limiter.
        
        # Acquire a global request token first. This will block if the overall
        # server request rate to Gemini is too high.
        await gemini_rate_limiter.acquire()
        
        # The rest of the logic can be run in a thread because the API calls are synchronous.
        def sync_logic():
            while True:
                key_info = None
                try:
                    key_info = self.api_key_manager.select_key()
                    key_token = key_info['token']
                    log_key_display = f"{key_info['name']}-{key_token[:5]}...{key_token[-7:]}"
                    console.print(f"[GeneratorGemini] Attempting with locked key '[cyan]{log_key_display}[/cyan]'.", style="yellow")
                    
                    self.genai.configure(api_key=key_token)
                    generation_config = {"max_output_tokens": max_new_tokens or self.max_new_tokens, 
                                         "temperature": temperature or self.temperature, 
                                        #  "seed": 42, 
                                         "top_p":0.9}
                    model = self.genai.GenerativeModel(model_name or self.model_name, generation_config=generation_config)
                    prompt_content, api_history = self._prepare_api_request(messages, images)
                    if not prompt_content: raise ValueError("Cannot send a message with no content.")
                    response = model.start_chat(history=api_history).send_message(prompt_content, stream=False)
                    
                    console.print(f"[GeneratorGemini] Request successful with key '[cyan]{log_key_display}[/cyan]'.", style="green")
                    # print(response)
                    return response.text
                except requests.HTTPError as http_err:
                    status_code = http_err.response.status_code 
                    if status_code in ["504", "503", "500", "429", "502"]:
                        if key_info: self.api_key_manager.retire_key(key_info)
                        continue
                    elif status_code in ["403", "400"]:
                        console.print(f"[GeneratorGemini]💀 Key '[cyan]{log_key_display}[/cyan]' is suspended or invalid. Removing permanently.", style="bold red")
                        self.api_key_manager.remove_key_permanently(key_info)
                        continue
                        # console.log(f"Unexpected error with key {log_key_display}. Error {e}")
                    else:
                        console.print(f"[GeminiWorker] Unexpected error with key '[cyan]{log_key_display}[/cyan]'. Retrying with another key.", style="bold red")
                        if key_info: self.api_key_manager.retire_key(key_info) # Return empty string for this chunk, don't crash the whole job.
                        raise f"Unexpected error with key {log_key_display}. Error {e}"
                except Exception as e:
                    error_str = str(e).lower()
                    if any(k in error_str for k in ["quota", "rate limit", "exceeded", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "invalid argument", "internal error", "504", "503", "500", "429"]):
                        if key_info: self.api_key_manager.retire_key(key_info)
                        continue
                    elif any(k in error_str for k in permanent_failure_keywords):
                        if key_info:
                            console.print(f"[GeneratorGemini]💀 Key '[cyan]{log_key_display}[/cyan]' is suspended or invalid. Removing permanently.", style="bold red")
                            self.api_key_manager.remove_key_permanently(key_info)
                        console.log(f"Unexpected error with key {log_key_display}. Error {e}")
                        continue
                        # raise ValueError(f"Key '{log_key_display}' is suspended or invalid. Please check your API keys.")
                    else:
                        console.print(f"[GeminiWorker] Unexpected error with key '[cyan]{log_key_display}[/cyan]'. Failing", style="bold red")
                        console.log(f"Unexpected error with key {log_key_display}. Error {e}")
                        if key_info: self.api_key_manager.retire_key(key_info)
                        continue
                finally:
                    if key_info: self.api_key_manager.release_key(key_info)

        # Execute the synchronous key selection and API call logic in a separate thread.
        return await asyncio.to_thread(sync_logic)
    

    async def transcribe_media(self,
                            # NO session_key parameter
                            file_content: bytes,
                            mime_type: str,
                            file_name: str,
                            model: str,
                            prompt: Optional[str] = None,
                            language: Optional[str] = None) -> str:
        """
        Transcribes a single file chunk in a completely stateless manner,
        mirroring the behavior of the `generate` method.
        """
        # Each chunk request must wait for a global rate limit token.
        await gemini_rate_limiter.acquire()

        def sync_logic():
            # The main retry loop for this ONE chunk.
            while True:
                key_info = None
                uploaded_file = None
                
                # The 'with' statement for the temp file goes here now.
                with tempfile.NamedTemporaryFile(delete=True, suffix=f"_{file_name}") as temp_f:
                    temp_f.write(file_content)
                    temp_file_path = temp_f.name
                    
                    try:
                        # Select a key just for this chunk's attempt.
                        key_info = self.api_key_manager.select_key()
                        key_token = key_info['token']
                        log_key_display = f"{key_info['name']}-{key_token[:7]}...{key_token[-4:]}"
                        console.print(f"[GeminiWorker] Transcribing '{file_name}' with key '[cyan]{log_key_display}[/cyan]'.", style="dim")
                        
                        self.genai.configure(api_key=key_token)

                        # The entire operation for this chunk uses this one key.
                        uploaded_file = self.genai.upload_file(path=temp_file_path, display_name=file_name)
                        language_prompt = language if language is not None else "Vietnamese"
                        model_prompt = (
                            f"You are an assistant specialized in transcribing speech. Please generate an accurate transcript of the {language_prompt} audio about TV new or gameshow provided below."
                            "Do not add any commentary, analysis, or introductory phrases."
                            "Return only the transcription, no other text."
                        )
                        
                        if prompt: model_prompt += f"\n\nAdditional instructions or context: {prompt}"
                        self.genai.configure(api_key=key_token)
                        generation_config = {"max_output_tokens": 8192, 
                                             "temperature": 0.0, 
                                            #  "seed": 42, 
                                             "top_p": 0.9}
                        genai_model = self.genai.GenerativeModel(model_name=model, generation_config=generation_config)
                        response = genai_model.generate_content([model_prompt, uploaded_file])
                        console.print(f"[GeminiWorker] Transcription successful for '{file_name}' with key '[cyan]{log_key_display}[/cyan]'.", style="green")
                        try:
                            return response.text
                        except ValueError:

                            # The response.candidates list holds the raw output from the API.
                            finish_reason_int = 0 # Default to UNSPECIFIED
                            if response.candidates:
                                finish_reason_int = response.candidates[0].finish_reason.value

                            # According to Gemini docs, FinishReason.STOP has an integer value of 1.
                            # This integer comparison will work on all SDK versions.
                            if finish_reason_int == 1: # STOP
                                console.print(f"[GeminiWorker] Chunk '{file_name}' contained no discernible speech. Returning empty string.", style="dim")
                                return "" # This is a valid, successful outcome.
                            else:
                                # For any other reason (SAFETY=3, MAX_TOKENS=2, etc.), it's a real issue.
                                raise ValueError(f"Model returned no content for chunk '{file_name}' due to finish reason code: {finish_reason_int}")
                    except requests.HTTPError as http_err:
                        status_code = http_err.response.status_code 
                        if status_code in ["504", "503", "500", "429"]:
                            if key_info: self.api_key_manager.retire_key(key_info)
                            continue
                        elif status_code in ["403", "400"]:
                            console.print(f"[GeneratorGemini]💀 Key '[cyan]{log_key_display}[/cyan]' is suspended or invalid. Removing permanently.", style="bold red")
                            self.api_key_manager.remove_key_permanently(key_info)
                            continue
                            # console.log(f"Unexpected error with key {log_key_display}. Error {e}")
                        else:
                            console.print(f"[GeminiWorker] Unexpected error with key '[cyan]{log_key_display}[/cyan]'. Retrying with another key.", style="bold red")
                            if key_info: self.api_key_manager.retire_key(key_info)
                            console.log(f"Unexpected error with key {log_key_display}. Error {e}") # Return empty string for this chunk, don't crash the whole job.
                            continue
                    except Exception as e:
                        error_str = str(e).lower()
                        if any(k in error_str for k in ["quota", "rate limit", "exceeded", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "invalid argument","504", "503", "500", "429", "internal error"]):
                            if key_info: self.api_key_manager.retire_key(key_info)
                            console.log(f"Error with key {log_key_display}. Error {e}")
                            continue
                        elif any(k in error_str for k in permanent_failure_keywords):
                            if key_info:
                                console.print(f"[GeneratorGemini]💀 Key '[cyan]{log_key_display}[/cyan]' is suspended or invalid. Removing permanently.", style="bold red")
                                console.log(f"Unexpected error with key {log_key_display}. Error {e}")
                                self.api_key_manager.remove_key_permanently(key_info)
                            continue
                            # raise ValueError(f"Key '{log_key_display}' is suspended or invalid. Please check your API keys.")
                        else:
                            console.print(f"[GeminiWorker] Unexpected error with key '[cyan]{log_key_display}[/cyan]'. Retrying with another key.", style="bold red")
                            if key_info: self.api_key_manager.retire_key(key_info)
                            console.log(f"Unexpected error with key {log_key_display}. Error {e}") # Return empty string for this chunk, don't crash the whole job.
                            continue
                    finally:
                        # Always release the key and the uploaded Google file for this attempt.
                        if key_info:
                            self.api_key_manager.release_key(key_info)
                        if uploaded_file:
                            try:
                                self.genai.delete_file(uploaded_file.name)
                            except Exception:
                                pass

        return await asyncio.to_thread(sync_logic)
# 4. Server Configuration and Global State
# ==============================================================================

# --- Configuration ---
MISTRAL_SHEET_URL = os.getenv("MISTRAL_SHEET_URL", "https://docs.google.com/spreadsheets/d/1NAlj7OiD9apH3U47RLJK0en1wLSW78X5zqmf6NmVUA4/export?format=csv&gid=0")
GEMINI_SHEET_URL = os.getenv("GEMINI_SHEET_URL", "https://docs.google.com/spreadsheets/d/1gqlLToS3OXPA-CvfgXRnZ1A6n32eXMTkXz4ghqZxe2I/export?format=csv&gid=0")
# GEMINI_SHEET_URL = os.getenv("GEMINI_SHEET_URL", "https://docs.google.com/spreadsheets/d/1Lm4C27XX9rIMT8V1Z4oFHwit_mXNk2LdqvNUXhKoYS0/export?format=csv&gid=0")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 9600))
MAX_CONCURRENT_REQUESTS = 100
REQUEST_TIMEOUT = 120

# --- Model Routing Configuration ---
DEFAULT_MISTRAL_MODEL = "mistral-medium-latest"
DEFAULT_MISTRAL_VISION_MODEL = "mistral-large-latest"
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest"

# Gemini and related Google models (CORRECTED & EXPANDED LIST)
GEMINI_MODEL_NAMES = {
    # Core Models
    "gemini-1.5-pro-latest",
    "gemini-pro",
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash", # Added
    "gemini-2.5-pro",   # Added
    "gemini-2.5-flash", # Added
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash-001",
    # OpenAI compatible names routed to Gemini
    "gpt-4o",
    "gpt-4-vision-preview",
}

# Mistral models with vision/multimodal capabilities
MISTRAL_VISION_MODELS = {
    "mistral-large-latest",
    "mistral-medium-latest",
    "pixtral-large-latest" # Added for clarity,
    "mistral-medium-2505",
    "pixtral-12B-latest",
    "pixtral-12b-2409",
    "pixtral-large-2411"
}
ALL_VISION_MODELS = GEMINI_MODEL_NAMES.union(MISTRAL_VISION_MODELS)

# --- Global State Variables ---
mistral_key_manager: Optional[MistralApiKeyManager] = None
mistral_rate_limiter: Optional[AsyncTokenBucket] = None
gemini_rate_limiter: Optional[AsyncTokenBucket] = None
gemini_key_manager: Optional[GeminiApiKeyManager] = None
gemini_generator: Optional[GeneratorGemini] = None
# request_semaphore has been removed as the key managers now handle concurrency.

# ==============================================================================
# 5. Real-Time Metrics & Dashboard
# ==============================================================================
class RealTimeMetrics:
    def __init__(self):
        self.active_requests, self.request_history = {}, deque(maxlen=1000)
        self.stats = defaultdict(float, {"uptime": time.time(), "current_load": 0.0})
        self.response_times, self.minute_buckets = deque(maxlen=100), defaultdict(int)
        self.error_types, self.model_usage = defaultdict(int), defaultdict(int)
        self.websocket_clients = set()
    def start_request(self, request_id: str, model: str, has_images: bool = False):
        self.active_requests[request_id] = {"start_time": time.time(), "model": model, "has_images": has_images}
        self.stats["active_count"] = len(self.active_requests)
        self.stats["peak_concurrent"] = max(self.stats["peak_concurrent"], self.stats["active_count"])
        self.model_usage[model] += 1
        if has_images: self.stats["multimodal_requests"] += 1
    def end_request(self, request_id: str, success: bool, error_type: Optional[str] = None):
        if request_id not in self.active_requests: return
        req_info = self.active_requests.pop(request_id)
        response_time = time.time() - req_info["start_time"]
        self.stats["total_requests"] += 1
        self.stats["successful_requests"] += success
        self.stats["failed_requests"] += not success
        if error_type: self.error_types[error_type] += 1
        self.response_times.append(response_time)
        self.stats["avg_response_time"] = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        self.request_history.append({"timestamp": time.time(), "response_time": response_time, "success": success, "model": req_info["model"], "has_images": req_info["has_images"], "error_type": error_type})
        self.minute_buckets[int(time.time() // 60)] += 1
        recent_minutes = [self.minute_buckets.get(int(time.time() // 60) - i, 0) for i in range(5)]
        self.stats["requests_per_minute"] = sum(recent_minutes) / 5.0
        self.stats["active_count"] = len(self.active_requests)
    def get_dashboard_data(self):
        uptime_seconds = time.time() - self.stats["uptime"]
        total_req = max(self.stats["total_requests"], 1)
        self.stats["mistral_keys_available"] = mistral_key_manager.get_active_key_count() if mistral_key_manager else 0
        self.stats["gemini_keys_available"] = gemini_key_manager.get_active_key_count() if gemini_key_manager else 0
        dashboard_stats = {
            **self.stats, "uptime_formatted": str(timedelta(seconds=int(uptime_seconds))),
            "success_rate": (self.stats["successful_requests"] / total_req) * 100,
            "error_rate": (self.stats["failed_requests"] / total_req) * 100,
            "multimodal_percentage": (self.stats["multimodal_requests"] / total_req) * 100
        }
        active_reqs = [{"id": req_id, "duration": time.time() - data["start_time"], **data} for req_id, data in self.active_requests.items()]
        recent_perf = [{"timestamp": r["timestamp"] * 1000, "response_time": r["response_time"] * 1000, "success": r["success"], "has_images": r.get("has_images", False)} for r in list(self.request_history)[-50:]]
        return {"stats": dashboard_stats, "active_requests": active_reqs, "recent_performance": recent_perf, "error_breakdown": dict(self.error_types), "model_usage": dict(self.model_usage), "timestamp": time.time() * 1000}

metrics = RealTimeMetrics()

# ==============================================================================
# 6. Helper Functions and Parsers
# ==============================================================================

def is_base64_image(data_url: str) -> bool:
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"): return False
    try:
        base64.b64decode(data_url.split("base64,")[1], validate=True); return True
    except: return False

def parse_openai_messages(messages: List[Dict]) -> tuple[List[Dict], List[Image.Image], bool]:
    cleaned_messages, pil_images, has_images = [], [], False
    for msg in messages:
        if isinstance(msg.get("content"), list):
            text_parts = []
            for part in msg["content"]:
                if part.get("type") == "text": text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if is_base64_image(url):
                        has_images = True
                        try: pil_images.append(Image.open(io.BytesIO(base64.b64decode(url.split("base64,")[1]))).convert('RGB'))
                        except Exception as e: console.print(f"[Parser] Failed to decode Base64 image: {e}", style="red")
            cleaned_messages.append({"role": msg.get("role"), "content": " ".join(text_parts)})
        else: cleaned_messages.append(msg)
    return cleaned_messages, pil_images, has_images

def generate_simple_id(length_bytes: int = 9) -> str:
    return base64.urlsafe_b64encode(os.urandom(length_bytes)).rstrip(b'=').decode("ascii")

def create_openai_response(response_text: str, model: str) -> Dict[str, Any]:
    return {"id": generate_simple_id(16), "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

def create_mistral_response_dict(chat_response, model: str) -> Dict[str, Any]:
    choice = chat_response.choices[0] if hasattr(chat_response, "choices") and chat_response.choices else None
    usage = chat_response.usage if hasattr(chat_response, "usage") else None
    return {"id": getattr(chat_response, "id", ""), "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": getattr(choice.message, "content", "")}, "finish_reason": getattr(choice, "finish_reason", "stop")}] if choice else [], "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", 0), "completion_tokens": getattr(usage, "completion_tokens", 0), "total_tokens": getattr(usage, "total_tokens", 0)} if usage else {}}

# ==============================================================================
# 7. Core API Call Processors (REFACTORED)
# ==============================================================================
async def _execute_mistral_request(request_id: str, api_call_func, *args, **kwargs):
    """A generic wrapper for executing any Mistral API call with the explicit lock/release pattern."""
    session_key_info = kwargs.pop('session_key_info', None)

    while True:
        key_info = None
        try:
            # 1. SELECT and LOCK a key for this attempt.
            # If it's a session-based call, we use the provided key, otherwise we get one from the pool.
            key_info = session_key_info or mistral_key_manager.select_key()
            console.print(f"[{request_id}] Attempting with locked key '[cyan]{key_info['name']}[/cyan]'.", style="yellow")

            # 2. EXECUTE the API call.
            await mistral_rate_limiter.acquire()
            client = Mistral(api_key=key_info['token'])
            result = await asyncio.to_thread(api_call_func, client, *args, **kwargs)

            console.print(f"[{request_id}] Request successful with key '[cyan]{key_info['name']}[/cyan]'.", style="green")
            return result, key_info # Success! Exit the loop.

        except Exception as e:
            # 3. HANDLE failure.
            error_msg = str(e).lower()
            if any(k in error_msg for k in ["quota", "rate limit", "exceeded", "permission_denied", "unauthorized", "invalid api key"]):
                console.print(f"[{request_id}] Auth/Quota error on key '[cyan]{key_info['name']}[/cyan]'. Retiring it.", style="bold red")
                if key_info:
                    mistral_key_manager.retire_key(key_info)
                if session_key_info: # If a pinned session key fails, the whole operation must fail.
                    raise ApiKeyQuotaExceededError(f"Session key {key_info['name']} is invalid or exhausted.")
                continue # Loop to try again with a new key.
            else:
                console.print(f"[{request_id}] Unexpected API error: {e}", style="bold red")
                raise # Re-raise other errors to be caught by the web handler.

        finally:
            # 4. ALWAYS RELEASE/UNLOCK the key for this attempt (unless it's a pinned session key).
            if key_info and not session_key_info:
                mistral_key_manager.release_key(key_info)
                console.print(f"[{request_id}] Released key '[cyan]{key_info['name']}[/cyan]'.", style="dim")
        
async def process_gemini_call(request_data: Dict[str, Any]) -> Dict[str, Any]:
    request_id, model, messages = f"req_gemini_{int(time.time() * 1000)}", request_data.get("model", DEFAULT_GEMINI_MODEL), request_data.get("messages", [])
    if not messages: return {"status": "error", "error_message": "'messages' field is required"}
    
    cleaned_messages, pil_images, has_images = parse_openai_messages(messages)
    metrics.start_request(request_id, model, has_images)
    
    try:
        # MODIFIED: Await the now-async generate method directly.
        response_text = await gemini_generator.generate(
            messages=cleaned_messages, images=pil_images, model_name=model,
            max_new_tokens=request_data.get("max_tokens"), temperature=request_data.get("temperature")
        )
        response_dict = create_openai_response(response_text, model)
        metrics.end_request(request_id, True)
        return {"status": "success", "data": response_dict}
    except Exception as e:
        console.print(f"[Gemini Processor] [bold red]Error processing request {request_id}: {e}[/bold red]")
        console.print_exception()
        metrics.end_request(request_id, False, "gemini_api_error")
        return {"status": "error", "error_message": f"Gemini request failed: {e}"}

async def process_mistral_call(request_data: Dict[str, Any]) -> Dict[str, Any]:
    request_id = f"req_mistral_{int(time.time() * 1000)}"
    model = request_data.get("model", DEFAULT_MISTRAL_MODEL)
    messages = request_data.get("messages", [])
    if not messages: return {"status": "error", "error_message": "'messages' field is required"}
    
    # Mistral-specific message conversion for vision
    converted_messages, has_images = [], False
    for msg in messages:
        if isinstance(msg.get('content'), list):
            has_images = True
            mistral_content = [{"type": "text", "text": p.get("text", "")} if p['type'] == 'text' else {"type": "image_url", "image_url": p.get("image_url", {}).get("url")} for p in msg['content']]
            converted_messages.append({"role": msg['role'], "content": mistral_content})
        else: converted_messages.append(msg)
    
    if has_images and model not in MISTRAL_VISION_MODELS: model = DEFAULT_MISTRAL_VISION_MODEL
    
    metrics.start_request(request_id, model, has_images)
    try:
        def api_call(client, **params): return client.chat.complete(**params)
        params = {"model": model, "messages": converted_messages}
        for p in ["temperature", "max_tokens", "top_p"]:
            if p in request_data: params[p] = request_data[p]
        chat_response, _ = await _execute_mistral_request(request_id, api_call, **params)
        metrics.end_request(request_id, True)
        return {"status": "success", "data": create_mistral_response_dict(chat_response, request_data.get("model", DEFAULT_MISTRAL_MODEL))}
    except Exception as e:
        metrics.end_request(request_id, False, "mistral_api_error")
        return {"status": "error", "error_message": f"Mistral request failed: {e}"}

# All other Mistral processors will now use the generic wrapper
async def process_mistral_ocr_call(request_data: Dict[str, Any]) -> Dict[str, Any]:
    request_id = f"req_ocr_{int(time.time() * 1000)}"
    model = request_data.get("model", "mistral-ocr-latest")
    image_url = request_data.get("image")
    if not image_url or not is_base64_image(image_url): return {"status": "error", "error_message": "A valid base64 'image' data URL is required."}
    metrics.start_request(request_id, model, True)
    try:
        def api_call(client, **kwargs): return client.ocr.process(**kwargs)
        document_payload = {"type": "image_url", "image_url": image_url}
        ocr_response, _ = await _execute_mistral_request(request_id, api_call, model=model, document=document_payload)
        metrics.end_request(request_id, True)
        return {"status": "success", "data": ocr_response.model_dump()}
    except Exception as e:
        metrics.end_request(request_id, False, "mistral_ocr_api_error")
        return {"status": "error", "error_message": f"Mistral OCR request failed: {e}"}

async def process_mistral_file_upload(request_id: str, file_name: str, file_content: bytes, purpose: str) -> Dict[str, Any]:
    metrics.start_request(request_id, f"file-upload-{purpose}", False)
    try:
        def api_call(client, **kwargs): return client.files.upload(**kwargs)
        upload_response, key_info = await _execute_mistral_request(request_id, api_call, file=File(file_name=file_name, content=file_content), purpose=purpose)
        
        response_data = upload_response.model_dump()
        response_data['key_info_for_session'] = key_info # Pass key info for subsequent batch calls
        
        metrics.end_request(request_id, True)
        return {"status": "success", "data": response_data}
    except Exception as e:
        metrics.end_request(request_id, False, "mistral_file_api_error")
        return {"status": "error", "error_message": f"Mistral file upload failed: {e}"}

async def process_mistral_batch_job_create(request_id: str, batch_data: Dict[str, Any]) -> Dict[str, Any]:
    session_key_info = batch_data.pop('key_info_for_session', None)
    model = batch_data.get("model")
    metrics.start_request(request_id, f"batch-create-{model}", False)
    try:
        def api_call(client, **kwargs): return client.batch.jobs.create(**kwargs)
        response, _ = await _execute_mistral_request(request_id, api_call, session_key_info=session_key_info, **batch_data)
        metrics.end_request(request_id, True)
        return {"status": "success", "data": response.model_dump()}
    except Exception as e:
        metrics.end_request(request_id, False, "mistral_batch_api_error")
        return {"status": "error", "error_message": f"Mistral batch job creation failed: {e}"}

# ... other process functions (download, get, list, cancel) would be refactored similarly ...
# For brevity, I'll show one more example for 'get'
async def process_mistral_batch_job_get(request_id: str, job_id: str) -> Dict[str, Any]:
    metrics.start_request(request_id, "batch-get", False)
    try:
        def api_call(client, **kwargs): return client.batch.jobs.get(**kwargs)
        response, _ = await _execute_mistral_request(request_id, api_call, job_id=job_id)
        metrics.end_request(request_id, True)
        return {"status": "success", "data": response.model_dump()}
    except Exception as e:
        metrics.end_request(request_id, False, "mistral_batch_api_error")
        return {"status": "error", "error_message": f"Mistral batch job retrieval failed: {e}"}
    
async def process_gemini_transcription_call(
    # session_key: Optional[str], # NEW
    file_name: str,
    file_content: bytes,
    mime_type: str,
    model: str,
    prompt: Optional[str],
    language: Optional[str]
) -> Dict[str, Any]:
    
    request_id = f"req_transcribe_{int(time.time() * 1000)}"
    # model = "gemini-1.5-pro-latest" # Transcription is best with this model
    metrics.start_request(request_id, model, has_images=False) # Or a new category?
    
    try:
        # MODIFIED: Pass the file_name to the generator method
        transcribed_text = await gemini_generator.transcribe_media(
            # session_key=session_key,  # Use the session key if provided
            file_content=file_content,
            mime_type=mime_type,
            file_name=file_name,
            model=model,
            prompt=prompt,
            language=language
        )
        # Format the response to be compatible with OpenAI's transcription object.
        response_data = {"text": transcribed_text}
        metrics.end_request(request_id, True)
        return {"status": "success", "data": response_data}
    except Exception as e:
        console.print(f"[Gemini Transcription] [bold red]Error processing request {request_id}: {e}[/bold red]")
        console.print_exception()
        metrics.end_request(request_id, False, "gemini_transcription_error")
        return {"status": "error", "error_message": f"Gemini transcription failed: {e}"}        
# ==============================================================================
# 8. Web Handlers and Application Setup
# ==============================================================================

async def handle_chat_completions(request: web.Request) -> web.Response:
    try: request_data = await request.json()
    except json.JSONDecodeError: return web.json_response({"error": "Invalid JSON body."}, status=400)
    if request_data.get("stream"): return web.json_response({"error": "Streaming not supported."}, status=400)
    
    model_name = request_data.get("model", "").lower()
    if model_name in GEMINI_MODEL_NAMES or model_name.startswith("gemini-"):
        console.print(f"➡️ Routing request to [bold purple]Gemini[/bold purple] for model: {model_name}")
        result = await process_gemini_call(request_data)
    elif model_name in MISTRAL_VISION_MODELS or model_name.startswith("mistral-") or model_name.startswith("pixtral-") or model_name.startswith("codestral-"):
        console.print(f"➡️ Routing request to [bold blue]Mistral[/bold blue] for model: {model_name or DEFAULT_MISTRAL_MODEL}")
        result = await process_mistral_call(request_data)
        
    if result["status"] == "success": return web.json_response(result["data"])
    else: return web.json_response({"error": {"message": result.get("error_message"), "type": "api_error"}}, status=500)

async def handle_ocr(request: web.Request) -> web.Response:
    try: request_data = await request.json()
    except json.JSONDecodeError: return web.json_response({"error": "Invalid JSON body."}, status=400)
    result = await process_mistral_ocr_call(request_data)
    if result["status"] == "success": return web.json_response(result["data"])
    else: return web.json_response({"error": {"message": result.get("error_message"), "type": "api_error"}}, status=500)

async def handle_mistral_file_upload(request: web.Request) -> web.Response:
    request_id = f"req_file_upload_{int(time.time() * 1000)}"
    try:
        reader = await request.multipart()
        file_field = await reader.next()
        purpose_field = await reader.next()
        if not file_field or file_field.name != 'file' or not purpose_field or purpose_field.name != 'purpose':
            return web.json_response({"error": "multipart/form-data must contain 'file' and 'purpose' parts."}, status=400)
        
        file_name, file_content = file_field.filename, await file_field.read()
        purpose = (await purpose_field.read()).decode('utf-8')
        result = await process_mistral_file_upload(request_id, file_name, file_content, purpose)
        if result["status"] == "success": return web.json_response(result["data"])
        else: return web.json_response({"error": {"message": result.get("error_message"), "type": "api_error"}}, status=500)
    except Exception as e: return web.json_response({"error": f"Error handling file upload: {e}"}, status=500)

# async def handle_transcribe_begin_session(request: web.Request) -> web.Response:
#     try:
#         # select_key already marks the key as "in_use"
#         key_info = await asyncio.to_thread(gemini_key_manager.select_key)
#         log_key_display = f"{key_info['name']}-{key_info['token'][:5]}...{key_info['token'][-7:]}"
#         console.print(f"🔑 Started transcription session with key: [green]{log_key_display}[/green]", style="cyan")
#         return web.json_response({"session_key": key_info['token']})
#     except Exception as e:
#         return web.json_response({"error": f"Could not acquire a key for session: {e}"}, status=503)

# # NEW HANDLER
    
async def handle_transcriptions(request: web.Request) -> web.Response:
    try:
        reader = await request.multipart()
        
        part_data = {}
        file_content = None
        file_name = None
        mime_type = None
        # session_key = None
        async for part in reader:
            # --- CORRECTED PARSING LOGIC ---
            if part.filename:
                file_content = await part.read()
                file_name = part.filename
                # MODIFIED: Get content_type from the part's headers dictionary.
                # The .get() method is used for safety in case the header is missing.
                mime_type = part.headers.get('Content-Type')
            else:
                field_name = part.name
                field_value = (await part.read()).decode('utf-8')
                part_data[field_name] = field_value
            # --- END OF MODIFICATION ---
        
        if not file_content:
            return web.json_response({"error": "Missing 'file' part in multipart/form-data"}, status=400)
            
        # Prioritize explicit MIME type from form data if client sends it.
        # This is a good fallback if the part.content_type is not reliable.
        if 'mime_type' in part_data:
            mime_type = part_data['mime_type']
            console.print(f"ℹ️ Received explicit MIME type from client: [cyan]{mime_type}[/cyan]")
        
        if not mime_type:
             mime_type, _ = mimetypes.guess_type(file_name) # Last resort
        
        console.print(f"ℹ️ Received file '{file_name}' with final MIME type: [cyan]{mime_type}[/cyan]")

        if not mime_type or not (mime_type.startswith("audio/") or mime_type.startswith("video/")):
             return web.json_response({"error": f"Unsupported file type. Please upload a valid audio or video file. Detected: {mime_type}"}, status=400)
        
        # Now we get the rest of the data from the parsed part_data dict
        # session_key = part_data.get("session_key")
        model_to_use = part_data.get('model', "gemini-1.5-pro-latest")
        prompt = part_data.get('prompt')
        language = part_data.get('language')

        console.print(f"➡️ Routing request to [bold purple]Gemini Transcription[/bold purple] for file: {file_name} using model '{model_to_use}'")
        
        result = await process_gemini_transcription_call(
            # session_key=session_key,
            file_name=file_name,
            file_content=file_content,
            mime_type=mime_type,
            model=model_to_use,
            prompt=prompt,
            language=language
        )

        if result.get("status") == "success":
            return web.json_response(result["data"])
        else:
            error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
            return web.json_response(error_resp, status=500)

    except Exception as e:
        console.print_exception()
        return web.json_response({"error": f"Error handling file upload for transcription: {e}"}, status=500)
        
async def handle_mistral_batch_jobs_create(request: web.Request) -> web.Response:
    request_id = f"req_batch_create_{int(time.time() * 1000)}"
    try: request_data = await request.json()
    except json.JSONDecodeError: return web.json_response({"error": "Invalid JSON body."}, status=400)
    result = await process_mistral_batch_job_create(request_id, request_data)
    if result["status"] == "success": return web.json_response(result["data"])
    else: return web.json_response({"error": {"message": result.get("error_message"), "type": "api_error"}}, status=500)

# Other handlers like get/list/cancel batch jobs, download file, etc. would be implemented here
# They would all call their respective `process_*` function. For brevity, they are omitted but follow the same pattern.
async def handle_mistral_batch_job_details(request: web.Request) -> web.Response:
    job_id = request.match_info.get("job_id")
    if not job_id: return web.json_response({"error": "Job ID required."}, status=400)
    request_id = f"req_batch_get_{int(time.time() * 1000)}"
    result = await process_mistral_batch_job_get(request_id, job_id)
    if result["status"] == "success": return web.json_response(result["data"])
    else: return web.json_response({"error": {"message": result.get("error_message"), "type": "api_error"}}, status=500)

async def handle_health(request: web.Request) -> web.Response:
    data = metrics.get_dashboard_data()
    return web.json_response({
        "status": "healthy",
        "mistral_keys_in_pool": data["stats"]["mistral_keys_available"],
        "gemini_keys_in_pool": data["stats"]["gemini_keys_available"],
        "vision_support": True
    })
# The dashboard HTML and WebSocket logic remains the same.
async def handle_dashboard(request: web.Request) -> web.Response:
    dashboard_html = """...""" # HTML is large, omitting for brevity
    return web.Response(text=dashboard_html, content_type='text/html')

async def handle_dashboard_websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    metrics.websocket_clients.add(ws)
    try:
        await ws.send_str(json.dumps(metrics.get_dashboard_data()))
        async for msg in ws:
            if msg.type == WSMsgType.ERROR: break
    finally:
        metrics.websocket_clients.discard(ws)
    return ws
    
async def broadcast_metrics_updates():
    while True:
        await asyncio.sleep(1.0)
        if not metrics.websocket_clients: continue
        try:
            message = json.dumps(metrics.get_dashboard_data())
            disconnected = {ws for ws in metrics.websocket_clients if ws.closed}
            metrics.websocket_clients -= disconnected
            await asyncio.gather(*(ws.send_str(message) for ws in metrics.websocket_clients), return_exceptions=False)
        except Exception: pass

async def on_shutdown(app: web.Application):
    console.print("--- Server shutting down ---", style="bold yellow")
    if mistral_key_manager: mistral_key_manager.shutdown()
    if gemini_key_manager: gemini_key_manager.shutdown()
    console.print("--- Shutdown complete ---", style="bold green")

async def init_app() -> web.Application:
    global mistral_key_manager, mistral_rate_limiter, gemini_key_manager, gemini_generator, gemini_rate_limiter

    console.print("🚀 [bold]Starting Unified Gemini-Mistral API Gateway...[/bold]")
    loop = asyncio.get_running_loop()

    # --- Initialize Mistral Components ---
    console.print("🔄 [Mistral] Initializing components...", style="blue")
    try:
        # Here we tell the manager that each key can be used ~2 times per second.
        mistral_key_manager = MistralApiKeyManager(
            sheet_url=MISTRAL_SHEET_URL,
            key_rpm=110
        )
        mistral_rate_limiter = AsyncTokenBucket(MISTRAL_RPM_RATE, MISTRAL_RPM_CAPACITY, loop)
        console.print(f"✅ [Mistral] Components initialized.", style="green")
    except Exception as e:
        console.print(f"❌ [bold red]CRITICAL: Failed to initialize Mistral: {e}[/bold red]")

    # --- Initialize Gemini Components ---
    console.print("🔄 [Gemini] Initializing components...", style="purple")
    if genai:
        try:
            # You can set a different rate limit for Gemini keys if needed.
            gemini_key_manager = GeminiApiKeyManager(
                sheet_url=GEMINI_SHEET_URL,
                key_rpm=900 
            )
            gemini_rate_limiter = AsyncTokenBucket(
                rate=GEMINI_RPM_RATE,
                capacity=GEMINI_RPM_CAPACITY,
                loop=loop
            )
            gemini_generator = GeneratorGemini(api_key_manager=gemini_key_manager)
            console.print("✅ [Gemini] Components initialized successfully.", style="green")
        except Exception as e:
            console.print(f"❌ [bold red]CRITICAL: Failed to initialize Gemini: {e}[/bold red]")
    else:
        console.print("⚠️ [Gemini] 'google-generativeai' not installed. Gemini backend is disabled.", style="yellow")


    app = web.Application(client_max_size=1024**2 * 100)
    # app.router.add_post("/v1/audio/transcribe/begin_session", handle_transcribe_begin_session)
    app.router.add_post("/v1/audio/transcriptions", handle_transcriptions)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/ocr", handle_ocr)
    app.router.add_post("/v1/files", handle_mistral_file_upload)
    app.router.add_post("/v1/batch/jobs", handle_mistral_batch_jobs_create)
    app.router.add_get("/v1/batch/jobs/{job_id}", handle_mistral_batch_job_details)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/", lambda r: web.HTTPFound('/dashboard'))
    app.router.add_get("/dashboard", handle_dashboard)
    app.router.add_get("/ws", handle_dashboard_websocket)
    app.on_shutdown.append(on_shutdown)
    return app

async def main():
    if os.name != 'nt':
        try: 
            import uvloop
            uvloop.install()
            console.print("🏃 Using [cyan]uvloop[/cyan].")
        except ImportError: 
            pass
    
    app = await init_app()
    
    # --- CORRECTED STARTUP SEQUENCE ---
    # 1. Create the AppRunner instance
    runner = web.AppRunner(app)
    
    # 2. Set up the runner
    await runner.setup()
    
    # 3. Create the TCPSite using the *same*, now-setup runner instance
    site = web.TCPSite(runner, HOST, PORT)
    # --- END OF CORRECTION ---

    metrics_task = asyncio.create_task(broadcast_metrics_updates())
    
    try:
        # 4. Start the site (runner.setup() is already done)
        await site.start()
        console.print(f"✅ [bold green]Server is running at http://{HOST}:{PORT}[/bold green]")
        console.print("   Press Ctrl+C to stop.")
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n🛑 Stopping server...", style="yellow")
        metrics_task.cancel()
        await runner.cleanup()
        console.print("👋 Server shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"❌ [bold red]Failed to start server: {e}[/bold red]")
