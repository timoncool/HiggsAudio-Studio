"""AI-режиссёр текста — RPC через subprocess (изоляция от PyTorch/CUDA)."""
import os
import re
import json
import subprocess
import sys
import threading
import atexit

_MOCK = bool(os.environ.get("HIGGS_UI_MOCK"))

WHITELIST = {
    "emotion": {"affection", "amusement", "anger", "arousal", "awe", "bitterness", "confusion",
                "contemplation", "contentment", "determination", "disgust", "elation", "enthusiasm",
                "fear", "helplessness", "longing", "pride", "relief", "sadness", "shame", "surprise"},
    "prosody": {"speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast", "pitch_low",
                "pitch_high", "expressive_high", "expressive_low", "pause", "long_pause"},
    "style": {"singing", "shouting", "whispering"},
    "sfx": {"cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze"},
}

_ANGLE = re.compile(r"<[^<>\n]{1,48}>")
_VALID = re.compile(r"<\|(\w+):(\w+)\|>")


def filter_tags(text):
    if not text:
        return text
    def repl(m):
        s = m.group(0)
        v = _VALID.fullmatch(s)
        if v and v.group(2) in WHITELIST.get(v.group(1), ()):
            return s
        return ""
    return _ANGLE.sub(repl, text)


MODELS = {
    "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    "Qwen3.5-4B · Q4_K_M (лёгкая, ~2.5 ГБ)": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
}
DEFAULT_MODEL = "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)"


# --- Subprocess worker ---
_worker = None
_lock = threading.Lock()


def _start_worker():
    global _worker
    if _worker is not None and _worker.poll() is None:
        return
    
    script = os.path.join(os.path.dirname(__file__), "director_worker.py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIGGS_UI_MOCK"] = "1" if _MOCK else ""
    
    _worker = subprocess.Popen(
        [sys.executable, script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
        bufsize=1
    )
    print(f"[director] воркер запущен (PID {_worker.pid})")


def _stop_worker():
    global _worker
    if _worker is not None:
        try:
            _worker.stdin.close()
            _worker.terminate()
            _worker.wait(timeout=5)
        except Exception:
            pass
        _worker = None


atexit.register(_stop_worker)


def _call(action, text, label=DEFAULT_MODEL, n=2):
    if _MOCK:
        return text
    
    with _lock:
        _start_worker()
        
        if _worker.poll() is not None:
            stderr = _worker.stderr.read() if _worker.stderr else ""
            raise RuntimeError(f"воркер упал, код {_worker.poll()}, stderr: {stderr[:500]}")
        
        cmd = json.dumps({
            "action": action,
            "text": text,
            "label": label,
            "n": n
        }, ensure_ascii=False)
        
        try:
            _worker.stdin.write(cmd + "\n")
            _worker.stdin.flush()
        except BrokenPipeError:
            stderr = _worker.stderr.read() if _worker.stderr else ""
            raise RuntimeError(f"воркер упал при записи, stderr: {stderr[:500]}")
        
        resp = _worker.stdout.readline()
        if not resp:
            stderr = _worker.stderr.read() if _worker.stderr else ""
            raise RuntimeError(f"воркер вернул пустой ответ, stderr: {stderr[:500]}")
        
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            raise RuntimeError(f"невалидный JSON: {resp[:200]}")
        
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "unknown error"))
        
        return data["result"]


def enrich(text, label=DEFAULT_MODEL):
    return _call("enrich", text, label)


def write_podcast(topic, n_speakers=2, label=DEFAULT_MODEL):
    return _call("podcast", topic, label, n=max(2, int(n_speakers)))


def cast_audiobook(text, n_voices=2, label=DEFAULT_MODEL):
    return _call("audiobook", text, label, n=max(2, int(n_voices)))