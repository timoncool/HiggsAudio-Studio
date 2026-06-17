"""AI-режиссёр текста — выполняется в ОТДЕЛЬНОМ GPU-процессе (изоляция CUDA-рантайма от torch).

Почему процесс, а не in-process: llama.cpp (сборка cu124) и torch (cu126/cu128) в одном процессе
делят cublas64_12.dll по ИМЕНИ — Windows держит один модуль на процесс, версии разъезжаются →
"CUDA error: invalid argument". Подобрать совпадающие сборки нельзя (у abetlen llama макс cu124,
у torch 2.7.1 нет cu124). Поэтому режиссёр живёт в своём процессе со своим cublas 12.4.
РАБОТАЕТ НА GPU (n_gpu_layers=-1) — отдельный процесс != CPU, скорость полная. Воркер
короткоживущий: грузит модель, отвечает и завершается, освобождая VRAM перед TTS.

filter_tags / WHITELIST / MODELS — чистый Python (без llama/GPU), нужны UI и тестам.
"""
import os
import re
import json
import subprocess
import sys
import threading

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
    """Оставить ТОЛЬКО валидные <|cat:val|> из белого списка; вырезать прочие угловые конструкции."""
    if not text:
        return text

    def repl(m):
        s = m.group(0)
        v = _VALID.fullmatch(s)
        if v and v.group(2) in WHITELIST.get(v.group(1), ()):
            return s
        return ""

    return _ANGLE.sub(repl, text)


MODELS = {  # GGUF (llama.cpp, GPU): качается 4-бит Q4_K_M
    "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    "Qwen3.5-4B · Q4_K_M (лёгкая, ~2.5 ГБ)": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
}
DEFAULT_MODEL = "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)"

_lock = threading.Lock()


def _call(action, text, label=DEFAULT_MODEL, n=2):
    """Запустить воркер режиссёра (отдельный GPU-процесс), отдать запрос, получить результат.
    Воркер завершается сам → его VRAM и CUDA-контекст освобождаются до старта TTS."""
    if _MOCK:
        return text
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "director_worker.py")
    req = json.dumps({"action": action, "text": text, "label": label, "n": n}, ensure_ascii=False)
    # GPU НЕ глушим — режиссёр на GPU. В этом процессе нет torch, llama берёт свой cublas 12.4.
    # stderr → консоль (наследуется): логи и прогресс скачивания видны, пайп не копится.
    with _lock:  # один воркер за раз — не грузим две модели на GPU параллельно
        proc = subprocess.run(
            [sys.executable, script],
            input=req, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", env=os.environ.copy(),
        )
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError(f"режиссёр-воркер не ответил (код {proc.returncode}) — смотри лог в консоли")
    data = json.loads(out.splitlines()[-1])  # последняя строка stdout — JSON-ответ
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "режиссёр: неизвестная ошибка"))
    return data["result"]


def enrich(text, label=DEFAULT_MODEL):
    """РОЛЬ A — нормализация под произношение + лёгкая правка + теги по смыслу."""
    return _call("enrich", text, label)


def write_podcast(topic, n_speakers=2, label=DEFAULT_MODEL):
    """РОЛЬ B — мульти-спикерный диалог в индексном формате 'Speaker N: реплика'."""
    return _call("podcast", topic, label, n=max(2, int(n_speakers)))


def cast_audiobook(text, n_voices=2, label=DEFAULT_MODEL):
    """РОЛЬ C — атрибуция: Speaker 0 = рассказчик, 1.. = персонажи."""
    return _call("audiobook", text, label, n=max(2, int(n_voices)))
