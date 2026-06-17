"""AI-режиссёр текста: нормализация+теги (роль A), сценарий подкаста (B), кастинг аудиокниги (C).

LLM — Qwen3.5 GGUF (4-бит Q4_K_M) через llama.cpp на GPU (n_gpu_layers=-1).
CUDA-DLL кладутся install.bat рядом с llama.dll, поэтому импорт без трюков.
Фильтр тегов и тесты работают без llama.cpp/GPU (ленивые импорты).
"""
import os
import re

# llama.cpp по умолчанию пиннит host-память (cudaHostRegister); в одном процессе с pinned-
# аллокатором torch это даёт CUDA einval. Глушим ДО импорта llama_cpp (важно — до Llama()).
os.environ.setdefault("GGML_CUDA_NO_PINNED", "1")

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
    """Оставить ТОЛЬКО валидные <|cat:val|> из белого списка; вырезать любые иные угловые
    конструкции, включая кривые теги модели (<speed_1.2>, <emotion:excited>, <sfx:wind>)."""
    if not text:
        return text

    def repl(m):
        s = m.group(0)
        v = _VALID.fullmatch(s)
        if v and v.group(2) in WHITELIST.get(v.group(1), ()):
            return s
        return ""
    return _ANGLE.sub(repl, text)


MODELS = {  # GGUF (llama.cpp, GPU): качается 4-бит Q4_K_M, НЕ полный bf16
    "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    "Qwen3.5-4B · Q4_K_M (лёгкая, ~2.5 ГБ)": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
}
DEFAULT_MODEL = "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)"


def _tag_spec():
    return (
        "Разрешены ТОЛЬКО эти теги, строго в формате <|категория:значение|> (с вертикальными чертами):\n"
        "- emotion (в начале предложения): " + ", ".join(sorted(WHITELIST["emotion"])) + "\n"
        "- prosody: " + ", ".join(sorted(WHITELIST["prosody"])) + " (pause/long_pause — внутри строки)\n"
        "- style (в начале предложения): " + ", ".join(sorted(WHITELIST["style"])) + "\n"
        "- sfx (внутри строки, вплотную к звукоподражанию): " + ", ".join(sorted(WHITELIST["sfx"])) + "\n"
        "НЕ выдумывай другие теги и значения. ЗАПРЕЩЕНО писать <speed_1.2>, <emotion:excited>, <sfx:wind> — "
        "только значения из списка и только в формате <|категория:значение|>.\n"
        "Пример: <|emotion:elation|>Поздравляю всех! <|sfx:laughter|>ха-ха. <|prosody:long_pause|> Продолжаем.\n"
        "Верни ТОЛЬКО готовый текст, без пояснений и преамбул."
    )


_TAG_RULES = _tag_spec()

_llm = None
_cur = None


def load_llm(label=DEFAULT_MODEL):
    """Скачать GGUF (если нужно) и поднять llama.cpp на GPU (все слои)."""
    global _llm, _cur
    if _MOCK:
        return "MOCK"
    if _cur == label and _llm is not None:
        return _llm
    unload_llm()
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    repo, fname = MODELS[label]
    print(f"[director] загрузка {label} ({repo}/{fname})...")
    path = hf_hub_download(repo, fname)
    _llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=8192, verbose=False)
    _cur = label
    return _llm


def unload_llm():
    global _llm, _cur
    _llm = None
    _cur = None


def _strip_think(txt):
    return re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip()


def _chat(system, user, max_new=1024, temp=0.4, label=DEFAULT_MODEL):
    if _MOCK:
        return user  # в mock — проброс без изменений
    llm = load_llm(label)
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_new, temperature=temp, top_p=0.9)
    return _strip_think(out["choices"][0]["message"].get("content", ""))


def _filter_line(line):
    """Сохранить префикс 'ИМЯ:', отфильтровать теги в произносимой части."""
    if ":" in line:
        who, _, said = line.partition(":")
        return f"{who}:{filter_tags(said)}"
    return filter_tags(line)


def enrich(text, label=DEFAULT_MODEL):
    """РОЛЬ A — нормализация под произношение + лёгкая правка + теги по смыслу."""
    s = ("Ты — режиссёр озвучки. Нормализуй текст под произношение (числа, даты, аббревиатуры, "
         "валюты, единицы, символы — словами), исправь явные опечатки, и расставь эмоциональные / "
         "sfx / prosody-теги по смыслу. " + _TAG_RULES)
    return filter_tags(_chat(s, text, label=label))


def write_podcast(topic, n_speakers=2, label=DEFAULT_MODEL):
    """РОЛЬ B — мульти-спикерный диалог в индексном формате 'Speaker N: реплика' (как Qwen3-TTS)."""
    n = max(2, int(n_speakers))
    s = (f"Ты — сценарист подкаста на {n} спикеров (Speaker 0 .. Speaker {n - 1}). "
         "Напиши живой диалог: КАЖДАЯ строка строго в формате 'Speaker K: реплика', где K — номер от 0. "
         "Дай каждому спикеру свою манеру речи и характер. В репликах расставляй теги по смыслу. " + _TAG_RULES)
    out = _chat(s, topic, max_new=2048, label=label)
    return "\n".join(_filter_line(ln) for ln in out.splitlines())


def cast_audiobook(text, n_voices=2, label=DEFAULT_MODEL):
    """РОЛЬ C — атрибуция в индексном формате: Speaker 0 = рассказчик, 1.. = персонажи."""
    n = max(2, int(n_voices))
    s = ("Ты — кастинг-режиссёр аудиокниги. Раздели текст на речь рассказчика и реплики персонажей. "
         f"Speaker 0 — РАССКАЗЧИК (авторский текст), Speaker 1 .. Speaker {n - 1} — персонажи "
         "(закрепи за каждым персонажем свой номер и держи его постоянным). "
         "КАЖДАЯ строка строго в формате 'Speaker K: реплика'. Текст сохраняй ДОСЛОВНО, "
         "только размечай говорящего и добавляй теги по смыслу. " + _TAG_RULES)
    out = _chat(s, text, max_new=2048, label=label)
    return "\n".join(_filter_line(ln) for ln in out.splitlines())
