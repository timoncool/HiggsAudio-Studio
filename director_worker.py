"""Воркер режиссёра — отдельный процесс, llama.cpp на GPU (n_gpu_layers=-1).

Свой CUDA-контекст и свой cublas 12.4 (НЕ делит с torch) → "CUDA error: invalid argument"
исключён by design. Читает ОДИН JSON-запрос из stdin, печатает JSON-результат в stdout,
завершается (освобождая VRAM перед TTS). ВАЖНО: тут НЕ импортируется torch — иначе в процесс
заедет его cublas 12.8 и коллизия имени вернётся.
"""
import os
import sys
import json
import re
import traceback

# pinned-память llama даёт einval с pinned-аллокатором — глушим (на оффлоад слоёв не влияет).
os.environ.setdefault("GGML_CUDA_NO_PINNED", "1")
# CUDA_VISIBLE_DEVICES НЕ трогаем — режиссёр работает на GPU.

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

MODELS = {
    "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    "Qwen3.5-4B · Q4_K_M (лёгкая, ~2.5 ГБ)": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
}
DEFAULT_MODEL = "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)"


def log(msg):
    print(f"[режиссёр] {msg}", file=sys.stderr, flush=True)


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


def load_llm(label=DEFAULT_MODEL):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    repo, fname = MODELS[label]
    log(f"загрузка {label} на GPU...")
    path = hf_hub_download(repo, fname)
    return Llama(model_path=path, n_gpu_layers=-1, n_ctx=8192, verbose=False)


def _strip_think(txt):
    return re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip()


def _chat(llm, system, user, max_new=1024, temp=0.4):
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_new, temperature=temp, top_p=0.9)
    return _strip_think(out["choices"][0]["message"].get("content", ""))


# Director output language: "ru", "en", or "auto" (keep the input text's language).
# Instructions are phrased in the target language; "auto" phrases them in English and tells
# the model to mirror the input language instead of translating.
_TAG_RULES = {
    "ru": (
        "Разрешены ТОЛЬКО эти теги, строго в формате <|категория:значение|> (с вертикальными чертами):\n"
        "- emotion (в начале предложения): " + ", ".join(sorted(WHITELIST["emotion"])) + "\n"
        "- prosody: " + ", ".join(sorted(WHITELIST["prosody"])) + " (pause/long_pause — внутри строки)\n"
        "- style (в начале предложения): " + ", ".join(sorted(WHITELIST["style"])) + "\n"
        "- sfx (внутри строки, вплотную к звукоподражанию): " + ", ".join(sorted(WHITELIST["sfx"])) + "\n"
        "НЕ выдумывай другие теги и значения. ЗАПРЕЩЕНО писать <speed_1.2>, <emotion:excited>, <sfx:wind> — "
        "только значения из списка и только в формате <|категория:значение|>.\n"
        "Пример: <|emotion:elation|>Поздравляю всех! <|sfx:laughter|>ха-ха. <|prosody:long_pause|> Продолжаем.\n"
        "Верни ТОЛЬКО готовый текст, без пояснений и преамбул."
    ),
    "en": (
        "ONLY these tags are allowed, strictly in the format <|category:value|> (with vertical bars):\n"
        "- emotion (at the start of a sentence): " + ", ".join(sorted(WHITELIST["emotion"])) + "\n"
        "- prosody: " + ", ".join(sorted(WHITELIST["prosody"])) + " (pause/long_pause — inside the line)\n"
        "- style (at the start of a sentence): " + ", ".join(sorted(WHITELIST["style"])) + "\n"
        "- sfx (inside the line, right next to the onomatopoeia): " + ", ".join(sorted(WHITELIST["sfx"])) + "\n"
        "Do NOT invent other tags or values. It is FORBIDDEN to write <speed_1.2>, <emotion:excited>, <sfx:wind> — "
        "only values from the list and only in the format <|category:value|>.\n"
        "Example: <|emotion:elation|>Congratulations everyone! <|sfx:laughter|>ha-ha. <|prosody:long_pause|> Let's continue.\n"
        "Return ONLY the finished text, with no explanations or preamble."
    ),
}

_ENRICH_SYS = {
    "ru": ("Ты — режиссёр озвучки. Нормализуй текст под произношение (числа, даты, аббревиатуры, "
           "валюты, единицы, символы — словами), исправь явные опечатки, и расставь эмоциональные / "
           "sfx / prosody-теги по смыслу. "),
    "en": ("You are a voice-over director. Normalize the text for pronunciation (numbers, dates, "
           "abbreviations, currencies, units, symbols — spelled out in words), fix obvious typos, and add "
           "emotion / sfx / prosody tags according to the meaning. "),
}

_PODCAST_SYS = {
    "ru": ("Ты — сценарист подкаста на {n} спикеров (Speaker 0 .. Speaker {last}). "
           "Напиши живой диалог: КАЖДАЯ строка строго в формате 'Speaker K: реплика', где K — номер от 0. "
           "Дай каждому спикеру свою манеру речи и характер. В репликах расставляй теги по смыслу. "),
    "en": ("You are a podcast scriptwriter for {n} speakers (Speaker 0 .. Speaker {last}). "
           "Write a lively dialogue: EVERY line strictly in the format 'Speaker K: line', where K is a number from 0. "
           "Give each speaker their own speaking style and character. Add tags in the lines according to the meaning. "),
}

_AUDIOBOOK_SYS = {
    "ru": ("Ты — кастинг-режиссёр аудиокниги. Раздели текст на речь рассказчика и реплики персонажей. "
           "Speaker 0 — РАССКАЗЧИК (авторский текст), Speaker 1 .. Speaker {last} — персонажи "
           "(закрепи за каждым персонажем свой номер и держи его постоянным). "
           "КАЖДАЯ строка строго в формате 'Speaker K: реплика'. Текст сохраняй ДОСЛОВНО, "
           "только размечай говорящего и добавляй теги по смыслу. "),
    "en": ("You are a casting director for an audiobook. Split the text into narrator speech and character lines. "
           "Speaker 0 is the NARRATOR (authorial text), Speaker 1 .. Speaker {last} are characters "
           "(assign each character their own number and keep it constant). "
           "EVERY line strictly in the format 'Speaker K: line'. Keep the text VERBATIM, "
           "only mark the speaker and add tags according to the meaning. "),
}


def _norm_lang(lang):
    lang = (lang or "auto").lower()
    return lang if lang in ("ru", "en", "auto") else "auto"


def _base_lang(lang):
    """Language the director's own instructions are written in (auto → English)."""
    return "ru" if lang == "ru" else "en"


def _lang_clause(lang):
    """Explicit output-language directive appended to the system prompt."""
    if lang == "ru":
        return "Пиши на русском языке. "
    if lang == "en":
        return "Write in English. "
    return "Always write in the SAME language as the input text — never translate. "


def _filter_line(line):
    """Сохранить префикс 'ИМЯ:', отфильтровать теги в произносимой части."""
    if ":" in line:
        who, _, said = line.partition(":")
        return f"{who}:{filter_tags(said)}"
    return filter_tags(line)


def enrich(llm, text, lang="auto"):
    lang = _norm_lang(lang)
    b = _base_lang(lang)
    s = _ENRICH_SYS[b] + _lang_clause(lang) + _TAG_RULES[b]
    return filter_tags(_chat(llm, s, text))


def write_podcast(llm, topic, n_speakers=2, lang="auto"):
    lang = _norm_lang(lang)
    n = max(2, int(n_speakers))
    b = _base_lang(lang)
    s = _PODCAST_SYS[b].format(n=n, last=n - 1) + _lang_clause(lang) + _TAG_RULES[b]
    out = _chat(llm, s, topic, max_new=2048)
    return "\n".join(_filter_line(ln) for ln in out.splitlines())


def cast_audiobook(llm, text, n_voices=2, lang="auto"):
    lang = _norm_lang(lang)
    n = max(2, int(n_voices))
    b = _base_lang(lang)
    s = _AUDIOBOOK_SYS[b].format(last=n - 1) + _lang_clause(lang) + _TAG_RULES[b]
    out = _chat(llm, s, text, max_new=2048)
    return "\n".join(_filter_line(ln) for ln in out.splitlines())


def main():
    try:  # UTF-8 на pipe независимо от окружения (Windows-pipe иначе ANSI → кириллица бьётся)
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"плохой запрос: {e}"}, ensure_ascii=False))
        return
    action = req.get("action")
    text = req.get("text", "")
    label = req.get("label", DEFAULT_MODEL)
    n = req.get("n", 2)
    lang = req.get("lang", "auto")
    try:
        llm = load_llm(label)
        if action == "enrich":
            result = enrich(llm, text, lang)
        elif action == "podcast":
            result = write_podcast(llm, text, n, lang)
        elif action == "audiobook":
            result = cast_audiobook(llm, text, n, lang)
        else:
            result = text
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        sys.stdout.flush()
    except Exception as e:
        log(f"ОШИБКА: {e}")
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
