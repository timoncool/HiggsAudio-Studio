"""Воркер режиссёра"""
import os
import sys
import json
import re
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

# --- LLM ---
_llm = None
_cur = None

def log(msg):
    print(f"[WORKER] {msg}", file=sys.stderr, flush=True)

def load_llm(label=DEFAULT_MODEL):
    global _llm, _cur
    if _cur == label and _llm is not None:
        return _llm
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    repo, fname = MODELS[label]
    log(f"загрузка {label}...")
    path = hf_hub_download(repo, fname)
    _llm = Llama(model_path=path, n_gpu_layers=0, n_ctx=8192, verbose=False)
    _cur = label
    return _llm

def _strip_think(txt):
    return re.sub(r" thinking.*? end_thinking", "", txt or "", flags=re.S).strip()

def _chat(system, user, max_new=1024, temp=0.4, label=DEFAULT_MODEL):
    llm = load_llm(label)
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_new, temperature=temp, top_p=0.9)
    return _strip_think(out["choices"][0]["message"].get("content", ""))

_TAG_RULES = (
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

def enrich(text, label=DEFAULT_MODEL):
    s = ("Ты — режиссёр озвучки. Нормализуй текст под произношение (числа, даты, аббревиатуры, "
         "валюты, единицы, символы — словами), исправь явные опечатки, и расставь эмоциональные / "
         "sfx / prosody-теги по смыслу. " + _TAG_RULES)
    return filter_tags(_chat(s, text, label=label))

def write_podcast(topic, n_speakers=2, label=DEFAULT_MODEL):
    n = max(2, int(n_speakers))
    s = (f"Ты — сценарист подкаста на {n} спикеров (Speaker 0 .. Speaker {n - 1}). "
         "Напиши живой диалог: КАЖДАЯ строка строго в формате 'Speaker K: реплика', где K — номер от 0. "
         "Дай каждому спикеру свою манеру речи и характер. В репликах расставляй теги по смыслу. " + _TAG_RULES)
    out = _chat(s, topic, max_new=2048, label=label)
    return "\n".join(filter_tags(ln) if ":" not in ln else f"{ln.split(':')[0]}:{filter_tags(ln.split(':', 1)[1])}" for ln in out.splitlines())

def cast_audiobook(text, n_voices=2, label=DEFAULT_MODEL):
    n = max(2, int(n_voices))
    s = ("Ты — кастинг-режиссёр аудиокниги. Раздели текст на речь рассказчика и реплики персонажей. "
         f"Speaker 0 — РАССКАЗЧИК (авторский текст), Speaker 1 .. Speaker {n - 1} — персонажи "
         "(закрепи за каждым персонажем свой номер и держи его постоянным). "
         "КАЖДАЯ строка строго в формате 'Speaker K: реплика'. Текст сохраняй ДОСЛОВНО, "
         "только размечай говорящего и добавляй теги по смыслу. " + _TAG_RULES)
    out = _chat(s, text, max_new=2048, label=label)
    return "\n".join(filter_tags(ln) if ":" not in ln else f"{ln.split(':')[0]}:{filter_tags(ln.split(':', 1)[1])}" for ln in out.splitlines())

# --- Main loop ---
def main():
    log("готов к работе")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            action = cmd.get("action")
            text = cmd.get("text", "")
            label = cmd.get("label", DEFAULT_MODEL)
            n = cmd.get("n", 2)
            
            if action == "enrich":
                result = enrich(text, label)
            elif action == "podcast":
                result = write_podcast(text, n, label)
            elif action == "audiobook":
                result = cast_audiobook(text, n, label)
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