"""Движок Higgs Audio v3 TTS.

Загружает transformers-порт multimodalart/higgs-audio-v3-tts-4b-transformers,
авто-точность (bf16/8/4-бит через bitsandbytes по VRAM), generate_speech,
длинная форма с переносом голоса и мульти-спикер склейка.

Тяжёлые импорты ленивые (внутри функций) — mock-режим и UI поднимаются без torch.
"""
import os

TTS_REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"
SR = 24000
_MOCK = bool(os.environ.get("HIGGS_UI_MOCK"))
_model = None
_tok = None


def detect_device():
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return "cuda", p.name, p.total_memory / 1e9
    return "cpu", "CPU", 0.0


def device_info():
    if _MOCK:
        return "MOCK UI (без модели)"
    try:
        dev, name, vram = detect_device()
    except Exception:
        return "CPU"
    return f"{name} | VRAM {vram:.1f} ГБ" if dev == "cuda" else "CPU (медленно)"


def auto_precision(vram_gb, device):
    if device == "cpu":
        return "cpu"
    if vram_gb < 6:
        return "4bit"
    if vram_gb < 12:
        return "8bit"
    return "bf16"


def get_tts(precision=None):
    global _model, _tok
    if _MOCK:
        return "MOCK"
    if _model is not None:
        return _model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device, name, vram = detect_device()
    precision = precision or auto_precision(vram, device)
    print(f"[higgs] загрузка TTS ({precision}) на {name}...")
    quant = None
    if precision == "4bit":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif precision == "8bit":
        quant = BitsAndBytesConfig(load_in_8bit=True)
    _tok = AutoTokenizer.from_pretrained(TTS_REPO)
    kw = dict(trust_remote_code=True, dtype=torch.bfloat16)
    if quant is not None:
        kw["quantization_config"] = quant
        kw["device_map"] = "auto"
    try:  # flash-attention 2, если установлена (ускоритель из install.bat)
        _model = AutoModelForCausalLM.from_pretrained(TTS_REPO, attn_implementation="flash_attention_2", **kw)
    except Exception as e:
        print(f"[higgs] flash_attention_2 недоступна ({e}); стандартный attention")
        _model = AutoModelForCausalLM.from_pretrained(TTS_REPO, **kw)
    if quant is None and device == "cuda":
        _model = _model.to("cuda")
    _model.eval()
    try:
        _model.get_audio_codec()  # прогрев fp32-кодека
    except Exception:
        pass
    return _model


def unload_tts():
    """Выгрузить TTS из памяти (для последовательной загрузки с LLM-режиссёром)."""
    global _model, _tok
    _model = None
    _tok = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_ref(path):
    import torch
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data).mean(dim=1), sr  # mono [L], sr


def generate(text, ref_audio=None, ref_text=None, temperature=0.7, top_p=0.95,
             top_k=50, max_new_tokens=2048, seed=-1):
    """Озвучить один фрагмент. Возвращает (sr, np.float32[L]). ref_audio — путь к файлу."""
    import numpy as np
    text = (text or "").strip()
    if not text:
        return SR, np.zeros(0, np.float32)
    if _MOCK:
        n = int(SR * 1.4)
        return SR, (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.float32)
    import torch
    m = get_tts()
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
    kw = dict(max_new_tokens=int(max_new_tokens), temperature=max(0.05, float(temperature)),
              top_p=(float(top_p) if float(top_p) < 1.0 else None),
              top_k=(int(top_k) if int(top_k) > 0 else None))
    if ref_audio:
        wav, sr = _load_ref(ref_audio)
        kw["reference_audio"] = wav
        kw["reference_sample_rate"] = sr
        if ref_text and ref_text.strip():
            kw["reference_text"] = ref_text.strip()
    audio = m.generate_speech(text, _tok, **kw)
    return SR, audio.detach().cpu().numpy().astype(np.float32)


def _concat(chunks, gap=0.3):
    import numpy as np
    chunks = [c for c in chunks if c is not None and len(c)]
    if not chunks:
        return np.zeros(0, np.float32)
    sil = np.zeros(int(SR * gap), np.float32)
    out = []
    for i, c in enumerate(chunks):
        if i:
            out.append(sil)
        out.append(c)
    return np.concatenate(out)


def synth_longform(paragraphs, ref_audio=None, ref_text=None, **kw):
    """Длинный текст по абзацам. Первый кусок задаёт голос, его аудио — референс для остальных."""
    import tempfile
    import soundfile as sf
    chunks = []
    chain_ref, chain_txt = ref_audio, ref_text
    paras = [p for p in paragraphs if p and p.strip()]
    for i, para in enumerate(paras):
        _, a = generate(para, ref_audio=chain_ref, ref_text=chain_txt, **kw)
        chunks.append(a)
        if i == 0 and not _MOCK and ref_audio is None and len(a):
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.close()  # Windows: закрыть хэндл до повторного open в sf.write
            sf.write(f.name, a, SR)
            chain_ref, chain_txt = f.name, para
    return SR, _concat(chunks)


def synth_turns(turns, gap=0.4, **kw):
    """turns: [{'text','ref_audio','ref_text'}] — у каждого спикера свой голос через его референс."""
    chunks = []
    for t in turns:
        if t.get("text", "").strip():
            _, a = generate(t["text"], ref_audio=t.get("ref_audio"), ref_text=t.get("ref_text"), **kw)
            chunks.append(a)
    return SR, _concat(chunks, gap)
