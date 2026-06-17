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

# Клон: референс кодируется ЦЕЛИКОМ. Пресеты бывают до 300с → prefill на тысячи аудио-токенов
# → тормоза/залипание/мусор; бьёт и по подкасту, и по аудиокниге (общий путь generate). Режем.
REF_MAX_SEC = 30
_REF_CACHE = {}  # path → (mtime, codes_TN, trimmed): один голос не перекодируем повторно


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


_forced_precision = None  # выбор квантизации из UI-дропдауна


def set_precision(p):
    """UI выбор квантизации: '4bit' / '8bit' / 'bf16'. Выгружает модель — перезагрузится в новой точности."""
    global _forced_precision
    _forced_precision = p if p in ("4bit", "8bit", "bf16") else None
    unload_tts()


def auto_precision(vram_gb, device):
    # bf16 по умолчанию (чище; на 24 ГБ влезает свободно). nf4/8bit — выбором в UI.
    # Приоритет: UI-выбор > env HIGGS_TTS_PRECISION > дефолт.
    pick = _forced_precision or os.environ.get("HIGGS_TTS_PRECISION", "").strip().lower()
    if pick in ("bf16", "8bit", "4bit"):
        return pick if device == "cuda" else "cpu"
    if device == "cpu":
        return "cpu"
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
    # Голову/эмбеддинг аудио-кодов держим ВНЕ кванта: они tied и предсказывают стоп-токен (EOC);
    # под nf4 голова деградирует → модель не ловит конец → генерит вразнос (залипание).
    skip = ["audio_head", "audio_embedding"]
    if precision == "4bit":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                   llm_int8_skip_modules=skip)
    elif precision == "8bit":
        quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
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
    if device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")  # TF32 — безопасно и бесплатно
        except Exception:
            pass
    # torch.compile бэкбона: ~2.4x (dynamic=True). КРИТИЧНО: компиляция должна происходить в ГЛАВНОМ
    # потоке (прогрев на старте, см. prewarm() в app.py) — компиляция dynamo/inductor в рабочем потоке
    # gradio роняет процесс (особенно на клон-пути). Нужны Python-заголовки (install.bat ставит dev.msi).
    if device == "cuda" and precision == "bf16" and os.environ.get("HIGGS_NO_COMPILE", "").lower() not in ("1", "true", "yes"):
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True  # сбой компиляции в потоке → откат на eager, НЕ краш
            _model.model = torch.compile(_model.model, dynamic=True)
            print("[higgs] torch.compile (dynamic) ВКЛ — ~2x (прогрев компиляции на старте)")
        except Exception as e:
            print(f"[higgs] torch.compile недоступен ({e}); без компиляции")
    return _model


_CANCEL = False


def request_cancel():
    global _CANCEL
    _CANCEL = True
    print("[gen] STOP ОТМЕНА — прерываю генерацию на текущем токене", flush=True)


def clear_cancel():
    global _CANCEL
    _CANCEL = False


def cancelled():
    return _CANCEL


def unload_tts():
    """Выгрузить TTS из памяти (для последовательной загрузки с LLM-режиссёром)."""
    global _model, _tok
    _model = None
    _tok = None
    import gc; gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # ждём завершения всех операций
    except Exception:
        pass


import contextlib

@contextlib.contextmanager
def tts_offloaded():
    """Временно выгружает TTS. Автоматически восстанавливает при выходе."""
    unload_tts()
    try:
        yield
    finally:
        pass  # TTS перезагрузится лениво при get_tts()


def _load_ref(path):
    import torch
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data).mean(dim=1), sr  # mono [L], sr


def _ref_codes(m, path):
    """Референс → коды [T,N]: обрезка до REF_MAX_SEC + кэш по (path, mtime).
    Возвращает (codes_cpu, trimmed). Без обрезки длинный пресет (до 300с) кодируется целиком →
    prefill на тысячи токенов → клон тормозит/залипает (и тащит за собой подкаст с аудиокнигой)."""
    import os
    try:
        mt = os.path.getmtime(path)
    except OSError:
        mt = 0.0
    hit = _REF_CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1], hit[2]
    wav, sr = _load_ref(path)
    cap = int(sr * REF_MAX_SEC)
    trimmed = wav.shape[-1] > cap
    if trimmed:
        wav = wav[:cap]
        print(f"[gen] референс {os.path.basename(path)} обрезан до {REF_MAX_SEC}с (был длинный)", flush=True)
    codes = m._encode_reference(wav, sr).cpu()
    _REF_CACHE[path] = (mt, codes, trimmed)
    return codes, trimmed


_FRAMES_PER_SEC = None  # калибруется по факту первой генерации — для оценки секунд аудио на лету


def _modeling(m):
    """Модуль remote-code модели (apply_delay_pattern / reverse_delay_pattern / _SamplerState / _sampler_step).
    torch.compile оборачивает m.model, не сам m — поэтому type(m).__module__ остаётся модулем Higgs."""
    import sys
    return sys.modules[type(m).__module__]


def _generate_stream(m, tok, text, *, reference_audio=None, reference_sample_rate=None,
                     reference_codes=None, reference_text=None, max_new_tokens=2048,
                     temperature=1.0, top_p=None, top_k=None, label="", attempt=(1, 1)):
    """Точная копия HiggsMultimodalQwen3.generate_speech, но цикл наш →
    (а) живой прогресс по фреймам в терминал (tqdm); (б) отмена НА УРОВНЕ инференса —
    флаг _CANCEL проверяется каждый токен и рвёт реальный AR-цикл немедленно.

    Возвращает (audio_tensor_cpu_f32 | пустой, was_cancelled: bool).
    """
    global _FRAMES_PER_SEC
    import time
    import torch

    # Резолв внутренностей модели; если remote-API сдвинулся — честный фоллбек на штатный метод.
    try:
        mod = _modeling(m)
        apply_delay_pattern = mod.apply_delay_pattern
        reverse_delay_pattern = mod.reverse_delay_pattern
        _SamplerState = mod._SamplerState
        _sampler_step = mod._sampler_step
        N = m.num_codebooks
        _ = (m._encode_reference, m._build_prompt_ids, m._prefill_embeds, m._decode_codes,
             m.audio_head, m.audio_embedding, m.model)
    except Exception as e:
        print(f"[gen] внутренний цикл недоступен ({e}); штатный generate_speech без прогресса", flush=True)
        kw = dict(reference_text=reference_text, max_new_tokens=max_new_tokens,
                  temperature=temperature, top_p=top_p, top_k=top_k)
        if reference_codes is not None:
            kw["reference_codes"] = reference_codes
        elif reference_audio is not None:
            kw["reference_audio"] = reference_audio
            kw["reference_sample_rate"] = reference_sample_rate
        return m.generate_speech(text, tok, **kw), False

    att = f" · попытка {attempt[0]}/{attempt[1]}" if attempt[1] > 1 else ""
    print(f"[gen] >> синтез: {label}{att}", flush=True)

    with torch.no_grad():
        delayed_ref = None
        if reference_codes is not None:
            delayed_ref = apply_delay_pattern(reference_codes.to(torch.long))
        elif reference_audio is not None:
            sr = reference_sample_rate or m.config.sample_rate
            codes_TN = m._encode_reference(reference_audio, sr)
            delayed_ref = apply_delay_pattern(codes_TN.cpu())

        prompt_ids = m._build_prompt_ids(
            tok, text,
            num_ref_tokens=0 if delayed_ref is None else delayed_ref.shape[0],
            reference_text=reference_text,
        )
        inputs_embeds = m._prefill_embeds(prompt_ids, delayed_ref)
        out = m.model(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        hidden_last = out.last_hidden_state[:, -1, :]
        position = inputs_embeds.shape[1]

        state = _SamplerState(num_codebooks=N)
        rows = []
        cancelled_mid = False
        t0 = time.time()
        try:
            from tqdm import tqdm
            bar = tqdm(total=int(max_new_tokens), unit="frame", desc="[gen] озвучка",
                       dynamic_ncols=True, leave=False, ascii=True)
        except Exception:
            bar = None

        for step in range(int(max_new_tokens)):
            if _CANCEL:  # ← ОТМЕНА НА УРОВНЕ ИНФЕРЕНСА: рвём реальный цикл генерации
                cancelled_mid = True
                break
            logits_NV = m.audio_head(hidden_last).to(torch.float32)[0]
            codes_N = _sampler_step(logits_NV, state, temperature=temperature, top_p=top_p, top_k=top_k)
            if state.generation_done:
                break
            rows.append(codes_N.cpu())

            if bar is not None:
                bar.update(1)
                if _FRAMES_PER_SEC and step % 16 == 0:
                    bar.set_postfix_str(f"~{len(rows) / _FRAMES_PER_SEC:.1f}с аудио")
            elif step % 64 == 0 and step:
                el = time.time() - t0
                print(f"[gen]   {step} фреймов · {step / max(el, 1e-3):.0f} фрейм/с", flush=True)

            step_embed = m.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
            cache_pos = torch.tensor([position], device=m.device)
            out = m.model(inputs_embeds=step_embed.to(inputs_embeds.dtype),
                          past_key_values=past, use_cache=True, cache_position=cache_pos)
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]
            position += 1

        if bar is not None:
            bar.close()
        el = time.time() - t0

        if cancelled_mid:
            print(f"[gen] STOP прервано на {len(rows)} фреймах ({el:.1f}с)", flush=True)
            return torch.zeros(0, dtype=torch.float32), True
        if len(rows) < N:
            print(f"[gen] пусто ({len(rows)} фреймов < {N})", flush=True)
            return torch.zeros(0, dtype=torch.float32), False

        delayed_LN = torch.stack(rows, dim=0)
        codes_TN = reverse_delay_pattern(delayed_LN)
        audio = m._decode_codes(codes_TN)

    sec = audio.shape[-1] / SR
    if sec > 0.05:
        _FRAMES_PER_SEC = len(rows) / sec  # калибровка оценки секунд для следующих генераций
    print(f"[gen] OK {len(rows)} фреймов -> {sec:.1f}с аудио за {el:.1f}с ({len(rows) / max(el, 1e-3):.0f} фрейм/с)",
          flush=True)
    return audio, False


def generate(text, ref_audio=None, ref_text=None, temperature=1.0, top_p=0.95,
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
        ref_codes, trimmed = _ref_codes(m, ref_audio)
        kw["reference_codes"] = ref_codes
        if trimmed:
            ref_text = None  # транскрипт пресета относится к ПОЛНОМУ аудио → с обрезанным не совпадёт
        if ref_text and ref_text.strip():
            kw["reference_text"] = ref_text.strip()
    label = f"{len(text)} симв" + (" · клон по референсу" if ref_audio else "")
    # Анти-разнос (как retry_badcase в VoxCPM2): если аудио неправдоподобно длинное для
    # текста — модель пошла вразнос, перегенерируем (для случайного сида попытки разные).
    fixed_seed = seed is not None and int(seed) >= 0
    limit_sec = 0.13 * max(len(text), 1) + 3.0
    attempts = 1 if fixed_seed else 3
    audio = None
    for a in range(attempts):
        audio, was_cancelled = _generate_stream(m, _tok, text, label=label, attempt=(a + 1, attempts), **kw)
        if was_cancelled:
            return SR, np.zeros(0, np.float32)
        sec = (audio.shape[-1] if hasattr(audio, "shape") else len(audio)) / SR
        if sec <= limit_sec or a == attempts - 1:
            break
        print(f"[gen] разнос {sec:.1f}s > {limit_sec:.1f}s — повтор {a + 2}/{attempts}", flush=True)
    return SR, audio.detach().cpu().numpy().astype(np.float32)


TARGET_LUFS = -16.0              # стандарт подкастов/TTS (EBU R128, Google Assistant)
_PEAK_CEIL = 10 ** (-1.0 / 20)   # −1 dBFS — защита микса от клиппинга
_MAX_GAIN = 10 ** (20.0 / 20)    # не разгонять тихий фрагмент сильнее +20 dB (мусор/тишина)


def _loudness_normalize(x, sr=SR):
    """Фрагмент → целевая громкость, чтобы спикеры в миксе звучали ровно (один не тише другого).
    LUFS-метр BS.1770 (pyloudnorm) если доступен, иначе RMS-фоллбек на чистом numpy."""
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    try:
        import pyloudnorm as pyln
        loud = pyln.Meter(sr).integrated_loudness(x)
        if np.isfinite(loud):
            gain = 10 ** ((TARGET_LUFS - loud) / 20)
            return (x * min(gain, _MAX_GAIN)).astype(np.float32)
    except Exception:
        pass
    rms = float(np.sqrt(np.mean(x ** 2)))   # фоллбек: RMS к ~−20 dBFS (ориентир для речи)
    if rms < 1e-6:
        return x
    return (x * min((10 ** (-20.0 / 20)) / rms, _MAX_GAIN)).astype(np.float32)


def _peak_limit(x, ceil=_PEAK_CEIL):
    import numpy as np
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    return (x * (ceil / peak)).astype(np.float32) if peak > ceil else x


def _concat(chunks, gap=0.3, normalize=True):
    """Склейка фрагментов с паузой. normalize=True выравнивает громкость спикеров
    (LUFS/RMS на фрагмент) и ставит пик-лимит −1 dBFS на итоговый микс."""
    import numpy as np
    chunks = [c for c in chunks if c is not None and len(c)]
    if not chunks:
        return np.zeros(0, np.float32)
    if normalize:
        chunks = [_loudness_normalize(c) for c in chunks]
    sil = np.zeros(int(SR * gap), np.float32)
    out = []
    for i, c in enumerate(chunks):
        if i:
            out.append(sil)
        out.append(c)
    mix = np.concatenate(out)
    return _peak_limit(mix) if normalize else mix


def synth_longform(paragraphs, ref_audio=None, ref_text=None, **kw):
    """Длинный текст по абзацам. Первый кусок задаёт голос, его аудио — референс для остальных."""
    import tempfile
    import soundfile as sf
    chunks = []
    chain_ref, chain_txt = ref_audio, ref_text
    paras = [p for p in paragraphs if p and p.strip()]
    for i, para in enumerate(paras):
        if _CANCEL:
            print(f"[gen] STOP остановлено на чанке {i + 1}/{len(paras)}", flush=True)
            break
        print(f"[gen] лонг-форм: чанк {i + 1}/{len(paras)}", flush=True)
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
        if _CANCEL:
            break
        if t.get("text", "").strip():
            _, a = generate(t["text"], ref_audio=t.get("ref_audio"), ref_text=t.get("ref_text"), **kw)
            chunks.append(a)
    return SR, _concat(chunks, gap)
