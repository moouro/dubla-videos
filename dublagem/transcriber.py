# -*- coding: utf-8 -*-
"""
Módulo de Transcrição de Alta Fidelidade via faster-whisper (CTranslate2)
Inclui Silero VAD, Travas Anti-Alucinação, Timestamps por Palavra e Merge Inteligente
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm
from faster_whisper import WhisperModel

from dublagem.config import (
    MODELS_DIR, MSG, WHISPER_MODEL, WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE, WHISPER_CPU_THREADS,
    WHISPER_VAD_FILTER, WHISPER_VAD_MIN_SILENCE_MS,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT, WHISPER_REPETITION_PENALTY,
    WHISPER_NO_REPEAT_NGRAM_SIZE, WHISPER_HALLUCINATION_SILENCE_THRESHOLD,
    MERGE_MAX_GAP_MS, MERGE_MAX_DURATION_MS, MERGE_MAX_CHARS
)

# Cache em memória do modelo carregado (evita recarregar entre vídeos em lote)
_CACHED_MODEL: Optional[WhisperModel] = None
_CACHED_MODEL_NAME: Optional[str] = None

_END_PUNCTUATION_STRONG = {'!', '?', '。', '！', '？'}
_NON_BREAKING_ABBREVS = (
    "e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "ms.", "st.", "prof.",
    ".js.", ".ts.", ".py.", ".css.", ".html.", ".json.", ".yaml.", ".yml.",
    ".sh.", ".md.", ".env.", ".sql.", ".tsx.", ".jsx.", ".io.", ".dev.", ".net.", ".com."
)


def get_whisper_model(model_name: str = WHISPER_MODEL) -> WhisperModel:
    """
    Carrega e armazena em cache a instância do modelo faster-whisper.
    Salva os modelos baixados na pasta models/faster-whisper/.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME

    clean_name = model_name.lower().strip()
    if "large-v3" in clean_name and "turbo" not in clean_name:
        clean_name = "large-v3"
    elif "turbo" in clean_name:
        clean_name = "large-v3-turbo"
    elif "base" in clean_name:
        clean_name = "base"
    elif "medium" in clean_name:
        clean_name = "medium"
    elif "small" in clean_name:
        clean_name = "small"

    if _CACHED_MODEL is not None and _CACHED_MODEL_NAME == clean_name:
        return _CACHED_MODEL

    fw_models_dir = MODELS_DIR / "faster-whisper"
    fw_models_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        f"🧠 Carregando faster-whisper ({clean_name.upper()}) "
        f"[Device: {WHISPER_DEVICE} | Compute: {WHISPER_COMPUTE_TYPE} | Threads: {WHISPER_CPU_THREADS}]..."
    )

    # Tenta carregar localmente antes para funcionar 100% offline quando já baixado
    try:
        model = WhisperModel(
            model_size_or_path=clean_name,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
            download_root=str(fw_models_dir),
            local_files_only=True
        )
    except Exception:
        model = WhisperModel(
            model_size_or_path=clean_name,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
            download_root=str(fw_models_dir)
        )

    _CACHED_MODEL = model
    _CACHED_MODEL_NAME = clean_name
    return model


def load_audio_for_whisper(media_path: Path) -> np.ndarray:
    """
    Decodifica qualquer arquivo de áudio ou vídeo para float32 16kHz mono via FFmpeg.
    Contorna problemas de metadados do PyAV (>= 19.x) ao abrir arquivos diretamente.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(media_path),
        "-vn", "-ar", "16000", "-ac", "1", "-f", "s16le", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=300)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timeout ao decodificar áudio via FFmpeg para {media_path.name} (>300s).") from e
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else str(e)
        raise RuntimeError(f"Erro ao decodificar áudio via FFmpeg ({media_path.name}): {err_msg}") from e

    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_audio(
    media_path: Path,
    model_name: str = WHISPER_MODEL,
    language: str = "auto"
) -> Tuple[List[Dict], str]:
    """
    Transcreve áudio/vídeo utilizando faster-whisper com Silero VAD e travas anti-alucinação.
    Retorna uma tupla (segmentos_consolidados, idioma_detectado).
    """
    model = get_whisper_model(model_name)

    lang_param = None if (not language or language == "auto") else language
    logging.info(f"{MSG['transcrevendo']} (Arquivo: {media_path.name} | Idioma: {language})")

    audio_data = load_audio_for_whisper(media_path)
    total_duration = len(audio_data) / 16000.0

    vad_params = dict(
        min_silence_duration_ms=WHISPER_VAD_MIN_SILENCE_MS,
        speech_pad_ms=200
    )

    segments_gen, info = model.transcribe(
        audio_data,
        language=lang_param,
        vad_filter=WHISPER_VAD_FILTER,
        vad_parameters=vad_params,
        word_timestamps=True,
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        repetition_penalty=WHISPER_REPETITION_PENALTY,
        no_repeat_ngram_size=WHISPER_NO_REPEAT_NGRAM_SIZE,
        hallucination_silence_threshold=WHISPER_HALLUCINATION_SILENCE_THRESHOLD,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6
    )

    raw_lang = getattr(info, "language", None)
    detected_lang = raw_lang or (language if (language and language != "auto") else "en")
    prob = getattr(info, "language_probability", 1.0) or 1.0

    logging.info(f"🔍 Idioma detectado: {detected_lang.upper()} (confiança: {prob:.1%}) | Duração: {total_duration:.1f}s")

    raw_segments = []

    with tqdm(total=int(total_duration), desc="Transcrevendo (VAD)", unit="s", dynamic_ncols=True) as pbar:
        last_progress_s = 0.0
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue

            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)

            if seg.words:
                first_word = seg.words[0]
                last_word = seg.words[-1]
                start_ms = int(first_word.start * 1000)
                end_ms = int(last_word.end * 1000)

            if end_ms > start_ms:
                raw_segments.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text
                })

            progress_delta = max(0, int(seg.end - last_progress_s))
            if progress_delta > 0:
                pbar.update(progress_delta)
                last_progress_s = seg.end

        if total_duration > last_progress_s:
            pbar.update(int(total_duration - last_progress_s))

    if not raw_segments:
        raise RuntimeError("❌ Nenhum segmento de fala foi detectado no vídeo/áudio.")

    cleaned = _deduplicate_segments(raw_segments)
    no_loops = _remove_repetition_loops(cleaned)
    merged = _merge_segments(no_loops)

    logging.info(f"✅ Transcrição concluída: {len(raw_segments)} falas consolidadas em {len(merged)} segmentos lógicos.")
    return merged, detected_lang


def _deduplicate_segments(segments: List[Dict]) -> List[Dict]:
    """
    Resolve sobreposições entre segmentos consecutivos:
    - Sobreposição majoritária (>50%) ou segmento contido: mantém o de maior duração.
    - Pequena sobreposição (ex.: speech_pad_ms / word_timestamps): ajusta start_ms sem descartar fala.
    """
    if not segments:
        return segments

    ordered = sorted((s.copy() for s in segments), key=lambda s: s["start_ms"])
    cleaned = [ordered[0]]

    for seg in ordered[1:]:
        prev = cleaned[-1]
        if seg["start_ms"] < prev["end_ms"]:
            overlap = min(prev["end_ms"], seg["end_ms"]) - seg["start_ms"]
            prev_dur = max(1, prev["end_ms"] - prev["start_ms"])
            seg_dur = max(1, seg["end_ms"] - seg["start_ms"])

            if seg["end_ms"] <= prev["end_ms"] or (overlap / seg_dur) > 0.5 or (overlap / prev_dur) > 0.8:
                if seg_dur > prev_dur and (overlap / prev_dur) > 0.5:
                    cleaned[-1] = seg
            else:
                seg["start_ms"] = prev["end_ms"]
                if seg["end_ms"] > seg["start_ms"]:
                    cleaned.append(seg)
        else:
            cleaned.append(seg)

    return cleaned


def _remove_repetition_loops(segments: List[Dict]) -> List[Dict]:
    """
    Proteção adicional contra loops de alucinação do Whisper:
    Se 3 ou mais segmentos consecutivos possuírem texto normalizado idêntico,
    mantém apenas a primeira ocorrência e descarta as repetições seguintes.
    """
    if len(segments) < 3:
        return segments

    filtered: List[Dict] = []
    i = 0
    removed_total = 0

    while i < len(segments):
        norm_curr = re.sub(r"\s+", " ", segments[i]["text"].strip().lower())
        j = i + 1
        while j < len(segments):
            norm_next = re.sub(r"\s+", " ", segments[j]["text"].strip().lower())
            if norm_curr and norm_next == norm_curr:
                j += 1
            else:
                break

        run_len = j - i
        filtered.append(segments[i])
        if run_len >= 3:
            removed = run_len - 1
            removed_total += removed
            logging.warning(
                f"⚠️ Loop de repetição detectado pelo Whisper ({run_len}x): "
                f"'{segments[i]['text'][:50]}...'. Mantendo 1ª ocorrência e descartando {removed}."
            )
        else:
            for k in range(i + 1, j):
                filtered.append(segments[k])
        i = j

    if removed_total > 0:
        logging.info(f"🧹 Filtro anti-loop removeu {removed_total} segmentos duplicados em sequência.")

    return filtered


def _ends_sentence(text: str) -> bool:
    """
    Verifica se o texto encerra realmente uma frase, ignorando abreviações comuns
    e notações/extensões técnicas que terminam com ponto.
    """
    stripped = text.strip()
    if not stripped:
        return True

    last_char = stripped[-1]
    if last_char in _END_PUNCTUATION_STRONG:
        return True
    if last_char != '.':
        return False

    lower = stripped.lower()
    if lower.endswith("..."):
        return False
    if any(lower.endswith(abbrev) for abbrev in _NON_BREAKING_ABBREVS):
        return False
    if re.search(r"\bv\d+\.$", lower):
        return False

    return True


def _merge_segments(segments: List[Dict]) -> List[Dict]:
    """
    Junta segmentos consecutivos que pertencem à mesma frase lógica,
    preservando start_ms inicial e end_ms final e respeitando limites de pausa, duração e caracteres.
    """
    if not segments:
        return segments

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start_ms"] - prev["end_ms"]
        prev_text = prev["text"].strip()
        seg_text = seg["text"].strip()
        merged_duration = seg["end_ms"] - prev["start_ms"]
        merged_chars = len(prev_text) + 1 + len(seg_text)

        should_merge = (
            gap <= MERGE_MAX_GAP_MS
            and merged_duration <= MERGE_MAX_DURATION_MS
            and merged_chars <= MERGE_MAX_CHARS
            and prev_text
            and not _ends_sentence(prev_text)
        )

        if should_merge:
            prev["end_ms"] = seg["end_ms"]
            prev["text"] = f"{prev_text} {seg_text}" if seg_text else prev_text
        else:
            merged.append(seg.copy())

    return merged
