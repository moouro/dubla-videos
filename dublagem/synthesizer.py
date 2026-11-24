# -*- coding: utf-8 -*-
"""
Módulo de Síntese de Voz Neural Offline via Coqui XTTS-v2
Qualidade máxima em PT-BR com suporte a speakers nativos e clonagem de voz zero-shot.
"""

import os
from pathlib import Path

from dublagem.config import (
    MODELS_DIR, MSG, TTS_MODEL, TTS_CPU_THREADS, VOZES
)

# Variáveis exigidas pelo Coqui XTTS-v2 no PyTorch moderno antes de importar torch/TTS
os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TTS_HOME", str(MODELS_DIR / "coqui-tts"))

import asyncio
import logging
import re
from typing import List, Dict, Optional

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

_CACHED_TTS = None
_CACHED_TTS_MODEL: Optional[str] = None


def _has_speakable_text(text: str) -> bool:
    """Verifica se o texto contém ao menos um caractere alfanumérico pronunciável."""
    return bool(text and text.strip() and re.search(r"\w", text))


def _prepare_xtts_text(text: str, max_chars: int = 185) -> str:
    """
    Garante que nenhuma sentença ultrapasse o limite de ~203 caracteres do XTTS-v2 em 'pt',
    quebrando trechos longos em pontuação intermediária (vírgula/ponto-e-vírgula) ou espaço.
    """
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= max_chars:
        return cleaned

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    result_parts: List[str] = []

    for sent in sentences:
        remaining = sent.strip()
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            cut_pos = -1
            cut_len = 1

            for sep in (", ", "; ", ": "):
                idx = window.rfind(sep)
                if idx >= 25:
                    cut_pos = idx
                    cut_len = len(sep)
                    break

            if cut_pos == -1:
                idx = window.rfind(" ")
                cut_pos = idx if idx > 0 else max_chars
                cut_len = 1 if idx > 0 else 0

            head = remaining[:cut_pos].rstrip(" ,;:")
            if head and head[-1] not in ".!?":
                head += "."
            if head:
                result_parts.append(head)
            remaining = remaining[cut_pos + cut_len:].lstrip()

        if remaining:
            result_parts.append(remaining)

    return " ".join(result_parts)


def _resolve_xtts_language(lang: str) -> str:
    """Normaliza o código de idioma para o padrão aceito pelo XTTS-v2 ('pt', 'en', 'es', etc.)."""
    base = (lang or "pt").strip().lower().replace("_", "-").split("-")[0]
    return "zh-cn" if base == "zh" else base


def _resolve_speaker(voice: str) -> str:
    """Resolve alias amigável (ex: 'marcos') para o nome oficial de speaker do XTTS-v2."""
    if not voice:
        return VOZES.get("marcos", "Marcos Rudaski")
    key = voice.strip().lower()
    if key in VOZES:
        return VOZES[key]
    return voice.strip()


def _get_xtts_model():
    """Carrega e armazena em cache a instância do Coqui XTTS-v2 rodando na CPU."""
    global _CACHED_TTS, _CACHED_TTS_MODEL

    if _CACHED_TTS is not None and _CACHED_TTS_MODEL == TTS_MODEL:
        return _CACHED_TTS

    coqui_dir = MODELS_DIR / "coqui-tts"
    coqui_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TTS_HOME"] = str(coqui_dir)

    torch.set_num_threads(TTS_CPU_THREADS)

    logging.info(f"🧠 Carregando Coqui XTTS-v2 offline ({TTS_MODEL}) [CPU | Threads: {TTS_CPU_THREADS}]...")
    from TTS.api import TTS

    tts = TTS(TTS_MODEL).to("cpu")
    _CACHED_TTS = tts
    _CACHED_TTS_MODEL = TTS_MODEL
    return tts


def extract_voice_sample(
    original_wav: Path,
    output_sample_wav: Path,
    segments: List[Dict],
    target_duration_s: float = 10.0
) -> Optional[Path]:
    """
    Extrai de 8s a 12s de fala limpa do áudio original para clonagem zero-shot (--voice clone).
    Seleciona trechos com maior duração contínua dentro do intervalo [2s, 12s] e energia válida.
    """
    if not original_wav or not original_wav.exists():
        return None

    try:
        audio_data, sr = sf.read(str(original_wav), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        total_samples = len(audio_data)
        target_samples = int(target_duration_s * sr)
        min_samples = int(6.0 * sr)

        candidates = []
        for seg in segments:
            dur_s = (seg["end_ms"] - seg["start_ms"]) / 1000.0
            if dur_s < 1.8:
                continue
            s_idx = max(0, int((seg["start_ms"] / 1000.0) * sr))
            e_idx = min(total_samples, int((seg["end_ms"] / 1000.0) * sr))
            chunk = audio_data[s_idx:e_idx]
            if len(chunk) < int(1.5 * sr):
                continue
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > 0.01:
                candidates.append((dur_s, rms, chunk))

        if not candidates:
            return None

        # Prioriza trechos mais longos com boa energia vocal
        candidates.sort(key=lambda item: (min(item[0], 12.0), item[1]), reverse=True)

        collected = []
        collected_len = 0
        pause = np.zeros(int(0.15 * sr), dtype=np.float32)

        for _, _, chunk in candidates:
            needed = target_samples - collected_len
            if needed <= 0:
                break
            piece = chunk[:needed]
            if collected:
                collected.append(pause)
                collected_len += len(pause)
            collected.append(piece)
            collected_len += len(piece)

        if collected_len < min_samples:
            return None

        sample_audio = np.concatenate(collected)[:int(12.0 * sr)]
        output_sample_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_sample_wav), sample_audio, sr, subtype="PCM_16")
        logging.info(f"🧬 Amostra vocal extraída para clonagem ({len(sample_audio) / sr:.1f}s): {output_sample_wav.name}")
        return output_sample_wav
    except Exception as e:
        logging.warning(f"⚠️ Falha ao extrair amostra de voz para clonagem: {e}")
        return None


def _synthesize_one_xtts(
    tts,
    text: str,
    output_path: Path,
    language: str,
    speaker_name: Optional[str] = None,
    speaker_wav: Optional[Path] = None
) -> bool:
    """Executa a síntese síncrona de um único segmento com XTTS-v2 e valida o WAV resultante."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_text = _prepare_xtts_text(text)

        kwargs = {
            "text": prepared_text,
            "file_path": str(output_path),
            "language": language,
            "split_sentences": True,
        }
        if speaker_wav is not None and speaker_wav.exists():
            kwargs["speaker_wav"] = str(speaker_wav)
        else:
            kwargs["speaker"] = speaker_name

        tts.tts_to_file(**kwargs)

        if not output_path.exists() or output_path.stat().st_size <= 1000:
            size = output_path.stat().st_size if output_path.exists() else 0
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Arquivo WAV inválido ou pequeno demais ({size} bytes).")

        audio_check, sr_check = sf.read(str(output_path), dtype="float32")
        if sr_check <= 0 or len(audio_check) == 0:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("Arquivo WAV gerado sem amostras válidas.")

        return True
    except Exception as e:
        output_path.unlink(missing_ok=True)
        logging.error(f"❌ Falha no XTTS-v2 para '{text[:40]}...': {e}")
        return False


async def synthesize_all_segments(
    segments: List[Dict],
    temp_dir: Path,
    voice: str,
    rate: str = "+0%",
    dest_lang: str = "pt",
    speaker_wav: Optional[Path] = None,
    original_audio_path: Optional[Path] = None
) -> List[Dict]:
    """
    Sintetiza todos os segmentos sequencialmente na CPU usando Coqui XTTS-v2 offline
    com os threads do PyTorch dedicados por requisição para máxima estabilidade e qualidade.
    """
    resolved_speaker = _resolve_speaker(voice)
    xtts_lang = _resolve_xtts_language(dest_lang)

    active_speaker_wav = speaker_wav if (speaker_wav and speaker_wav.exists()) else None
    if resolved_speaker == "auto_clone" and active_speaker_wav is None and original_audio_path:
        auto_sample_path = temp_dir.parent / "voice_clone_ref.wav"
        if auto_sample_path.exists() and auto_sample_path.stat().st_size > 10000:
            active_speaker_wav = auto_sample_path
        else:
            active_speaker_wav = extract_voice_sample(original_audio_path, auto_sample_path, segments)

        if active_speaker_wav is None:
            logging.warning("⚠️ Amostra limpa insuficiente para '--voice clone'. Usando speaker padrão Marcos Rudaski.")
            resolved_speaker = VOZES["marcos"]

    voice_label = f"Clone ({active_speaker_wav.name})" if active_speaker_wav else resolved_speaker
    logging.info(f"{MSG['sintetizando']} (Voz: {voice_label} | Idioma: {xtts_lang})")

    tts = await asyncio.to_thread(_get_xtts_model)

    results: List[Dict] = []
    skipped_count = 0

    with tqdm(total=len(segments), desc="Sintetizando (XTTS-v2)", unit="seg", dynamic_ncols=True) as pbar:
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if not _has_speakable_text(text):
                results.append({
                    **seg,
                    "audio_path": None,
                    "tts_success": False,
                    "tts_skipped": True,
                })
                skipped_count += 1
                pbar.update(1)
                continue

            seg_path = Path(seg["target_path"]) if seg.get("target_path") else (temp_dir / f"seg_{i:04d}.wav")

            success = await asyncio.to_thread(
                _synthesize_one_xtts,
                tts,
                text,
                seg_path,
                xtts_lang,
                resolved_speaker,
                active_speaker_wav
            )

            results.append({
                **seg,
                "audio_path": str(seg_path) if success else None,
                "tts_success": success,
                "tts_skipped": False,
            })
            pbar.update(1)

    ok = sum(1 for r in results if r.get("tts_success"))
    skip_info = f" ({skipped_count} sem texto pronunciável ignorados)" if skipped_count else ""
    logging.info(f"✅ Síntese XTTS-v2 Concluída: {ok}/{len(results)} segmentos sintetizados{skip_info}.")
    return results
