# -*- coding: utf-8 -*-
"""
Módulo de Manipulação de Vídeo e Subtítulos via FFmpeg
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import List, Dict

from dublagem.config import (
    AUDIO_SAMPLE_RATE, MSG
)


def extract_original_audio(video_path: Path, output_wav_path: Path) -> None:
    """Extrai a faixa de áudio original do vídeo em formato WAV mono (PCM 16-bit)."""
    logging.info(f"🔊 Extraindo faixa de áudio original do vídeo: {video_path.name}")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1",
            str(output_wav_path)
        ], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timeout ao extrair áudio original de {video_path.name} (>300s).") from e
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or str(e)).strip()
        logging.error(f"❌ Erro ao extrair áudio original: {err_msg}")
        raise RuntimeError(f"Erro ao extrair áudio: {err_msg}") from e


def replace_video_audio(video_path: Path, audio_path: Path, output_video_path: Path) -> None:
    """
    Combina o vídeo original com a nova faixa de áudio dublada.
    Usa '-c:v copy' para copiar sem re-encoding; faz fallback para libx264 se o codec de origem for incompatível com MP4.
    """
    logging.info(MSG["finalizando"])
    copy_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        str(output_video_path)
    ]
    try:
        subprocess.run(
            copy_cmd, check=True, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=600
        )
        return
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timeout ao mesclar vídeo e áudio final ({video_path.name}).") from e
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or str(e)).strip()
        logging.warning(
            f"⚠️ Remux rápido '-c:v copy' falhou para {video_path.name} ({err_msg.splitlines()[-1] if err_msg else 'erro FFmpeg'}). "
            "Tentando re-encoding compatível com MP4 (libx264)..."
        )

    fallback_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        str(output_video_path)
    ]
    try:
        subprocess.run(
            fallback_cmd, check=True, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=1800
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timeout no re-encoding fallback para {video_path.name}.") from e
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or str(e)).strip()
        logging.error(f"❌ Erro ao mesclar vídeo e áudio final (fallback): {err_msg}")
        raise RuntimeError(f"Erro ao fundir vídeo e áudio: {err_msg}") from e


def save_srt(segments: List[Dict], output_path: Path) -> None:
    """Exporta a lista de segmentos traduzidos em um arquivo de legenda padrão SRT."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments):
                start = _ms_to_srt_time(seg["start_ms"])
                end = _ms_to_srt_time(seg["end_ms"])
                f.write(f"{i + 1}\n{start} --> {end}\n{seg['text']}\n\n")
        logging.info(f"✅ Legendas geradas com sucesso: {output_path.name}")
    except Exception as e:
        logging.error(f"❌ Falha ao salvar arquivo de legenda SRT: {e}")


def parse_srt(srt_path: Path) -> List[Dict]:
    """
    Lê um arquivo .srt existente e faz o parse de volta para o formato de segmentos do pipeline.
    Suporta UTF-8 com BOM (Bloco de Notas Windows), quebras CRLF e separador ',' ou '.' em milissegundos.
    """
    segments = []
    try:
        content = srt_path.read_text(encoding='utf-8-sig')
        normalized = content.replace('\r\n', '\n').replace('\r', '\n').strip()
        if not normalized:
            return []

        blocks = re.split(r'\n\s*\n', normalized)

        for block in blocks:
            lines = [ln for ln in block.strip().split('\n') if ln is not None]
            if len(lines) < 3:
                continue

            time_match = re.match(
                r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
                lines[1].strip()
            )
            if not time_match:
                continue

            g = [int(x) for x in time_match.groups()]
            start_ms = g[0] * 3600000 + g[1] * 60000 + g[2] * 1000 + g[3]
            end_ms = g[4] * 3600000 + g[5] * 60000 + g[6] * 1000 + g[7]
            text = ' '.join(lines[2:]).strip()

            if end_ms > start_ms and text:
                segments.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text
                })

        logging.info(f"📄 SRT importado com sucesso: {len(segments)} falas carregadas de {srt_path.name}")
        return segments
    except Exception as e:
        logging.error(f"❌ Falha ao realizar parse do arquivo SRT: {e}")
        return []


def _ms_to_srt_time(ms: int) -> str:
    """Converte milissegundos inteiros no formato de tempo SRT (HH:MM:SS,mmm)."""
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1_000
    rest = ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{rest:03d}"
