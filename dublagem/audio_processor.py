# -*- coding: utf-8 -*-
"""
Módulo de Processamento Digital de Sinais (DSP) de Áudio
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

from dublagem.config import (
    AUDIO_SAMPLE_RATE, MIN_SPEED_FACTOR, MAX_SPEED_FACTOR, MSG,
    DUCKING_VOLUME, ORIGINAL_VOLUME, DUCKING_FADE_MS,
    OVERFLOW_GAP_RATIO, OVERFLOW_MAX_MS, INSTRUMENTAL_VOLUME
)


def separate_audio(audio_path: Path, temp_dir: Path, model_filename: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Separa o áudio original em faixas de Vocais e Instrumental usando o audio-separator (ONNX).
    Retorna uma tupla (vocal_wav_path, instrumental_wav_path).
    """
    try:
        logging.info(f"🎵 Isolando vocais/trilha com o modelo {model_filename}...")
        from audio_separator.separator import Separator

        separator = Separator(output_dir=str(temp_dir), log_level=logging.WARNING)
        separator.load_model(model_filename=model_filename)

        output_files = separator.separate(str(audio_path))

        vocal_path = None
        instrumental_path = None

        for file in output_files:
            file_path = temp_dir / file
            if "(Vocals)" in file or "Vocals" in file:
                vocal_path = file_path
            elif "(Instrumental)" in file or "Instrumental" in file:
                instrumental_path = file_path

        if vocal_path and vocal_path.exists():
            logging.info(f"🎤 Faixa de vocais isolada: {vocal_path.name}")
        if instrumental_path and instrumental_path.exists():
            logging.info(f"🎸 Trilha de efeitos/ruídos isolada: {instrumental_path.name}")

        return vocal_path, instrumental_path
    except Exception as e:
        logging.error(f"❌ Falha ao separar áudio original: {e}", exc_info=True)
        return None, None


def get_duration_ms(filepath: Path) -> float:
    """Obtém a duração exata de um arquivo de áudio/vídeo em milissegundos usando o ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=True
        )
        return float(result.stdout.strip()) * 1000
    except Exception as e:
        logging.error(f"❌ Erro ao ler duração via ffprobe para {filepath.name}: {e}")
        return 0.0


async def get_duration_ms_async(filepath: Path) -> float:
    """Versão assíncrona não-bloqueante de get_duration_ms via ffprobe com timeout."""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(filepath),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            return 0.0
        out_str = stdout.decode("utf-8", errors="replace").strip()
        return float(out_str) * 1000 if out_str else 0.0
    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        logging.warning(f"⚠️ Timeout no ffprobe ao medir duração de {filepath.name}")
        return 0.0
    except Exception as e:
        logging.warning(f"⚠️ Falha no ffprobe assíncrono para {filepath.name}: {e}")
        return 0.0


def _build_speed_filter(factor: float) -> str:
    """Constrói a string de filtros FFmpeg 'atempo' permitindo fatores fora do range padrão 0.5-2.0."""
    if abs(factor - 1.0) < 0.01:
        return "atempo=1.0"

    filters = []
    remaining = factor

    if remaining < 1.0:
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")
    else:
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        filters.append(f"atempo={remaining:.4f}")

    return ",".join(filters)


async def adjust_segment_speed_async(
    input_path: Path,
    output_path: Path,
    target_duration_ms: float,
    semaphore: Optional[asyncio.Semaphore] = None
) -> bool:
    """
    Ajusta a velocidade de um segmento de voz assincronamente usando FFmpeg.
    Preserva o pitch original e executa sob semáforo e timeout explícito.
    """
    sem = semaphore or asyncio.Semaphore(6)
    async with sem:
        proc = None
        try:
            current_ms = await get_duration_ms_async(input_path)
            if current_ms <= 0 or target_duration_ms <= 0:
                return False

            speed_factor = float(np.clip(current_ms / target_duration_ms, MIN_SPEED_FACTOR, MAX_SPEED_FACTOR))
            speed_filter = _build_speed_filter(speed_factor)

            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", str(input_path),
                "-filter:a", f"{speed_filter},aresample=async=1",
                "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1",
                str(output_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )

            await asyncio.wait_for(proc.wait(), timeout=60)
            return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 100
        except asyncio.TimeoutError:
            if proc and proc.returncode is None:
                proc.kill()
                await proc.wait()
            logging.warning(f"⚠️ Timeout no FFmpeg ao ajustar velocidade de {input_path.name}")
            return False
        except Exception as e:
            logging.warning(f"⚠️ Ajuste de velocidade assíncrona falhou para {input_path.name}: {e}")
            return False


def _resample_if_needed(audio_data: np.ndarray, src_sr: int) -> np.ndarray:
    """Converte áudio para mono (se estéreo) e reamostra para AUDIO_SAMPLE_RATE se necessário."""
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    if src_sr != AUDIO_SAMPLE_RATE and len(audio_data) > 1:
        ratio = AUDIO_SAMPLE_RATE / src_sr
        new_len = int(len(audio_data) * ratio)
        indices = np.linspace(0, len(audio_data) - 1, new_len)
        audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)
    return audio_data.astype(np.float32, copy=False)


def _build_ducking_envelope(
    total_samples: int,
    segments: List[Dict],
    ducking_vol: float,
    original_vol: float,
    fade_ms: float
) -> np.ndarray:
    """
    Cria um envelope de volume dinâmico suave para o áudio original de fundo.
    Volume = ducking_vol nas falas, original_vol nos gaps silenciosos. Fades aplicados nas transições.
    """
    envelope = np.full(total_samples, original_vol, dtype=np.float32)
    fade_samples = int((fade_ms / 1000.0) * AUDIO_SAMPLE_RATE)

    for seg in segments:
        if not seg.get("tts_success"):
            continue

        start = max(0, int((seg["start_ms"] / 1000.0) * AUDIO_SAMPLE_RATE))
        end = min(total_samples, int((seg["end_ms"] / 1000.0) * AUDIO_SAMPLE_RATE))

        if start >= end:
            continue

        envelope[start:end] = ducking_vol

        fade_out_start = max(0, start - fade_samples)
        fade_out_len = start - fade_out_start
        if fade_out_len > 0:
            fade_curve = np.linspace(original_vol, ducking_vol, fade_out_len, dtype=np.float32)
            envelope[fade_out_start:start] = np.minimum(
                envelope[fade_out_start:start], fade_curve
            )

        fade_in_end = min(total_samples, end + fade_samples)
        fade_in_len = fade_in_end - end
        if fade_in_len > 0:
            fade_curve = np.linspace(ducking_vol, original_vol, fade_in_len, dtype=np.float32)
            envelope[end:fade_in_end] = np.minimum(
                envelope[end:fade_in_end], fade_curve
            )

    return envelope


def _extract_room_tone(audio_data: np.ndarray, sample_rate: int, segment_duration_s: float = 1.0) -> np.ndarray:
    """
    Extrai o ruído ambiente (room tone) contínuo do áudio original do vídeo.
    Analisa blocos de 100ms e monta um loop suavizado.
    """
    target_samples = int(segment_duration_s * sample_rate)
    block_size = int(0.1 * sample_rate)
    num_blocks = len(audio_data) // block_size

    if num_blocks < 3:
        return np.random.normal(0, 0.0005, target_samples).astype(np.float32)

    blocks = audio_data[:num_blocks * block_size].reshape(num_blocks, block_size)
    rms_per_block = np.sqrt(np.mean(blocks ** 2, axis=1))
    top_count = max(3, num_blocks // 5)
    quiet_indices = np.argsort(rms_per_block)[:top_count]

    quiet_blocks = [blocks[b] for b in quiet_indices if rms_per_block[b] > 1e-6]

    if quiet_blocks:
        room_tone = np.concatenate(quiet_blocks)
        if len(room_tone) >= target_samples:
            room_tone = room_tone[:target_samples].copy()
        else:
            repeats = (target_samples // len(room_tone)) + 1
            room_tone = np.tile(room_tone, repeats)[:target_samples]

        fade = int(0.01 * sample_rate)
        if len(room_tone) > fade * 2:
            room_tone[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            room_tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return room_tone.astype(np.float32)

    return np.random.normal(0, 0.0005, target_samples).astype(np.float32)


def _fill_silent_gaps(
    audio: np.ndarray,
    room_tone: np.ndarray,
    sample_rate: int,
    silence_threshold: float = 0.001,
    min_gap_ms: float = 80
) -> np.ndarray:
    """
    Preenche regiões com silêncio digital absoluto no áudio mixado usando NumPy vetorizado
    com loops do room tone e fades suaves nas bordas.
    """
    filled = audio.copy()
    room_len = len(room_tone)
    total_len = len(audio)

    if room_len == 0 or total_len == 0:
        return filled

    block_size = max(1, int(0.01 * sample_rate))  # blocos de 10ms
    min_gap_samples = int((min_gap_ms / 1000.0) * sample_rate)

    n_full = total_len // block_size
    rms_list = []
    if n_full > 0:
        blocks = audio[:n_full * block_size].reshape(n_full, block_size)
        rms_list.append(np.sqrt(np.mean(blocks ** 2, axis=1)))
    rem = total_len % block_size
    if rem > 0:
        tail_rms = np.sqrt(np.mean(audio[n_full * block_size:] ** 2))
        rms_list.append(np.array([tail_rms], dtype=np.float32))

    rms_arr = np.concatenate(rms_list) if rms_list else np.empty(0, dtype=np.float32)
    is_silent = (rms_arr < silence_threshold).astype(np.int8)

    padded = np.concatenate(([0], is_silent, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0] * block_size
    ends = np.minimum(np.where(diffs == -1)[0] * block_size, total_len)

    fade_samples = int(0.005 * sample_rate)

    for start, end in zip(starts, ends):
        gap_len = int(end - start)
        if gap_len < min_gap_samples:
            continue

        repeats = (gap_len // room_len) + 1
        patch = np.tile(room_tone, repeats)[:gap_len].copy()

        fade = min(fade_samples, gap_len // 4)
        if fade > 0:
            patch[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            patch[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        filled[start:end] += patch

    return filled


async def assemble_dubbed_audio(
    segments: List[Dict],
    total_duration_ms: float,
    output_path: Path,
    temp_dir: Path,
    original_audio_path: Optional[Path] = None,
    vocal_audio_path: Optional[Path] = None,
    instrumental_audio_path: Optional[Path] = None,
    keep_audio: bool = False,
    is_tech_course: bool = True
) -> bool:
    """
    Orquestra a montagem do arquivo de áudio dublado final em sincronia.
    - Ajusta velocidades em paralelo via FFmpeg com semáforo no event loop atual.
    - Mescla com crossfade suave de 30ms sem sobrescrita destrutiva em overflow.
    - Mantém mapeamento exato por índice de segmento (zero risco de desalinhamento).
    - Aplica Modo Filme (0%/0%) como padrão ou Voice-Over quando solicitado (--keep-audio).
    """
    logging.info(MSG["sincronizando"])

    total_samples = int((total_duration_ms / 1000.0) * AUDIO_SAMPLE_RATE)
    if total_samples <= 0:
        logging.error("❌ Duração total inválida para montagem do áudio.")
        return False

    dubbed_audio = np.zeros(total_samples, dtype=np.float32)
    ffmpeg_semaphore = asyncio.Semaphore(6)

    orig_data = None
    if original_audio_path and original_audio_path.exists():
        try:
            raw_orig, orig_sr = sf.read(str(original_audio_path), dtype='float32')
            orig_data = _resample_if_needed(raw_orig, orig_sr)
        except Exception as e:
            logging.warning(f"⚠️ Não foi possível ler o áudio original para Fallback/Ducking: {e}")

    # --- FASE 1: AGENDAMENTO E EXECUÇÃO DO AJUSTE DE VELOCIDADE EM PARALELO ---
    speed_tasks = []
    task_meta = []

    for i, seg in enumerate(segments):
        if not seg.get("tts_success") or not seg.get("audio_path"):
            continue

        audio_path = Path(seg["audio_path"])
        if not audio_path.exists():
            continue

        seg_duration_ms = seg["end_ms"] - seg["start_ms"]
        if seg_duration_ms <= 0:
            continue

        gap_to_next_ms = total_duration_ms - seg["end_ms"]
        for j in range(i + 1, len(segments)):
            next_seg = segments[j]
            if next_seg.get("tts_success"):
                gap_to_next_ms = next_seg["start_ms"] - seg["end_ms"]
                break

        overflow_ms = max(0, min(gap_to_next_ms * OVERFLOW_GAP_RATIO, OVERFLOW_MAX_MS))
        target_ms = seg_duration_ms + overflow_ms

        adjusted_path = temp_dir / f"adj_{i:04d}.wav"
        speed_tasks.append(
            adjust_segment_speed_async(audio_path, adjusted_path, target_ms, ffmpeg_semaphore)
        )
        task_meta.append((i, adjusted_path))

    logging.info(f"⚡ Ajustando velocidade de {len(speed_tasks)} falas concorrentemente...")
    speed_results = await asyncio.gather(*speed_tasks) if speed_tasks else []
    adjusted_map = {
        idx: (adj_path, ok)
        for (idx, adj_path), ok in zip(task_meta, speed_results)
    }

    # --- FASE 2: MONTAGEM E MIXAGEM DOS ÁUDIOS EM MEMÓRIA ---
    placed = 0
    placed_regions = []
    crossfade_samples = int(0.030 * AUDIO_SAMPLE_RATE)

    for i, seg in enumerate(segments):
        start_ms = seg["start_ms"]
        end_ms = seg["end_ms"]
        start_sample = int((start_ms / 1000.0) * AUDIO_SAMPLE_RATE)
        end_sample = int((end_ms / 1000.0) * AUDIO_SAMPLE_RATE)

        if i in adjusted_map:
            adjusted_path, speed_ok = adjusted_map[i]
            success_placed = False

            if speed_ok and adjusted_path.exists():
                try:
                    raw_seg, sr = sf.read(str(adjusted_path), dtype='float32')
                    audio_data = _resample_if_needed(raw_seg, sr)

                    if len(audio_data) > crossfade_samples * 2:
                        audio_data[:crossfade_samples] *= np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
                        audio_data[-crossfade_samples:] *= np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)

                    end_sample_placed = min(start_sample + len(audio_data), total_samples)
                    available_space = end_sample_placed - start_sample

                    if available_space > 0 and start_sample >= 0:
                        dubbed_audio[start_sample:end_sample_placed] += audio_data[:available_space]
                        placed += 1
                        success_placed = True

                        actual_end_ms = (end_sample_placed / AUDIO_SAMPLE_RATE) * 1000.0
                        placed_regions.append({
                            "start_ms": start_ms,
                            "end_ms": actual_end_ms,
                            "tts_success": True
                        })
                except Exception as e:
                    logging.warning(f"⚠️ Erro ao ler áudio ajustado do segmento {i}: {e}")

            if not success_placed and not seg.get("tts_skipped") and orig_data is not None:
                logging.warning(f"⚠️ Fallback de emergência (falha pós-ajuste) para o segmento {i}: voz original injetada.")
                avail = min(end_sample, len(orig_data)) - start_sample
                if avail > 0 and start_sample >= 0:
                    dubbed_audio[start_sample:start_sample + avail] += orig_data[start_sample:start_sample + avail]
                    placed += 1
        else:
            if not seg.get("tts_skipped") and orig_data is not None and end_sample > start_sample:
                logging.warning(f"⚠️ Fallback de emergência: TTS do segmento {i} falhou. Injetando áudio original.")
                avail = min(end_sample, len(orig_data)) - start_sample
                if avail > 0 and start_sample >= 0:
                    dubbed_audio[start_sample:start_sample + avail] += orig_data[start_sample:start_sample + avail]
                    placed += 1

    if placed == 0:
        logging.error("❌ Erro fatal: Nenhum segmento pôde ser posicionado no áudio final.")
        return False

    # --- FASE 3: MIXAGEM DE TRILHAS E DUCKING INTELIGENTE ---

    # Caso 1: Trilha instrumental separada via --separate-audio (ONNX)
    if instrumental_audio_path and instrumental_audio_path.exists():
        try:
            logging.info("🎸 Mixando trilha instrumental/efeitos separada...")
            raw_inst, inst_sr = sf.read(str(instrumental_audio_path), dtype='float32')
            inst_data = _resample_if_needed(raw_inst, inst_sr)

            min_len = min(len(dubbed_audio), len(inst_data))
            dubbed_audio[:min_len] += inst_data[:min_len] * INSTRUMENTAL_VOLUME

            if keep_audio and vocal_audio_path and vocal_audio_path.exists():
                raw_voc, voc_sr = sf.read(str(vocal_audio_path), dtype='float32')
                voc_data = _resample_if_needed(raw_voc, voc_sr)

                duck_vol, orig_vol = (0.06, 0.25) if is_tech_course else (0.15, 0.50)
                ducking_regions = placed_regions if placed_regions else segments
                envelope = _build_ducking_envelope(min_len, ducking_regions, duck_vol, orig_vol, DUCKING_FADE_MS)
                dubbed_audio[:min_len] += voc_data[:min_len] * envelope[:min_len]

            logging.info("✅ Mixagem instrumental realizada com sucesso.")
        except Exception as e:
            logging.warning(f"⚠️ Falha ao mixar faixa instrumental: {e}")

    # Caso 2: Mixagem com trilha original (Modo Filme padrão 0%/0% ou Voice-Over com --keep-audio)
    elif orig_data is not None:
        try:
            if keep_audio:
                if is_tech_course:
                    duck_vol, orig_vol = 0.06, 0.25
                    logging.info("🎤 Modo Voice-Over (Screencast/TI): Original a 25% nos gaps / 6% sob dublagem.")
                else:
                    duck_vol, orig_vol = 0.15, 0.50
                    logging.info("🔊 Modo Voice-Over Padrão: Original a 50% nos gaps / 15% sob dublagem.")

                min_len = min(len(dubbed_audio), len(orig_data))
                ducking_regions = placed_regions if placed_regions else segments
                envelope = _build_ducking_envelope(min_len, ducking_regions, duck_vol, orig_vol, DUCKING_FADE_MS)
                dubbed_audio[:min_len] += orig_data[:min_len] * envelope[:min_len]
                logging.info("✅ Ducking dinâmico do áudio original aplicado.")
            else:
                duck_vol = DUCKING_VOLUME
                orig_vol = ORIGINAL_VOLUME
                if duck_vol == 0.0 and orig_vol == 0.0:
                    logging.info("🎬 Modo Filme: Voz original completamente substituída pela dublagem.")
                else:
                    min_len = min(len(dubbed_audio), len(orig_data))
                    logging.info(f"🎤 Ducking Customizado: Original a {int(orig_vol*100)}% nos gaps / {int(duck_vol*100)}% sob dublagem.")
                    ducking_regions = placed_regions if placed_regions else segments
                    envelope = _build_ducking_envelope(min_len, ducking_regions, duck_vol, orig_vol, DUCKING_FADE_MS)
                    dubbed_audio[:min_len] += orig_data[:min_len] * envelope[:min_len]
        except Exception as e:
            logging.warning(f"⚠️ Falha ao aplicar mixagem/ducking: {e}")

    # --- FASE 4: PREENCHIMENTO DE GAPS COM ROOM TONE ---
    if orig_data is not None:
        try:
            room_tone = _extract_room_tone(orig_data, AUDIO_SAMPLE_RATE)
            dubbed_audio = _fill_silent_gaps(
                dubbed_audio, room_tone, AUDIO_SAMPLE_RATE,
                silence_threshold=0.001, min_gap_ms=80
            )
            logging.info("🔇 Room tone seamless aplicado: sem gaps de silêncio digital artificial.")
        except Exception as e:
            logging.warning(f"⚠️ Falha ao preencher gaps com room tone: {e}")

    # Normalização final preventiva contra clipping
    peak = float(np.max(np.abs(dubbed_audio)))
    if peak > 0.95:
        dubbed_audio *= 0.95 / peak

    sf.write(str(output_path), dubbed_audio, AUDIO_SAMPLE_RATE)
    logging.info(f"✅ Arquivo de áudio consolidado gerado: {output_path.name}")
    return True
