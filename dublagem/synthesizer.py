# -*- coding: utf-8 -*-
"""
Módulo de Síntese de Voz Paralela via Microsoft Edge TTS
"""

import asyncio
import logging
import random
import re
from pathlib import Path
from typing import List, Dict

import edge_tts
from tqdm import tqdm

from dublagem.config import (
    MAX_RETRIES, RETRY_DELAY, TTS_TIMEOUT, MSG
)


def _has_speakable_text(text: str) -> bool:
    """Verifica se o texto contém ao menos um caractere alfanumérico pronunciável."""
    return bool(text and text.strip() and re.search(r"\w", text))


async def synthesize_segment(
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    semaphore: asyncio.Semaphore
) -> bool:
    """
    Sintetiza um segmento de texto individual em arquivo de áudio MP3.
    Executado sob semáforo de concorrência com backoff exponencial, jitter e limpeza de parciais.
    """
    if not _has_speakable_text(text):
        return False

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                communicate = edge_tts.Communicate(text, voice=voice, rate=rate)

                await asyncio.wait_for(
                    communicate.save(str(output_path)),
                    timeout=TTS_TIMEOUT
                )

                if output_path.exists() and output_path.stat().st_size > 100:
                    return True

                # Arquivo ausente ou corrompido (<= 100 bytes)
                output_size = output_path.stat().st_size if output_path.exists() else 0
                output_path.unlink(missing_ok=True)
                raise RuntimeError(f"Áudio MP3 inválido ou vazio ({output_size} bytes).")

            except Exception as e:
                output_path.unlink(missing_ok=True)
                err_lower = str(e).lower()
                is_throttled = any(k in err_lower for k in ("429", "403", "rate", "too many"))
                wait_time = RETRY_DELAY * (2 ** attempt) + random.uniform(0.3, 1.2)
                if is_throttled:
                    wait_time += 2.0

                if attempt < MAX_RETRIES - 1:
                    logging.warning(
                        f"⚠️ Falha no TTS para '{text[:30]}...' (tentativa {attempt + 1}/{MAX_RETRIES}). "
                        f"Aguardando {wait_time:.1f}s... Erro: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"❌ Falha fatal no Edge TTS para segmento: {text[:40]}... Erro: {e}")

        return False


async def synthesize_all_segments(
    segments: List[Dict],
    temp_dir: Path,
    voice: str,
    rate: str,
    concurrency: int = 5
) -> List[Dict]:
    """
    Sintetiza a voz de todos os segmentos em paralelo utilizando um pool assíncrono.
    Segmentos apenas com pontuação/símbolos são marcados como 'tts_skipped' sem desperdício de retries.
    """
    logging.info(f"{MSG['sintetizando']} (Voz: {voice} | Concorrência: {concurrency})")

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int, text: str, path: Path):
        success = await synthesize_segment(text, path, voice, rate, semaphore)
        return index, success, path

    results = [None] * len(segments)
    tasks = []
    skipped_count = 0

    for i, seg in enumerate(segments):
        if not _has_speakable_text(seg.get("text", "")):
            results[i] = {
                **seg,
                "audio_path": None,
                "tts_success": False,
                "tts_skipped": True,
            }
            skipped_count += 1
            continue

        seg_path = Path(seg["target_path"]) if seg.get("target_path") else (temp_dir / f"seg_{i:04d}.mp3")
        tasks.append(worker(i, seg["text"], seg_path))

    if tasks:
        with tqdm(total=len(tasks), desc="Sintetizando Voz", unit="seg", dynamic_ncols=True) as pbar:
            for fut in asyncio.as_completed(tasks):
                index, success, path = await fut
                results[index] = {
                    **segments[index],
                    "audio_path": str(path) if success else None,
                    "tts_success": success,
                    "tts_skipped": False,
                }
                pbar.update(1)

    ok = sum(1 for r in results if r and r.get("tts_success"))
    skip_info = f" ({skipped_count} sem texto pronunciável ignorados)" if skipped_count else ""
    logging.info(f"✅ Geração de Voz Concluída: {ok}/{len(results)} segmentos sintetizados com sucesso{skip_info}.")
    return results
