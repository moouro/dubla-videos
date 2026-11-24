# -*- coding: utf-8 -*-
"""
Orquestrador CLI e Pipeline Principal da Dublagem v4.0 (100% Offline com Qualidade Máxima)
"""

import argparse
import asyncio
import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional, List

from dublagem.config import (
    INPUT_DIR, OUTPUT_DIR, VOZ_PADRAO, VELOCIDADE_PADRAO,
    FORMATOS_VIDEO, USE_AUDIO_SEPARATOR, AUDIO_SEPARATOR_MODEL,
    TRANSCRIBE_ORIGINAL_VOCALS, setup_logging, MSG,
    WHISPER_MODEL, IDIOMA_DESTINO, DUCKING_VOLUME, ORIGINAL_VOLUME,
    MAX_FAILURE_RATIO, TRANSLATOR_MODEL, TTS_MODEL
)
from dublagem.utils import (
    calculate_video_hash, get_cache_paths, load_cache, save_cache
)
from dublagem.transcriber import transcribe_audio
from dublagem.translator import translate_segments
from dublagem.synthesizer import synthesize_all_segments, _resolve_speaker, _has_speakable_text
from dublagem.audio_processor import separate_audio, assemble_dubbed_audio, get_duration_ms
from dublagem.video_processor import extract_original_audio, replace_video_audio, save_srt, parse_srt


def natural_sort_key(path: Path) -> List:
    """Chave para ordenação alfanumérica natural humana (ex: 1, 2, ..., 10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]


def _sanitize_for_path(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", value.strip()) or "default"


async def process_video(
    video_path: Path,
    output_dir: Path,
    src_lang: str = "auto",
    dest_lang: str = IDIOMA_DESTINO,
    voice: str = VOZ_PADRAO,
    rate: str = VELOCIDADE_PADRAO,
    speaker_wav: Optional[Path] = None,
    keep_audio: bool = False,
    srt_only: bool = False,
    from_srt: Optional[Path] = None,
    use_separator: bool = USE_AUDIO_SEPARATOR,
    explicit_separator: bool = False,
    model_name: str = WHISPER_MODEL,
    is_tech_course: bool = True,
    force: bool = False
) -> bool:
    """
    Pipeline principal de processamento 100% offline de um vídeo.
    Aplica Fail-Fast rigoroso (> MAX_FAILURE_RATIO) para nunca gerar vídeo falsamente dublado.
    """
    logging.info(f"{'=' * 60}")
    logging.info(f"🎬 Iniciando processamento: {video_path.name}")
    logging.info(f"{'=' * 60}")

    base_name = video_path.stem
    output_video = output_dir / f"{base_name}_dubbed.mp4"
    output_srt = output_dir / f"{base_name}_dubbed.srt"

    if not force and not srt_only and not from_srt and output_video.exists():
        logging.info(f"⏭️  Vídeo já dublado anteriormente: {output_video.name} (use --force para reprocessar)")
        return True

    try:
        # === CÁLCULO DE CACHE INTELIGENTE ===
        logging.info("🔍 Calculando hash do vídeo para verificação de cache...")
        video_hash = calculate_video_hash(video_path)
        cache = get_cache_paths(video_hash, model_name=model_name, dest_lang=dest_lang)

        resolved_speaker = _resolve_speaker(voice)
        safe_voice = _sanitize_for_path(resolved_speaker)
        if speaker_wav and speaker_wav.exists():
            ref_hash = hashlib.md5(str(speaker_wav.resolve()).encode("utf-8")).hexdigest()[:6]
            safe_voice = f"clone_{ref_hash}"
        safe_rate = rate.replace('+', 'plus').replace('-', 'minus').replace('%', '')
        voice_audio_cache_dir = cache["dir"] / f"audio_xtts_{safe_voice}_{safe_rate}"
        voice_audio_cache_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Extração de Áudio Original ---
        original_wav = cache["dir"] / "original.wav"
        if not original_wav.exists() or force:
            extract_original_audio(video_path, original_wav)
        else:
            logging.info("💾 Áudio original restaurado do cache local.")

        # --- 2. Isolamento de Voz/Trilha (Opcional via --separate-audio) ---
        vocal_wav = None
        instrumental_wav = None

        if explicit_separator:
            use_separator = True
        elif use_separator and is_tech_course:
            logging.info("ℹ️ Tutoriais/Screencasts: Separador desativado por padrão (use --separate-audio para forçar).")
            use_separator = False

        if use_separator:
            cached_vocal = cache["dir"] / f"vocals_{AUDIO_SEPARATOR_MODEL}.wav"
            cached_instrumental = cache["dir"] / f"instrumental_{AUDIO_SEPARATOR_MODEL}.wav"

            if cached_vocal.exists() and cached_instrumental.exists() and not force:
                vocal_wav = cached_vocal
                instrumental_wav = cached_instrumental
                logging.info("💾 Faixas vocal/instrumental isoladas restauradas do cache local.")
            else:
                vocal_wav, instrumental_wav = separate_audio(original_wav, cache["dir"], AUDIO_SEPARATOR_MODEL)
                if vocal_wav and instrumental_wav:
                    vocal_wav = vocal_wav.replace(cached_vocal)
                    instrumental_wav = instrumental_wav.replace(cached_instrumental)

        # --- 3. Transcrição Offline (faster-whisper com VAD e Travas Anti-Loop) ---
        segments = None
        detected_lang = src_lang

        if from_srt:
            logging.info(f"📄 Carregando legenda customizada: {from_srt.name}")
            translated = parse_srt(from_srt)
            if not translated:
                logging.error("❌ O arquivo SRT fornecido está vazio ou é inválido.")
                return False
        else:
            cached_trans = None if force else load_cache(cache["transcription"])

            if isinstance(cached_trans, dict) and "segments" in cached_trans:
                segments = cached_trans["segments"]
                detected_lang = cached_trans.get("language", src_lang)
                logging.info(f"💾 Transcrição restaurada do cache local (idioma: {detected_lang}).")
            elif isinstance(cached_trans, list):
                segments = cached_trans
                logging.info("💾 Transcrição restaurada do cache local (formato legado).")

            if segments is None:
                transcribe_input = vocal_wav if (use_separator and vocal_wav and TRANSCRIBE_ORIGINAL_VOCALS) else original_wav
                segments, detected_lang = transcribe_audio(transcribe_input, model_name=model_name, language=src_lang)

                if not segments:
                    logging.error("❌ Transcrição falhou: nenhum segmento de áudio detectado.")
                    return False
                save_cache(cache["transcription"], {"language": detected_lang, "segments": segments})

            # --- 4. Tradução Offline (NLLB-200 via CTranslate2) ---
            translated = None if force else load_cache(cache["translation"])
            if translated is not None:
                # Rejeita caches antigos/corrompidos com muitas falhas de tradução
                cached_failed = sum(1 for s in translated if s.get("translation_failed"))
                if translated and (cached_failed / len(translated)) > MAX_FAILURE_RATIO:
                    logging.warning("⚠️ Cache de tradução anterior contém falhas acima do limite. Retraduzindo...")
                    translated = None

            if translated is None:
                effective_src = detected_lang if src_lang == "auto" else src_lang
                translated, failed_trans = translate_segments(segments, effective_src, dest_lang)
                if not translated:
                    logging.error("❌ Falha na tradução dos segmentos.")
                    return False

                speakable_trans = [s for s in translated if _has_speakable_text(s.get("text", ""))]
                if speakable_trans:
                    failure_ratio = failed_trans / len(speakable_trans)
                    if failure_ratio > MAX_FAILURE_RATIO:
                        logging.error(
                            f"❌ FAIL-FAST (Tradução): {failed_trans}/{len(speakable_trans)} segmentos falharam "
                            f"({failure_ratio:.1%} > {MAX_FAILURE_RATIO:.0%}). Abortando para evitar vídeo sem tradução."
                        )
                        return False

                save_cache(cache["translation"], translated)
            else:
                logging.info("💾 Tradução restaurada do cache local.")

            save_srt(translated, output_srt)

        if srt_only:
            logging.info(f"✅ Subtítulos traduzidos criados com sucesso em: {output_srt.name}")
            logging.info("💡 Revise a legenda se necessário e execute novamente com '--from-srt' para dublar.")
            return True

        # --- 5. Síntese de Voz Offline (Coqui XTTS-v2) ---
        synth_segments = []
        segments_to_synthesize = []

        for i, seg in enumerate(translated):
            text = seg.get("text", "")
            if not _has_speakable_text(text):
                synth_segments.append({
                    **seg,
                    "audio_path": None,
                    "tts_success": False,
                    "tts_skipped": True,
                })
                continue

            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
            seg_cache_wav = voice_audio_cache_dir / f"seg_{i:04d}_{text_hash}.wav"

            if not force and seg_cache_wav.exists() and seg_cache_wav.stat().st_size > 1000:
                synth_segments.append({
                    **seg,
                    "audio_path": str(seg_cache_wav),
                    "tts_success": True,
                    "tts_skipped": False,
                })
            else:
                segments_to_synthesize.append((i, {
                    **seg,
                    "target_path": str(seg_cache_wav)
                }))
                synth_segments.append(None)

        if segments_to_synthesize:
            logging.info(f"🎙️ Geração de Voz (XTTS-v2): {len(segments_to_synthesize)} novos segmentos serão sintetizados.")
            raw_to_synth = [item[1] for item in segments_to_synthesize]

            synth_results = await synthesize_all_segments(
                segments=raw_to_synth,
                temp_dir=voice_audio_cache_dir,
                voice=voice,
                rate=rate,
                dest_lang=dest_lang,
                speaker_wav=speaker_wav,
                original_audio_path=original_wav
            )

            for idx_res, (orig_idx, _) in enumerate(segments_to_synthesize):
                synth_segments[orig_idx] = synth_results[idx_res]
        else:
            logging.info("💾 Todos os segmentos de voz XTTS-v2 foram restaurados do cache local.")

        # === FAIL-FAST RIGOROSO PÓS-TTS (>10% falha = aborta geração de vídeo) ===
        speakable_segs = [s for s in synth_segments if s and not s.get("tts_skipped")]
        failed_tts = [s for s in speakable_segs if not s.get("tts_success")]
        if not speakable_segs:
            logging.error("❌ FAIL-FAST (TTS): Nenhum segmento pronunciável para sintetizar.")
            return False

        tts_fail_ratio = len(failed_tts) / len(speakable_segs)
        if tts_fail_ratio > MAX_FAILURE_RATIO:
            logging.error(
                f"❌ FAIL-FAST (TTS): {len(failed_tts)}/{len(speakable_segs)} segmentos falharam na síntese "
                f"({tts_fail_ratio:.1%} > {MAX_FAILURE_RATIO:.0%}). "
                "Abortando geração de '_dubbed.mp4' para evitar saída com áudio original em inglês."
            )
            return False

        # --- 6. Montagem DSP de Áudio Sincronizado ---
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dubbed_wav = temp_path / "dubbed_final.wav"

            total_duration_ms = get_duration_ms(video_path)

            success = await assemble_dubbed_audio(
                segments=synth_segments,
                total_duration_ms=total_duration_ms,
                output_path=dubbed_wav,
                temp_dir=temp_path,
                original_audio_path=original_wav,
                vocal_audio_path=vocal_wav,
                instrumental_audio_path=instrumental_wav,
                keep_audio=keep_audio,
                is_tech_course=is_tech_course
            )

            if not success:
                logging.error("❌ Falha crítica ao realizar a montagem DSP do áudio.")
                return False

            # --- 7. Mesclagem do Vídeo e Áudio Dublado ---
            replace_video_audio(video_path, dubbed_wav, output_video)

        if output_video.exists() and output_video.stat().st_size > 1000:
            size_mb = output_video.stat().st_size / (1024 * 1024)
            logging.info(f"🎯 {MSG['concluido']} -> {output_video.name} ({size_mb:.1f} MB)")
            return True
        else:
            logging.error("❌ O arquivo final gerado está corrompido ou ausente.")
            return False

    except Exception as e:
        logging.error(f"❌ Erro crítico ao processar o vídeo {video_path.name}: {e}", exc_info=True)
        return False


async def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="LexiVox v4.0 — Sistema de Dublagem Automática 100% Offline (Whisper + NLLB-200 + Coqui XTTS-v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input", "-i", type=str, help="Arquivo de vídeo ou diretório de entrada")
    parser.add_argument("--output", "-o", type=str, help="Diretório de saída para os vídeos dublados")
    parser.add_argument("--src", default="auto", help="Idioma de origem do áudio (auto = detecção automática)")
    parser.add_argument("--dest", default=IDIOMA_DESTINO, help=f"Idioma de destino para tradução e dublagem (padrão: {IDIOMA_DESTINO})")
    parser.add_argument("--voice", default=VOZ_PADRAO, help=f"Voz XTTS-v2: marcos, damien, claribel, ana, clone ou nome exato (padrão: {VOZ_PADRAO})")
    parser.add_argument("--speaker-wav", type=str, default=None, help="Arquivo WAV (6–12s) para clonagem zero-shot no XTTS-v2")
    parser.add_argument("--rate", default=VELOCIDADE_PADRAO, help=f"Identificador de taxa/preset no cache (padrão: {VELOCIDADE_PADRAO.replace('%', '%%')})")
    parser.add_argument("--keep-audio", action="store_true", help="Ativa modo Voice-Over mantendo o áudio original audível ao fundo")
    parser.add_argument("--srt-only", action="store_true", help="Gera somente arquivo de legendas SRT traduzidas")
    parser.add_argument("--from-srt", type=str, default=None, help="Executa a dublagem a partir de um SRT editado manualmente (use com -i <video>)")
    parser.add_argument("--separate-audio", action="store_true", help="Força o uso do separador de áudio vocal/instrumental (requer audio-separator)")
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL,
        choices=["base", "small", "medium", "large-v3", "large-v3-turbo"],
        help=f"Modelo faster-whisper (padrão: {WHISPER_MODEL} para máxima fidelidade)"
    )
    parser.add_argument("--not-tech", action="store_true", help="Desativa ajustes específicos de cursos de TI (ex: com --keep-audio usa Voice-Over genérico 15%%/50%%)")
    parser.add_argument("--force", "-f", action="store_true", help="Força o reprocessamento mesmo se o vídeo já existir na pasta de saída")

    args = parser.parse_args()

    input_path = Path(args.input) if args.input else INPUT_DIR
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    from_srt_path = Path(args.from_srt) if args.from_srt else None
    speaker_wav_path = Path(args.speaker_wav) if args.speaker_wav else None

    if from_srt_path and not from_srt_path.exists():
        print(f"\n❌ Arquivo SRT não encontrado: {from_srt_path}")
        return
    if speaker_wav_path and not speaker_wav_path.exists():
        print(f"\n❌ Arquivo --speaker-wav não encontrado: {speaker_wav_path}")
        return

    video_files = []
    if input_path.is_file():
        if input_path.suffix.lower() in FORMATOS_VIDEO:
            video_files = [input_path]
    elif input_path.is_dir():
        for ext in FORMATOS_VIDEO:
            video_files.extend(input_path.glob(f"*{ext}"))
            video_files.extend(input_path.glob(f"*{ext.upper()}"))

    video_files = sorted(list(set(video_files)), key=natural_sort_key)

    if not video_files:
        print(f"\n❌ Nenhum arquivo de vídeo compatível encontrado em: {input_path}")
        print(f"💡 Formatos suportados: {', '.join(FORMATOS_VIDEO)}")
        return

    if from_srt_path and len(video_files) > 1:
        print("\n❌ '--from-srt' deve ser usado com um único vídeo específico via '-i <arquivo>'.")
        print(f"💡 Vídeos encontrados na pasta: {len(video_files)}. Informe qual deles corresponde ao SRT.")
        return

    is_tech_course = not args.not_tech

    if args.srt_only:
        modo = "SOMENTE LEGENDAS (SRT-ONLY)"
    elif from_srt_path:
        modo = f"A PARTIR DE SRT ({from_srt_path.name})"
    elif args.keep_audio:
        modo = "VOICE-OVER (TI: 6%/25%)" if is_tech_course else "VOICE-OVER GENÉRICO (15%/50%)"
    elif args.separate_audio or USE_AUDIO_SEPARATOR:
        modo = "MODO FILME + INSTRUMENTAL SEPARADO (ONNX)"
    elif DUCKING_VOLUME == 0.0 and ORIGINAL_VOLUME == 0.0:
        modo = "MODO FILME — SUBSTITUIÇÃO TOTAL (0%/0% + Room Tone)"
    else:
        modo = f"DUCKING CUSTOMIZADO ({int(DUCKING_VOLUME*100)}%/{int(ORIGINAL_VOLUME*100)}%)"

    resolved_voice_label = _resolve_speaker(args.voice)
    if speaker_wav_path:
        resolved_voice_label = f"Clone ({speaker_wav_path.name})"

    print(f"\n🎙️  LEXIVOX v4.0 — DUBLAGEM 100% OFFLINE DE ALTA QUALIDADE")
    print("=" * 70)
    print(f"📂 Entrada:        {input_path}")
    print(f"📂 Saída:          {output_dir}")
    print(f"📽️  Vídeos:         {len(video_files)} arquivo(s)")
    print(f"🗣️  Voz XTTS-v2:    {resolved_voice_label} ({TTS_MODEL.split('/')[-1]})")
    print(f"🌐 Tradutor:       {TRANSLATOR_MODEL} ({args.src} ➡️ {args.dest})")
    print(f"🧠 Transcrição:    faster-whisper ({args.model.upper()}) + Silero VAD + Anti-Loop")
    print(f"🎧 Modo Ativo:     {modo}")
    print("=" * 70 + "\n")

    successful = 0
    failed = 0

    for video_file in video_files:
        try:
            success = await process_video(
                video_path=video_file,
                output_dir=output_dir,
                src_lang=args.src,
                dest_lang=args.dest,
                voice=args.voice,
                rate=args.rate,
                speaker_wav=speaker_wav_path,
                keep_audio=args.keep_audio,
                srt_only=args.srt_only,
                from_srt=from_srt_path,
                use_separator=args.separate_audio or USE_AUDIO_SEPARATOR,
                explicit_separator=args.separate_audio,
                model_name=args.model,
                is_tech_course=is_tech_course,
                force=args.force
            )
            if success:
                successful += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\n⏹️  Operação cancelada pelo usuário.")
            break
        except Exception as e:
            logging.error(f"❌ Falha crítica ao processar {video_file.name}: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print("📊 SUMÁRIO DE EXECUÇÃO")
    print("=" * 70)
    print(f"✅ Vídeos com sucesso: {successful}")
    print(f"❌ Vídeos com falha:   {failed}")
    print(f"📂 Pasta de saída:     {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Operação cancelada pelo usuário.")
    except Exception as e:
        logging.error(f"❌ Erro fatal: {e}", exc_info=True)
        sys.exit(1)
