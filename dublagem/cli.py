# -*- coding: utf-8 -*-
"""
Orquestrador CLI e Pipeline Principal da Dublagem v3.2 (Otimizado para Screencasts e Cursos de TI)
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
    WHISPER_MODEL, IDIOMA_DESTINO, DUCKING_VOLUME, ORIGINAL_VOLUME
)
from dublagem.utils import (
    calculate_video_hash, get_cache_paths, load_cache, save_cache
)
from dublagem.transcriber import transcribe_audio
from dublagem.translator import translate_segments
from dublagem.synthesizer import synthesize_all_segments
from dublagem.audio_processor import separate_audio, assemble_dubbed_audio, get_duration_ms
from dublagem.video_processor import extract_original_audio, replace_video_audio, save_srt, parse_srt


def natural_sort_key(path: Path) -> List:
    """Chave para ordenação alfanumérica natural humana (ex: 1, 2, ..., 10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]


async def process_video(
    video_path: Path,
    output_dir: Path,
    src_lang: str = "auto",
    dest_lang: str = IDIOMA_DESTINO,
    voice: str = VOZ_PADRAO,
    rate: str = VELOCIDADE_PADRAO,
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
    Pipeline principal de processamento de um vídeo.
    Gerencia o cache inteligente das etapas (por hash de vídeo, modelo e idioma) e orquestra a execução modular.
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

        safe_rate = rate.replace('+', 'plus').replace('-', 'minus').replace('%', '')
        voice_audio_cache_dir = cache["dir"] / f"audio_{voice}_{safe_rate}"
        voice_audio_cache_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Extração de Áudio Original ---
        original_wav = cache["dir"] / "original.wav"
        if not original_wav.exists() or force:
            extract_original_audio(video_path, original_wav)
        else:
            logging.info("💾 Áudio original restaurado do cache local.")

        # --- 2. Isolamento de Voz/Trilha (Opcional via --separate-audio ou config) ---
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

        # --- 3. Transcrição (faster-whisper com VAD) ---
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

            # --- 4. Tradução (Deep Translator + fallback MyMemory) ---
            translated = None if force else load_cache(cache["translation"])
            if translated is None:
                effective_src = detected_lang if src_lang == "auto" else src_lang
                translated = translate_segments(segments, effective_src, dest_lang)
                if not translated:
                    logging.error("❌ Falha na tradução dos segmentos.")
                    return False
                save_cache(cache["translation"], translated)
            else:
                logging.info("💾 Tradução restaurada do cache local.")

            save_srt(translated, output_srt)

        if srt_only:
            logging.info(f"✅ Subtítulos traduzidos criados com sucesso em: {output_srt.name}")
            logging.info("💡 Revise a legenda se necessário e execute novamente com '--from-srt' para dublar.")
            return True

        # --- 5. Síntese de Voz (Edge TTS) ---
        synth_segments = []
        segments_to_synthesize = []

        for i, seg in enumerate(translated):
            text_hash = hashlib.md5(seg["text"].encode("utf-8")).hexdigest()[:8]
            seg_cache_mp3 = voice_audio_cache_dir / f"seg_{i:04d}_{text_hash}.mp3"

            if not force and seg_cache_mp3.exists() and seg_cache_mp3.stat().st_size > 100:
                synth_segments.append({
                    **seg,
                    "audio_path": str(seg_cache_mp3),
                    "tts_success": True,
                    "tts_skipped": False,
                })
            else:
                segments_to_synthesize.append((i, {
                    **seg,
                    "target_path": str(seg_cache_mp3)
                }))
                synth_segments.append(None)

        if segments_to_synthesize:
            logging.info(f"🎙️ Geração de Voz: {len(segments_to_synthesize)} novos segmentos serão sintetizados.")
            raw_to_synth = [item[1] for item in segments_to_synthesize]

            synth_results = await synthesize_all_segments(
                raw_to_synth, voice_audio_cache_dir, voice, rate, concurrency=5
            )

            for idx_res, (orig_idx, _) in enumerate(segments_to_synthesize):
                synth_segments[orig_idx] = synth_results[idx_res]
        else:
            logging.info("💾 Todos os segmentos de voz foram restaurados do cache local.")

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
        description="LexiVox — Sistema de Dublagem Automática de Vídeos v3.2 (Otimizado para Screencasts e Tutoriais)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input", "-i", type=str, help="Arquivo de vídeo ou diretório de entrada")
    parser.add_argument("--output", "-o", type=str, help="Diretório de saída para os vídeos dublados")
    parser.add_argument("--src", default="auto", help="Idioma de origem do áudio (auto = detecção automática)")
    parser.add_argument("--dest", default=IDIOMA_DESTINO, help=f"Idioma de destino para a tradução e dublagem (padrão: {IDIOMA_DESTINO})")
    parser.add_argument("--voice", default=VOZ_PADRAO, help=f"Voz do Edge TTS (padrão: {VOZ_PADRAO})")
    parser.add_argument("--rate", default=VELOCIDADE_PADRAO, help=f"Fator de velocidade da voz (padrão: {VELOCIDADE_PADRAO.replace('%', '%%')})")
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

    if from_srt_path and not from_srt_path.exists():
        print(f"\n❌ Arquivo SRT não encontrado: {from_srt_path}")
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

    print(f"\n🎙️  LEXIVOX — SISTEMA DE DUBLAGEM AUTOMÁTICA v3.2")
    print("=" * 70)
    print(f"📂 Entrada:        {input_path}")
    print(f"📂 Saída:          {output_dir}")
    print(f"📽️  Vídeos:         {len(video_files)} arquivo(s)")
    print(f"🗣️  Voz TTS:        {args.voice}")
    print(f"🌐 Idiomas:        {args.src} ➡️ {args.dest}")
    print(f"🧠 Motor Whisper:  faster-whisper ({args.model.upper()}) + Silero VAD")
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
