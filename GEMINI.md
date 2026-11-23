# GEMINI.md — Memória Técnica Permanente (LexiVox / Dubla-videos)

> Leia este arquivo no início de qualquer nova sessão neste repositório.
> Última atualização: Novembro de 2026 (v3.2 — Auditado e pronto para GitHub).

---

## 1. Visão Geral & Regras de Ouro

* **Propósito:** Pipeline local em Python para dublar vídeos técnicos/cursos/screencasts (principalmente inglês → português brasileiro `pt-BR`), geralmente sem câmera do instrutor na tela.
* **Prioridade Absoluta — Qualidade > Velocidade:** O PC do usuário opera 24/7. **Nunca** rebaixe o modelo padrão (`large-v3`) para `base` ou `small` por motivo de performance.
* **Modo de Áudio Padrão — Modo Filme (Substituição Total `0% / 0%`):**
  * O usuário rejeita explicitamente Voice-Over com voz em inglês audível no fundo por padrão.
  * Padrão obrigatório em `config_ptbr.py`: `DUCKING_VOLUME = 0.0` e `ORIGINAL_VOLUME = 0.0` (voz original 100% removida + Room Tone nos gaps + Smart Overflow `85% / 1500ms`).
  * Voice-Over (`6%/25%` em cursos TI ou `15%/50%` genérico) só deve ser ativado quando o usuário passar `--keep-audio`.

---

## 2. Stack & Ambiente Validado

* **OS:** Windows 11 (ReviOS), console requer UTF-8 (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`).
* **Hardware:** Intel Core i5-13400F (10 cores / 16 threads), 32 GB RAM, RX 6600M (GPU AMD não usada na inferência; `faster-whisper` roda em CPU `int8` com `10` threads).
* **Python:** `3.14.10` (64-bit).
* **FFmpeg / FFprobe:** Pacote Winget `yt-dlp.FFmpeg` (sem filtro `--enable-whisper`). `config_ptbr.py` injeta dinamicamente no `PATH` caso `shutil.which("ffmpeg")` retorne `None`.
* **Dependências Principais (`requirements.txt`):**
  * `faster-whisper>=1.2.0` + `onnxruntime>=1.18.0` (Silero VAD)
  * `edge-tts>=7.0.0`
  * `deep-translator>=1.11.0`
  * `soundfile>=0.12.0` + `numpy>=2.0.0` + `tqdm>=4.66.0`
  * Opcional fora do padrão: `audio-separator` (somente para `--separate-audio`).

---

## 3. Armadilhas Conhecidas (Não Regredir)

1. **Bug do PyAV (`av >= 19.0`) com `faster-whisper`:** `av.open(..., metadata_errors="ignore")` falha ao passar caminho de arquivo direto ao `model.transcribe()`. **Solução mantida:** `load_audio_for_whisper()` decodifica para PCM `s16le` 16kHz mono via subprocesso `ffmpeg` com pipe e entrega `np.ndarray float32`.
2. **Rate-limit agressivo do `GoogleTranslator` (HTTP 429):** `translator.py` faz fallback automático para `MyMemoryTranslator` com código regional dinâmico (`_to_mymemory_lang`), respeitando teto de **450 chars/requisição** do MyMemory e usando `ThreadPoolExecutor` + `shutdown(wait=False, cancel_futures=True)` para timeout real.
3. **`asyncio.WindowsProactorEventLoopPolicy` deprecado no Python 3.14:** **Não** reintroduzir em `run.py` ou `dublar.py`; usar apenas `asyncio.run(main())`.
4. **Semáforos `asyncio` em escopo de módulo:** No Python 3.12+, instanciar `asyncio.Semaphore` fora do loop causa problemas em testes/múltiplas execuções; manter criação dentro das corrotinas (`assemble_dubbed_audio`, `synthesize_all_segments`).
5. **Sobrescrita de segmentos com overflow e índices no DSP:** Na Fase 2 de `audio_processor.py`, usar mapeamento por índice (`adjusted_map`) e soma (`+=`) com crossfade de 30ms em vez de atribuição (`=`).
6. **Git Ownership no Windows:** Se o Git acusar `dubious ownership`, usar `-c safe.directory=C:/Users/Administrador/Desktop/Dubla-videos` ou adicionar nas configs globais.
7. **`.gitignore`:** Nunca commitar `.cache/`, `models/`, `input_videos/`, `output_videos/`, vídeos, áudios ou logs; manter versionados `glossario.txt`, `GEMINI.md`, `CONTEXTO_PROJETO.md`, `README.md`, `requirements.txt`.

---

## 4. Arquitetura dos Módulos

* `run.py` / `dublar.py`: entry points equivalentes (UTF-8 setup + `asyncio.run(main())`).
* `config_ptbr.py`: configuração editável pelo usuário (`WHISPER_*`, `VOZES`, `TERMOS_TECNICOS` + `glossario.txt`, `MERGE_*`, `DUCKING_*`) + diagnóstico `validar()`.
* `dublagem/config.py`: reexporta configs com defaults seguros alinhados e configura `logging` em `output_videos/dubbing.log`.
* `dublagem/utils.py`: hash MD5 rápido por conteúdo + tamanho (`calculate_video_hash`), resolução de caminhos de cache por `(video_hash, model_name, dest_lang)` e I/O JSON UTF-8.
* `dublagem/transcriber.py`: cache em memória do modelo `faster-whisper`, decode via FFmpeg, Silero VAD + word-level timestamps, retorno `(segments, detected_lang)`, deduplicação com clamping de sobreposições pequenas e merge inteligente por pausa/duração/caracteres/pontuação.
* `dublagem/translator.py`: proteção/restauração regex de termos técnicos (`_protect_tech_terms`, tolerância a espaços no placeholder e escaping com `lambda`), chunking, timeout real, backoff com jitter e fallback Google → MyMemory.
* `dublagem/synthesizer.py`: Edge TTS concorrente (`semaphore=5`), skip limpo de segmentos sem texto alfanumérico (`tts_skipped`), limpeza de `.mp3` corrompidos e backoff com jitter.
* `dublagem/audio_processor.py`: cálculo assíncrono de velocidade (`ffprobe` + `ffmpeg atempo`), Smart Overflow, fallback de áudio original em falhas reais de TTS, mixagem Modo Filme / Voice-Over / Instrumental ONNX e `_fill_silent_gaps` vetorizado com NumPy.
* `dublagem/video_processor.py`: extração WAV, remux `-c:v copy` com fallback `libx264` em codecs incompatíveis com MP4, geração/leitura resiliente de `.srt` (UTF-8 BOM + CRLF).
* `dublagem/cli.py`: parser CLI (`--input`, `--output`, `--src`, `--dest`, `--voice`, `--rate`, `--keep-audio`, `--srt-only`, `--from-srt`, `--separate-audio`, `--model`, `--not-tech`, `--force`), ordenação alfanumérica natural e orquestração com cache por etapa e hash de texto no TTS.

---

## 5. Estado Atual Validado (v3.2)

* Pipeline completo (`faster-whisper large-v3` + tradução com fallback MyMemory + Edge TTS + Modo Filme `0%/0%` + Room Tone) validado em vídeos screencast reais em `input_videos/`.
* Código auditado para consistência de índices, timeouts reais em subprocessos/requests, compatibilidade Windows UTF-8 e `.gitignore` preparado para publicação no GitHub.
* Nota local de disco: pode existir `models/ggml-large-v3.bin` (~3 GB) herdado da v2; está ignorado pelo Git e não é usado pela v3.2 (`models/faster-whisper/` é o diretório ativo).

---

## 6. Próximos Passos (Backlog Priorizado)

1. Integração nativa de separação com **Demucs (`htdemucs`)** para vídeos com música de fundo contínua.
2. **Diarização de falantes** (`pyannote-audio`) para alternar vozes PT-BR em diálogos/entrevistas.
3. Suporte a download direto com `yt-dlp` (`--url`), preview curto (`--preview`) e queima opcional de legendas (`--burn-srt`).
4. Interface Web local (Gradio ou Streamlit) com edição visual de `.srt` antes da dublagem.
