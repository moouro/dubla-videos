# GEMINI.md — Memória Técnica Permanente (LexiVox / Dubla-videos)

> Leia este arquivo no início de qualquer nova sessão neste repositório.
> Última atualização: Novembro de 2026 (v3.2 — Auditado e sincronizado com o GitHub).

---

## 1. Visão Geral e Stack

* **Propósito:** Pipeline local em Python para dublar vídeos técnicos, cursos de programação e screencasts (em sua maioria sem câmera do instrutor na tela, apenas voz sobre o código), principalmente de inglês para português brasileiro (`pt-BR`), com experiência de **filme dublado**.
* **Ambiente Validado:**
  * **SO:** Windows 11 (ReviOS), console UTF-8 (`reconfigure(encoding="utf-8", errors="replace")` antes dos imports).
  * **Hardware:** Intel Core i5-13400F (10 cores / 16 threads, AVX2), 32 GB RAM, AMD Radeon RX 6600M (não utilizada pelo Whisper; inferência roda em CPU `int8` com 10 threads).
  * **Python:** `3.14.10` (64-bit).
  * **FFmpeg / FFprobe:** Pacote Winget `yt-dlp.FFmpeg` (`N-128251`, compilado sem `--enable-whisper`). `config_ptbr.py` injeta o binário no `PATH` dinamicamente apenas se `shutil.which("ffmpeg")` retornar `None`.
* **Dependências (`requirements.txt`):**
  * Transcrição: `faster-whisper>=1.2.0` (`1.2.1` + `ctranslate2 4.8.2`) e `onnxruntime>=1.18.0` (`1.30.0`, Silero VAD).
  * Tradução: `deep-translator>=1.11.0` (`1.11.4` — `GoogleTranslator` com fallback `MyMemoryTranslator`).
  * Síntese TTS: `edge-tts>=7.0.0` (`7.2.8` — voz padrão `pt-BR-AntonioNeural`).
  * Áudio/DSP e CLI: `soundfile>=0.12.0` (`0.14.0`), `numpy>=2.0.0` (`2.5.3`), `tqdm>=4.66.0` (`4.70.1`).
  * Opcional fora do padrão: `audio-separator` (somente se `--separate-audio` for passado).
* **Fluxo do Pipeline (7 etapas):**
  1. Extração WAV mono `44100 Hz` (`video_processor.extract_original_audio`).
  2. Separação vocal/instrumental ONNX opcional (`audio_processor.separate_audio`).
  3. Transcrição `faster-whisper` com decodificação PCM `16 kHz` via pipe FFmpeg + Silero VAD + word-level timestamps + deduplicação com clamping + merge inteligente (`transcriber.transcribe_audio`).
  4. Tradução com proteção regex de termos técnicos (`config_ptbr.py` + `glossario.txt`), chunking, timeout real e fallback Google → MyMemory (`translator.translate_segments`) + geração `.srt` (`video_processor.save_srt`).
  5. Síntese concorrente Edge TTS (`concurrency=5`) com skip de trechos sem texto alfanumérico e cache por hash MD5 do texto (`synthesizer.synthesize_all_segments`).
  6. DSP assíncrono: `ffprobe` + `ffmpeg atempo`, Smart Overflow, soma não destrutiva (`+=`) com crossfade `30ms`, Modo Filme (`0%/0%`) ou Voice-Over (`--keep-audio`), Room Tone vetorizado com NumPy e normalização anti-clipping `0.95` (`audio_processor.assemble_dubbed_audio`).
  7. Remux MP4 via `-c:v copy` + AAC `192k` com fallback automático para `libx264` em containers incompatíveis (`video_processor.replace_video_audio`).
* **Mapa dos Arquivos:**
  * `run.py` / `dublar.py`: entry points equivalentes (`asyncio.run(main())` com setup UTF-8 pré-import).
  * `config_ptbr.py`: configurações do usuário, lista de termos técnicos + `glossario.txt`, e diagnóstico `validar()`.
  * `dublagem/config.py`: reexporta configs com defaults seguros alinhados e configura log em `output_videos/dubbing.log`.
  * `dublagem/utils.py`: `calculate_video_hash`, `get_cache_paths(video_hash, model_name, dest_lang)`, `load_cache`, `save_cache`.
  * `dublagem/transcriber.py`, `dublagem/translator.py`, `dublagem/synthesizer.py`, `dublagem/audio_processor.py`, `dublagem/video_processor.py`, `dublagem/cli.py`.

---

## 2. Decisões Técnicas e Ajustes de Qualidade

* **Qualidade Máxima > Velocidade:** Máquina ligada 24/7; padrão obrigatório é `WHISPER_MODEL = "large-v3"`, `WHISPER_DEVICE = "cpu"`, `WHISPER_COMPUTE_TYPE = "int8"`, `WHISPER_CPU_THREADS = 10`, `beam_size=5`, `best_of=5`, `temperature=[0.0, 0.2, 0.4]`. Nunca rebaixar o padrão para `base` ou `small`.
* **Modo Filme Obrigatório por Padrão (`0% / 0%`):**
  * `DUCKING_VOLUME = 0.0` e `ORIGINAL_VOLUME = 0.0` removem 100% da voz original (sem fala em inglês audível ao fundo).
  * Voice-Over (`6%` sob fala / `25%` nos gaps para cursos TI, ou `15%/50%` com `--not-tech`) só é ativado explicitamente via `--keep-audio`.
* **Parâmetros Validados de Sincronia e Fluidez:**
  * Velocidade TTS: `VELOCIDADE_PADRAO = "+0%"`, `MIN_SPEED_FACTOR = 0.7`, `MAX_SPEED_FACTOR = 1.15`.
  * Smart Overflow (sem boca humana em screencasts): `OVERFLOW_GAP_RATIO = 0.85`, `OVERFLOW_MAX_MS = 1500`.
  * Merge Inteligente: `MERGE_MAX_GAP_MS = 600`, `MERGE_MAX_DURATION_MS = 14000`, `MERGE_MAX_CHARS = 450`.
  * VAD & Áudio: `WHISPER_VAD_FILTER = True`, `WHISPER_VAD_MIN_SILENCE_MS = 400`, `speech_pad_ms = 200`, `DUCKING_FADE_MS = 150`, crossfade `30ms`, Room Tone em gaps `>= 80ms` (`silence_threshold=0.001`).
* **Erros Conhecidos Já Resolvidos (Não Regredir):**
  1. **PyAV (`av >= 19.0`) com `faster-whisper`:** `av.open(..., metadata_errors="ignore")` falha ao passar caminho direto para `model.transcribe()`. Solução: `load_audio_for_whisper()` decodifica PCM `s16le` `16 kHz` mono via pipe `ffmpeg` e entrega `np.ndarray float32`.
  2. **Rate-limit do `GoogleTranslator` (HTTP 429):** fallback automático para `MyMemoryTranslator` com mapeamento regional dinâmico (`_to_mymemory_lang`), teto de `450` chars/chunk e `ThreadPoolExecutor` com `shutdown(wait=False, cancel_futures=True)` para timeout real.
  3. **`asyncio.WindowsProactorEventLoopPolicy` deprecado no Python 3.14:** removido dos entry points.
  4. **`asyncio.Semaphore` em escopo de módulo:** instanciado sempre dentro das corrotinas (`assemble_dubbed_audio`, `synthesize_all_segments`).
  5. **Desalinhamento de índices e corte de cauda no DSP:** Fase 2 de `audio_processor.py` mapeia por índice (`adjusted_map`) e usa soma (`+=`) em vez de atribuição (`=`).
  6. **Perda de fala em sobreposição mínima de timestamps:** `_deduplicate_segments()` faz clamping do `start_ms` em pequenas sobreposições (`speech_pad_ms` / `word_timestamps`).
  7. **Console Windows `cp1252`:** UTF-8 forçado antes dos imports em `run.py` e `dublar.py`; subprocessos FFmpeg decodificados com `encoding="utf-8", errors="replace"`.
  8. **Git `dubious ownership` no Windows:** usar `-c safe.directory=C:/Users/Administrador/Desktop/Dubla-videos` ou adicionar nas configs globais.

---

## 3. Estado Atual (Ponta a Ponta — v3.2)

- [x] Pipeline completo validado fim-a-fim em vídeos screencast reais (`input_videos/` → `output_videos/`).
- [x] Transcrição `faster-whisper large-v3` com Silero VAD, word-level timestamps, clamping seguro e Merge Inteligente de Segmentos (gap/duração/caracteres/pontuação sem quebrar abreviações).
- [x] Tradução com preservação de +120 termos técnicos (`config_ptbr.py` + `glossario.txt`), chunking, timeout real e fallback automático para MyMemory usando idioma detectado pelo Whisper.
- [x] Síntese concorrente Edge TTS com cache por `(voz, velocidade, hash MD5 do texto)`, limpeza de parciais corrompidos, skip limpo de trechos sem texto alfanumérico (`tts_skipped`) e backoff com jitter.
- [x] Mixagem DSP Modo Filme (`0%/0%`) + Room Tone vetorizado + Smart Overflow (`85% / 1500ms`) e suporte funcional às flags `--keep-audio`, `--not-tech`, `--separate-audio` e `--force`.
- [x] Modo semi-automático `--srt-only` / `--from-srt` (parse SRT compatível com UTF-8 BOM + CRLF e proteção contra uso sem `-i` em pastas com múltiplos vídeos) + fallback `libx264` no mux MP4.
- [x] Ordenação alfanumérica natural de aulas, cache versionado por `(hash de conteúdo, modelo, idioma)` e `.gitignore` blindado sincronizado com o GitHub (`origin/main`).

---

## 4. Próximos Passos Planejados

1. **Separação Nativa com Demucs (`htdemucs`):** Integração direta e out-of-the-box para vídeos com música de fundo contínua, sem depender de configuração manual de `audio-separator`.
2. **Diarização de Falantes (`pyannote-audio`):** Detectar múltiplos locutores em entrevistas/diálogos e alternar vozes PT-BR automaticamente.
3. **Automação CLI:** Download direto via `yt-dlp` (`--url`), modo de preview rápido (`--preview`) e queima opcional de legendas (`--burn-srt`).
4. **Interface Web Local (Gradio/Streamlit):** Painel visual para envio de vídeos e revisão rápida de `.srt` antes da síntese.
5. *(Manutenção local opcional)* Excluir `models/ggml-large-v3.bin` (~3 GB, legado da v2 e já ignorado pelo Git; `models/faster-whisper/` é o ativo).

---

## 5. Manutenção de Memória do Projeto

* Sempre que o usuário pedir para **"salvar progresso"**, **"encerrar sessão"** ou **"atualizar memória"**, atualize o arquivo `GEMINI.md` na raiz do projeto marcando as tarefas concluídas em **"Estado Atual"**, ajustando os **"Próximos Passos"** e registrando a data e o resumo técnico das mudanças no final do arquivo.

---

## 6. Resumo Técnico da Última Sessão (`2026-11-23` — Auditoria & Release v3.2)

* **Correções de bugs e resiliência aplicadas em toda a base `dublagem/`:**
  * `audio_processor.py`: corrigido desalinhamento de `speed_result_idx` via `adjusted_map`; trocado `=` por `+=` no timeline para preservar cauda de overflow; corrigida lógica de `--keep-audio` e `--separate-audio`; movido `FFMPEG_SEMAPHORE` para escopo da corrotina; adicionado `get_duration_ms_async()` + timeout `60s` no FFmpeg com `proc.kill()`; vetorizado `_fill_silent_gaps` com NumPy; evitado cálculo de envelope quando `DUCKING_VOLUME == 0.0` e `ORIGINAL_VOLUME == 0.0`.
  * `transcriber.py`: retorno `(segments, detected_lang)`; `timeout=300` + erro UTF-8 em `load_audio_for_whisper()`; clamping em `_deduplicate_segments()`; `_ends_sentence()` + `MERGE_MAX_CHARS = 450` em `_merge_segments()`; remoção do alias morto `transcribe_with_ffmpeg`.
  * `translator.py`: adicionado `_split_text_chunks()`, `_translate_with_timeout()` (sem bloqueio no `shutdown`), `_to_mymemory_lang()` dinâmico com teto de `450` chars, detecção de avisos da API MyMemory, backoff com jitter e regex tolerante a espaços (`TECHTERM\s*NNN`) com `lambda _: original`.
  * `synthesizer.py`: `_has_speakable_text()` marcando trechos sem alfanuméricos como `tts_skipped`; `unlink(missing_ok=True)` em parciais corrompidos; backoff exponencial com jitter.
  * `video_processor.py`: fallback `libx264` em `replace_video_audio()` quando `-c:v copy` falha; `parse_srt()` tolerante a BOM UTF-8, `\r\n` e `.`/`,` nos milissegundos.
  * `utils.py`, `cli.py`, `config_ptbr.py`, `dublagem/config.py`, `run.py`, `dublar.py`: cache versionado por `(hash conteúdo, modelo, idioma)`, hash desacoplado do nome do arquivo, remoção de código morto (`download_file_with_progress`, `DETECCAO_AUTOMATICA`, `PRESETS`), validação de `ffprobe` + `onnxruntime` + deps em `config_ptbr.py`, UTF-8 garantido antes dos imports.
* **Repositório e Docs:** `.gitignore` blindado contra modelos/vídeos/áudios/caches/logs (mantendo `glossario.txt`, `GEMINI.md`, `CONTEXTO_PROJETO.md`, `README.md`, `requirements.txt`); documentação atualizada para `v3.2`; validações (`py_compile`, `config_ptbr.py`, `--help` e checagens funcionais unitárias) 100% aprovadas; código publicado no GitHub (`origin/main`, commit `129b6ba`).
