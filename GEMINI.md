# GEMINI.md — Memória Técnica Permanente (LexiVox / Dubla-videos)

> Leia este arquivo no início de qualquer nova sessão neste repositório.
> Última atualização: 24 de Novembro de 2026 (v4.0 — Arquitetura 100% Offline Validada Ponta-a-Ponta).

---

## 1. Visão Geral e Stack

* **Propósito:** Pipeline local 100% offline em Python para dublar vídeos técnicos, cursos de programação e screencasts (em sua maioria sem câmera facial) de inglês (e outros idiomas) para português brasileiro (`pt-BR`) com experiência de **filme dublado**.
* **Ambiente Validado:**
  * **SO:** Windows 11 (ReviOS), console UTF-8.
  * **Hardware:** Intel Core i5-13400F (10 cores / 16 threads, AVX2), 32 GB RAM, AMD Radeon RX 6600M (não usada; toda inferência roda em CPU `10 threads`). Máquina operada **24/7**.
  * **Python:** Ambiente virtual dedicado `.venv/` em **Python `3.11.9` (64-bit)** para compatibilidade com PyTorch + `coqui-tts` (`run.py` e `dublar.py` fazem bootstrap automático para `.venv\Scripts\python.exe`).
  * **FFmpeg / FFprobe:** Pacote Winget `yt-dlp.FFmpeg` (`N-128251`).
* **Dependências (`requirements.txt`):**
  * Transcrição: `faster-whisper>=1.2.0` + `onnxruntime>=1.18.0` (Silero VAD) + `ctranslate2>=4.5.0`.
  * Tradução Offline: `ctranslate2` com `OpenNMT/nllb-200-3.3B-ct2-int8` + `transformers>=4.57.0,<5.0.0` (`4.57.6`) / `sentencepiece` / `huggingface-hub` (`facebook/nllb-200-distilled-600M` tokenizer local).
  * Síntese TTS Offline: `torch` (`2.15.0+cpu`) + `torchaudio` (`2.11.1+cpu`) + `coqui-tts[codec]>=0.24.0` (`0.28.0` com `torchcodec 0.17.0`) + `six>=1.16.0` (`tts_models/multilingual/multi-dataset/xtts_v2`).
  * Áudio/DSP e CLI: `soundfile>=0.12.0`, `numpy>=1.26.0`, `tqdm>=4.66.0`.
  * Opcional fora do padrão: `audio-separator` (somente com `--separate-audio`).
* **Fluxo do Pipeline (7 etapas):**
  1. Extração WAV mono `44100 Hz` (`video_processor.extract_original_audio`).
  2. Separação vocal/instrumental ONNX opcional (`audio_processor.separate_audio`).
  3. Transcrição `faster-whisper large-v3` via pipe PCM `16 kHz` + Silero VAD + travas anti-alucinação + `_deduplicate_segments` (clamping) + `_remove_repetition_loops` + `_merge_segments` (`transcriber.transcribe_audio`).
  4. Tradução 100% offline com `NLLB-200 3.3B` (`ctranslate2`), proteção de termos técnicos (`config_ptbr.py` + `glossario.txt`, preservando siglas maiúsculas `<= 6` letras diretamente e usando marcadores `[0]`, `[1]`), chunking (`450` chars) e Fail-Fast (`>10%`) (`translator.translate_segments`) + `.srt` (`video_processor.save_srt`).
  5. Síntese sequencial offline em CPU via **Coqui XTTS-v2** (`marcos`, `damien`, `claribel`, `ana`, `clone` ou `--speaker-wav`) com `_prepare_xtts_text(max_chars=185)`, cache por `(speaker, rate, hash MD5 do texto)`, validação `.wav` e Fail-Fast (`>10%`) (`synthesizer.synthesize_all_segments`).
  6. DSP assíncrono: `ffprobe` + `ffmpeg atempo`, Smart Overflow, soma `+=` com crossfade `30ms`, Modo Filme (`0%/0%`), Room Tone vetorizado, verificação `tts_placed > 0` e normalização `0.95` (`audio_processor.assemble_dubbed_audio`).
  7. Remux MP4 `-c:v copy` + AAC `192k` com fallback `libx264` (`video_processor.replace_video_audio`).

---

## 2. Decisões Técnicas e Ajustes de Qualidade

* **100% Offline, Qualidade Máxima > Velocidade:**
  * Abandonamos `edge-tts` (bloqueios HTTP 403) e `deep-translator` Google/MyMemory (rate-limit 429 / cota diária).
  * Padrões fixos: `WHISPER_MODEL = "large-v3"`, `TRANSLATOR_MODEL = "OpenNMT/nllb-200-3.3B-ct2-int8"`, `TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"`, `VOZ_PADRAO = "Marcos Rudaski"`.
* **Travas Anti-Alucinação no Whisper (`transcriber.py`):**
  * `condition_on_previous_text=False`, `repetition_penalty=1.1`, `no_repeat_ngram_size=3`, `hallucination_silence_threshold=2.0`, `compression_ratio_threshold=2.4`, `no_speech_threshold=0.6`, além de `_remove_repetition_loops()` (descarta repetições consecutivas `>= 3x`).
* **Fail-Fast Rigoroso (`MAX_FAILURE_RATIO = 0.10`):**
  * Se `>10%` dos segmentos falharem na tradução ou no XTTS-v2, ou se `tts_placed == 0` no DSP, o vídeo é **abortado sem gerar `*_dubbed.mp4`** (preservando os `.wav` válidos em `.cache/`).
* **Modo Filme Obrigatório por Padrão (`0% / 0%`):**
  * `DUCKING_VOLUME = 0.0` e `ORIGINAL_VOLUME = 0.0`; Voice-Over (`6%/25%` TI ou `15%/50%`) só via `--keep-audio`.
* **Parâmetros de Sincronia e Merge:**
  * `MIN_SPEED_FACTOR = 0.7`, `MAX_SPEED_FACTOR = 1.15`, `OVERFLOW_GAP_RATIO = 0.85`, `OVERFLOW_MAX_MS = 1500`.
  * `MERGE_MAX_GAP_MS = 600`, `MERGE_MAX_DURATION_MS = 14000`, `MERGE_MAX_CHARS = 350`.
* **Armadilhas Resolvidas (Não Regredir):**
  1. PyAV `>=19.0` (`metadata_errors="ignore"`) → áudio decodificado via pipe FFmpeg em `load_audio_for_whisper()`.
  2. Coqui XTTS-v2 (`0.28.0`) no PyTorch `>=2.9` → exige `coqui-tts[codec]` (`torchcodec`), `six>=1.16.0`, pin `transformers>=4.57.0,<5.0.0` (na `5.x`, `past_key_values` chega `None` em `gpt_inference.py:62` causando `AttributeError: 'NoneType' object has no attribute 'get_seq_length'`), além de `COQUI_TOS_AGREED=1`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` e `TTS_HOME=models/coqui-tts`.
  3. Tokenizer NLLB-200 salvo pela `transformers 5.x` gera `extra_special_tokens` como `list`, falhando em `4.57.x` (`'list' object has no attribute 'keys'`) → manter `models/nllb-200/tokenizer/` gerado por `4.57.x`.
  4. XTTS-v2 limite de `203` chars em `'pt'` → `_prepare_xtts_text(max_chars=185)` quebra em vírgulas/cláusulas sem truncar áudio.
  5. Placeholders no NLLB-200 → siglas puras `<= 6` letras (`CSS`, `HTML`, `API`, etc.) passam sem substituição e termos compostos usam `[0]`, `[1]`.
  6. Bootstrapping automático da `.venv` em `run.py` e `dublar.py` com guarda `LEXIVOX_VENV_BOOTSTRAPPED=1` e comparação de `sys.prefix`.
  7. `asyncio.Semaphore` criado sempre dentro das corrotinas; mapeamento DSP por `adjusted_map` e soma `+=`.
  8. Console Windows UTF-8 + Git `-c safe.directory=C:/Users/Administrador/Desktop/Dubla-videos`.

---

## 3. Estado Atual (v4.0 — Arquitetura Offline Validada)

- [x] Python 3.11.9 instalado e `.venv/` configurada com PyTorch CPU (`2.15.0+cpu`) + `coqui-tts[codec]` (`0.28.0`) + `six` + `transformers 4.57.6` + `faster-whisper` + `ctranslate2`.
- [x] Pipeline reescrito para v4.0 100% offline (`transcriber.py` anti-loop, `translator.py` NLLB-200 CTranslate2, `synthesizer.py` Coqui XTTS-v2 com suporte a clonagem, `cli.py` e `audio_processor.py` com Fail-Fast `>10%`).
- [x] Modelos locais (`faster-whisper large-v3`, `OpenNMT/nllb-200-3.3B-ct2-int8`, `facebook/nllb-200-distilled-600M` tokenizer e `Coqui XTTS-v2`) baixados em `models/` e validados 100% offline em `0001_01_What_Is_CSS.mp4` (`17/17` segmentos, 0 falhas, 0 warnings, saída `19.5 MB` em Modo Filme).
- [ ] Reprocessar os demais cursos em `input_videos/` (`0002_02_CSS_Basic_Syntax.mp4`, `0020_20_Floats.mp4`) com `--force` e commitar as mudanças da v4.0 no GitHub.

---

## 4. Próximos Passos Planejados

1. **Separação Nativa com Demucs (`htdemucs`):** Aproveitar o `torch` já presente na `.venv/` para vídeos com música contínua.
2. **Diarização de Falantes (`pyannote-audio`):** Alternar vozes XTTS-v2 em diálogos/entrevistas.
3. **Automação CLI (`--url`, `--preview`, `--burn-srt`) e Interface Web (Gradio/Streamlit).**
4. *(Manutenção local opcional)* Excluir `models/ggml-large-v3.bin` (~3 GB legado da v2).

---

## 5. Manutenção de Memória do Projeto

* Sempre que o usuário pedir para **"salvar progresso"**, **"encerrar sessão"** ou **"atualizar memória"**, atualize o arquivo `GEMINI.md` na raiz do projeto marcando as tarefas concluídas em **"Estado Atual"**, ajustando os **"Próximos Passos"** e registrando a data e o resumo técnico das mudanças no final do arquivo.

---

## 6. Resumo Técnico da Última Sessão (`2026-11-23/24` — Migração e Validação v4.0 Offline)

* **Diagnóstico do `dubbing.log`:** `edge-tts` falhou em 100% dos segmentos com `HTTP 403`, `GoogleTranslator`/`MyMemory` atingiram limite de cota no 2º vídeo, fallback de áudio injetou inglês mascarando erro como sucesso, e `faster-whisper` entrou em loop repetitivo no Vídeo 3.
* **Implementação v4.0:** Criada `.venv/` Python 3.11 com bootstrap automático em `run.py`/`dublar.py`; migrada tradução para `OpenNMT/nllb-200-3.3B-ct2-int8` (`ctranslate2` offline); migrado TTS para `Coqui XTTS-v2` (speakers nativos + `--voice clone` / `--speaker-wav`); adicionadas travas anti-alucinação + `_remove_repetition_loops()` no `transcriber.py`; implementado Fail-Fast rigoroso (`>10%` de falhas aborta antes de gerar MP4).
* **Ajustes pós-teste e validação real:** Instalados `coqui-tts[codec]` (`torchcodec`) e `six>=1.16.0`; fixado `transformers>=4.57.0,<5.0.0` (`4.57.6`) para corrigir `get_seq_length` no `GPT2InferenceModel` do XTTS-v2; siglas maiúsculas `<= 6` letras (`CSS`, `HTML`, etc.) preservadas diretamente no NLLB e demais termos via `[N]`; adicionado `_prepare_xtts_text(max_chars=185)` evitando truncamentos em `pt`; pipeline ponta-a-ponta validado 100% offline em `0001_01_What_Is_CSS.mp4` (`17/17` segmentos, 0 falhas, 0 warnings, `0001_01_What_Is_CSS_dubbed.mp4` 19.5 MB em Modo Filme).
