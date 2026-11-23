# 🎬 LexiVox — Sistema de Dublagem Automática de Alta Precisão

> Documento de contexto histórico e técnico do projeto. Nunca apague este arquivo.
> Última atualização: 23 de Novembro de 2026 (v3.2 — Auditoria Completa, Modo Filme & Resiliência)

---

## 📌 O Que É Este Projeto

Sistema para dublar automaticamente vídeos de qualquer idioma para **português brasileiro (PT-BR)**.
O fluxo é: **vídeo original → transcrição (VAD + word timestamps + merge) → tradução (glossário + fallback) → síntese neural (Edge TTS) → DSP (Smart Overflow + Modo Filme + Room Tone) → vídeo dublado**.

O objetivo é pessoal: dublar cursos, tutoriais de programação e screencasts (em sua maioria sem câmera facial, apenas voz sobre a tela) com experiência limpa equivalente a **filme dublado** (voz original substituída por completo).

---

## 🏗️ Histórico de Evolução

### v1.0 — Versão Original
- Usava `openai-whisper` (PyTorch ~2 GB) e `googletrans` (instável).
- Processava o texto inteiro sem timestamps por segmento e sem usar `config_ptbr.py`.

### v2.0 a v2.3 — Reescrita com Whisper embutido e DSP (Maio 2026)
- Migração temporária para FFmpeg + `whisper.cpp`, substituição por `deep-translator` e `edge-tts`.
- Introdução de legendas `.srt`, **Smart Overflow**, crossfades de 30ms, extração de **Room Tone** e normalização final anti-clipping.

### v3.0 — Refatoração Modular (Maio 2026)
- Divisão do monólito em pacote Python estruturado (`dublagem/`: `config`, `utils`, `transcriber`, `translator`, `synthesizer`, `audio_processor`, `video_processor`, `cli`).
- Adoção do **Whisper Large-v3** (~3 GB) como padrão para precisão cirúrgica em termos de programação.
- Cache inteligente em `.cache/<hash_md5>/` e paralelização assíncrona no TTS e ajustes de velocidade.

### v3.1 — Migração para `faster-whisper` & Modo Filme para Screencasts (Outubro 2026)
- **Desacoplamento do filtro whisper do FFmpeg:** nova máquina com Windows 11 limpo e FFmpeg moderno via Winget (`yt-dlp.FFmpeg` sem `--enable-whisper`). Migração para `faster-whisper` (CTranslate2 `int8`, 10 threads CPU) + **Silero VAD** (`onnxruntime`) + decodificação PCM via pipe FFmpeg (contorna bug de `metadata_errors` no `av >= 19.0`).
- **Fallback MyMemory no Tradutor:** contorna rate-limits HTTP 429 do `GoogleTranslator`.
- **Cache TTS com Hash do Texto:** evita reutilizar áudio antigo após edição de texto/SRT.
- **Correção Crítica — Modo Filme como Padrão (Outubro 2026):** alterado de Voice-Over (`6%`/`25%`) para **Modo Filme (`0.0`/`0.0`)** mesmo quando `is_tech_course=True`, removendo totalmente a voz inglesa do fundo conforme preferência do usuário.

### v3.2 — Auditoria Técnica, Resiliência & Preparação GitHub (Novembro 2026)
- **Correção de Bugs Silenciosos no DSP e Transcrição:**
  - Eliminação de desalinhamento de índices entre ajuste paralelo e montagem em `audio_processor.py` usando dicionário `adjusted_map`.
  - Troca de sobrescrita destrutiva (`=`) por soma mixada (`+=`) na inserção de segmentos com overflow.
  - Flags `--keep-audio` e `--separate-audio` passam a funcionar corretamente mesmo no modo padrão de cursos TI.
  - Vetorização completa de `_fill_silent_gaps` com NumPy e remoção de `FFMPEG_SEMAPHORE` em escopo de módulo.
  - Correção de `_deduplicate_segments` (clamping em pequenas sobreposições sem descartar fala legítima) e refinamento do merge com `MERGE_MAX_CHARS = 450` e proteção a abreviações/extensões (`e.g.`, `v3.`, `.js.`).
- **Resiliência de Tradução, TTS e Vídeo:**
  - `transcribe_audio` retorna `(segments, detected_lang)`, propagado ao cache e ao fallback `MyMemoryTranslator` com código regional dinâmico.
  - Chunking respeitando `MAX_TEXT_CHUNK_SIZE = 800` (Google) e `<= 450` chars (MyMemory), timeout real via executor sem trava no `shutdown`, backoff com jitter e proteção de regex (`lambda _: original`) no `_restore_tech_terms`.
  - Edge TTS ignora trechos sem caracteres pronunciáveis (`tts_skipped`), limpa `.mp3` parciais/corrompidos e aplica backoff exponencial com jitter.
  - `parse_srt` com suporte a BOM UTF-8 e `\r\n`; `replace_video_audio` com fallback automático `libx264` caso `-c:v copy` falhe em containers como `.webm`/`.avi`; chaves de cache incluindo modelo e idioma destino.
- **Limpeza & `.gitignore`:**
  - Remoção de código morto (`download_file_with_progress`, `transcribe_with_ffmpeg`, `DETECCAO_AUTOMATICA`, `PRESETS`).
  - `.gitignore` blindado contra envio acidental de `models/`, `.cache/`, `input_videos/`, `output_videos/`, binários e logs ao GitHub.

---

## 🖥️ Hardware e Ambiente Validado

| Componente | Especificação |
|---|---|
| CPU | Intel Core i5-13400F (10 cores, 16 threads, AVX2 — 10 threads configuradas) |
| RAM | 32 GB |
| GPU | AMD Radeon RX 6600M (não usada pelo `faster-whisper`, que roda em CPU `int8`) |
| OS | Windows 11 (ReviOS) — Console configurado com UTF-8 |
| Python | 3.14.10 (64-bit) |
| FFmpeg / FFprobe | `yt-dlp.FFmpeg` via Winget (`N-128251`+, sem `--enable-whisper`) |
| Internet | 24/7 (necessária para tradução online e Edge TTS) |

### 🚨 Regra de Ouro do Projeto (Prioridades do Usuário)
* **Qualidade e Precisão Absolutas:** Whisper **Large-v3** é o padrão obrigatório para precisão máxima em termos técnicos de programação.
* **Tempo Não Importa:** O PC opera **24/7**; prioridade é qualidade acústica e técnica da dublagem, não velocidade de transcrição.
* **Experiência Filme Dublado:** Vídeos em sua maioria sem rosto/câmera do instrutor; a voz original deve ser **totalmente substituída** (0% original + Room Tone + Smart Overflow). Voice-Over só deve ocorrer se o usuário passar `--keep-audio`.

---

## 📂 Stack Tecnológica Atual (v3.2)

| Componente | Tecnologia | Observação |
|---|---|---|
| Transcrição | `faster-whisper 1.2.x` (`CTranslate2 int8`) + `Silero VAD` (`onnxruntime`) | Decodificação de áudio via FFmpeg PCM pipe (contorna bug do PyAV 19.x) |
| Tradução | `deep-translator` (`GoogleTranslator` + fallback `MyMemoryTranslator`) | Proteção de glossário + chunking + timeout + backoff |
| Síntese de voz | `edge-tts` (`Microsoft Edge Neural`, PT-BR) | Assíncrono (concorrência 5), semáforo e jitter |
| DSP de Áudio | `NumPy` + `SoundFile` + `FFmpeg atempo` | Smart Overflow, crossfade 30ms, Room Tone vetorizado |
| Vídeo & Legendas | `FFmpeg` / `FFprobe` | `-c:v copy` instantâneo com fallback `libx264` + export/parse `.srt` |

---

## 📊 Performance Típica (vídeo screencast ~2 min no i5-13400F)

| Etapa | Tempo Estimado |
|---|---|
| Extração WAV | < 1s |
| Transcrição (`large-v3` CPU `int8` + VAD) | ~1 min 30s – 2 min 30s |
| Tradução (8–15 segmentos merged) | ~15s – 25s |
| Síntese Edge TTS (paralelo, 5 workers) | ~3s – 5s |
| Ajuste de velocidade + DSP + Mux MP4 | ~2s – 3s |
| **Total típico** | **~2 min – 3 min** (instantâneo em re-execuções com cache) |

---

## 🚀 Status das Melhorias e Backlog Futuro

### ✅ Já Implementadas no Código
- [x] **Merge Inteligente de Segmentos** (`_merge_segments` em `transcriber.py`, com limites de gap, duração, caracteres e pontuação real)
- [x] **Modo Semi-Automático** (`--srt-only` e `--from-srt`)
- [x] **Paralelização do TTS e Speed Adjust** (asyncio + semáforos)
- [x] **Cache Inteligente por Hash, Modelo, Idioma e Texto** (`.cache/`)
- [x] **Separação Vocal/Instrumental Opcional via ONNX** (`--separate-audio` com `audio-separator`, instalação opcional)

### 📋 Backlog Futuro (Não Implementado)
1. **Separação Nativa com Demucs (`htdemucs`):** Alternativa integrada e out-of-the-box para vídeos com música de fundo pesada, sem depender de configuração manual do `audio-separator`.
2. **Diarização de Falantes (`pyannote-audio` / múltiplos locutores):** Detectar diferentes vozes em entrevistas/podcasts e alternar automaticamente entre `AntonioNeural` / `FranciscaNeural`.
3. **Download Direto via `yt-dlp` (`--url`):** Baixar vídeo do YouTube/plataformas direto para `input_videos/` antes do pipeline.
4. **Queima de Legendas (`--burn-srt`):** Opcional para gerar versão com legenda embutida (re-encoding H.264/AMF).
5. **Preview Rápido (`--preview 30`):** Dublar apenas os primeiros segundos para testar voz/ritmo antes de processar aulas longas.
6. **Interface Web (Gradio/Streamlit):** Painel visual local para envio de vídeos e edição de `.srt`.
7. **Ajuste de Emoção/Entonação via SSML:** Controle fino de estilo/prosódia nas vozes neurais compatíveis.

---

## 📝 Notas Operacionais Importantes

- Os vídeos em `input_videos/` **nunca são modificados**.
- Se o vídeo `*_dubbed.mp4` já existir em `output_videos/`, ele é pulado automaticamente — use `--force` (`-f`) para reprocessar.
- Os modelos `faster-whisper` ficam em `models/faster-whisper/` (~3 GB para `large-v3`); o arquivo legado `models/ggml-large-v3.bin` não é mais utilizado pelo código e pode ser apagado manualmente se quiser liberar 3 GB no disco.
- Logs completos ficam em `output_videos/dubbing.log`.
- `models/`, `.cache/`, `input_videos/`, `output_videos/` e logs estão protegidos pelo `.gitignore`.
