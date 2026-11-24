# 🎬 LexiVox — Sistema de Dublagem Automática 100% Offline

> Documento de contexto histórico e técnico do projeto. Nunca apague este arquivo.
> Última atualização: 23 de Novembro de 2026 (v4.0 — Arquitetura 100% Offline: NLLB-200 + Coqui XTTS-v2 + Anti-Loop + Fail-Fast)

---

## 📌 O Que É Este Projeto

Sistema local em Python para dublar automaticamente vídeos técnicos, cursos de programação e screencasts para **português brasileiro (PT-BR)**, sem depender de APIs gratuitas sujeitas a bloqueio ou cota.
O fluxo é: **vídeo original → transcrição (faster-whisper + Silero VAD + Anti-Loop) → tradução offline (NLLB-200 3.3B CTranslate2) → síntese neural PT-BR offline (Coqui XTTS-v2) → DSP (Smart Overflow + Modo Filme + Room Tone) → vídeo dublado**.

---

## 🏗️ Histórico de Evolução

### v1.0 — Versão Original
- `openai-whisper` + `googletrans` em bloco único sem timestamps granulares.

### v2.0 a v2.3 — Reescrita com Whisper embutido e DSP (Maio 2026)
- `whisper.cpp` via FFmpeg, legendas `.srt`, **Smart Overflow**, crossfades de 30ms, **Room Tone** e normalização anti-clipping.

### v3.0 a v3.2 — Modularização, `faster-whisper` e Auditoria (Maio–Novembro 2026)
- Pacote modular `dublagem/`, `faster-whisper large-v3` + Silero VAD + pipe FFmpeg, Modo Filme (`0%/0%`) padrão para screencasts, vetorização do Room Tone, cache versionado e `.gitignore` blindado no GitHub.

### v4.0 — Migração 100% Offline & Fail-Fast Rigoroso (Novembro 2026)
- **Por que migramos:** Durante testes em lote real (`dubbing.log` de 23/11/2026), serviços gratuitos online colapsaram simultaneamente:
  1. `edge-tts` recebeu **HTTP 403** do endpoint Microsoft Edge em 100% dos segmentos (`0/14`, `0/56`, `0/61`).
  2. `GoogleTranslator` (`429 Too Many Requests`) + `MyMemoryTranslator` esgotaram a cota no 2º vídeo.
  3. O fallback de emergência do `audio_processor.py` mascarava a falha gerando vídeos `_dubbed.mp4` 100% em inglês com `✅ Sucesso`.
  4. O `faster-whisper` entrou em loop repetindo a mesma frase 40+ vezes em pausas de código (`condition_on_previous_text=True`).
- **O que mudou na v4.0:**
  - **Ambiente virtual dedicado `.venv/` em Python 3.11:** instalado ao lado do Python 3.14 para suportar `PyTorch CPU` + `coqui-tts` (`run.py` e `dublar.py` fazem bootstrap automático para `.venv\Scripts\python.exe`).
  - **Tradução 100% Offline (`NLLB-200 3.3B CTranslate2`):** `translator.py` migrado para `OpenNMT/nllb-200-3.3B-ct2-int8` (`ctranslate2`, CPU `int8`, 10 threads) com tokenizer local `facebook/nllb-200-distilled-600M` e preservação de glossário técnico. Zero rate-limit, zero cota.
  - **Voz Neural PT-BR 100% Offline (`Coqui XTTS-v2`):** `synthesizer.py` migrado para `tts_models/multilingual/multi-dataset/xtts_v2` em CPU (`10` threads PyTorch), com speakers PT-BR (`Marcos Rudaski`, `Damien Black`, `Claribel Dervla`, `Ana Florence`) e clonagem de voz zero-shot (`--voice clone` / `--speaker-wav`).
  - **Transcrição Anti-Loop:** `condition_on_previous_text=False`, `repetition_penalty=1.1`, `no_repeat_ngram_size=3`, `hallucination_silence_threshold=2.0`, `compression_ratio_threshold=2.4`, `no_speech_threshold=0.6` e filtro `_remove_repetition_loops`.
  - **Fail-Fast Rigoroso (`MAX_FAILURE_RATIO = 0.10`):** aborta o vídeo sem gerar `_dubbed.mp4` se >10% dos segmentos falharem na tradução ou TTS, ou se `tts_placed == 0` no DSP (mantendo no `.cache/` os `.wav` que já deram certo).

---

## 🖥️ Hardware e Ambiente Validado

| Componente | Especificação |
|---|---|
| CPU | Intel Core i5-13400F (10 cores, 16 threads, AVX2 — 10 threads configuradas) |
| RAM | 32 GB |
| GPU | AMD Radeon RX 6600M (não usada por PyTorch/CTranslate2 no Windows; pipeline roda 100% em CPU) |
| OS | Windows 11 (ReviOS) — Console configurado com UTF-8 |
| Python | `3.11.9` (64-bit) na `.venv/` (com bootstrap automático em `run.py` e `dublar.py`) |
| FFmpeg / FFprobe | `yt-dlp.FFmpeg` via Winget (`N-128251`+, sem `--enable-whisper`) |

### 🚨 Regras de Ouro do Projeto
* **Qualidade e Confiabilidade Máximas > Velocidade:** PC roda **24/7**. Padrão obrigatório: Whisper `large-v3` + tradutor `NLLB-200 3.3B` + voz neural `Coqui XTTS-v2`.
* **100% Offline após 1º download:** Nenhuma dependência de scrapers gratuitos sem SLA (`edge-tts`, `deep-translator` removidos).
* **Experiência Filme Dublado (`0% / 0%`):** Voz original totalmente substituída por padrão (+ Room Tone + Smart Overflow).
* **Fail-Fast Real:** Nunca mascarar falhas gerando vídeo final em idioma original.

---

## 📂 Stack Tecnológica Atual (v4.0)

| Componente | Tecnologia | Local dos Modelos |
|---|---|---|
| Transcrição | `faster-whisper` (`large-v3` `int8`) + `Silero VAD` (`onnxruntime`) | `models/faster-whisper/` |
| Tradução | `CTranslate2` (`OpenNMT/nllb-200-3.3B-ct2-int8`) + `transformers` | `models/nllb-200/` |
| Síntese de Voz | `Coqui XTTS-v2` (`torch` CPU + `coqui-tts`) | `models/coqui-tts/` |
| DSP de Áudio | `NumPy` + `SoundFile` + `FFmpeg atempo` | — |
| Vídeo & Legendas | `FFmpeg` / `FFprobe` (`-c:v copy` + fallback `libx264`) | — |

---

## 🚀 Backlog Futuro

1. **Separação Nativa com Demucs (`htdemucs`):** Já compatível com o `torch` instalado na `.venv` — integração direta para vídeos com música de fundo contínua.
2. **Diarização de Falantes (`pyannote-audio`):** Alternar speakers/clonagens diferentes em vídeos com múltiplos participantes.
3. **Automação CLI:** Download via `yt-dlp` (`--url`), `--preview`, `--burn-srt`.
4. **Interface Web (Gradio/Streamlit):** Painel local para envio de vídeos, seleção de voz/clonagem e edição visual de `.srt`.
