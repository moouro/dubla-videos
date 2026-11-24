# 🎬 LexiVox — Sistema de Dublagem Automática 100% Offline (v4.0)

**Dubla vídeos, cursos e screencasts de tecnologia de qualquer idioma para português brasileiro (PT-BR) 100% offline com qualidade neural máxima, zero dependência de APIs gratuitas instáveis e sincronização precisa estilo filme dublado.**

[![Python](https://img.shields.io/badge/python-3.11_(.venv)-blue)]()
[![Status](https://img.shields.io/badge/status-v4.0_100%25_offline-brightgreen)]()
[![Whisper](https://img.shields.io/badge/faster--whisper-Large--v3-purple)]()
[![Translator](https://img.shields.io/badge/NLLB--200-3.3B_CTranslate2-orange)]()
[![TTS](https://img.shields.io/badge/Coqui_XTTS--v2-Neural_PT--BR-cyan)]()

---

## ⚡ Início Rápido

### Pré-requisitos
* **Python 3.11** (recomendado via `.venv` na raiz para compatibilidade total com `PyTorch` e `coqui-tts`).
* **FFmpeg e FFprobe** instalados e acessíveis no `PATH` (ou via Winget):
  ```powershell
  winget install --id=yt-dlp.FFmpeg -e
  ```

### Instalação e Execução
```powershell
# 1. Criar ambiente virtual Python 3.11 (se ainda não existir)
py -3.11 -m venv .venv

# 2. Instalar PyTorch CPU + dependências v4.0
.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Verificar ambiente (FFmpeg, FFprobe, faster-whisper, NLLB-200, XTTS-v2)
.venv\Scripts\python.exe config_ptbr.py

# 4. Colocar seus vídeos na pasta input_videos/

# 5. Dublar! (run.py e dublar.py detectam a .venv automaticamente)
python run.py
# (ou python dublar.py)
```

> Na primeira execução, os modelos locais (`faster-whisper large-v3`, `OpenNMT/nllb-200-3.3B-ct2-int8` e `Coqui XTTS-v2`) são baixados automaticamente para `models/`. Depois disso, todo o pipeline opera **100% offline**, sem limites diários ou bloqueios HTTP 403/429.

Os vídeos dublados (`*_dubbed.mp4`) e as legendas sincronizadas (`*_dubbed.srt`) serão salvos em `output_videos/`.

---

## 🚀 Principais Recursos da v4.0

* **`faster-whisper` Large-v3 + Silero VAD + Travas Anti-Loop:**
  * Modelo **Large-v3** em `models/faster-whisper/` para máxima fidelidade em termos técnicos de programação.
  * Travas contra loops de alucinação em vídeos longos: `condition_on_previous_text=False`, `repetition_penalty=1.1`, `no_repeat_ngram_size=3`, `hallucination_silence_threshold=2.0` e filtro consecutivo `_remove_repetition_loops`.
  * **Merge Inteligente de Segmentos:** consolida frases por pausa (`<= 600ms`), duração (`<= 14s`), limite de caracteres (`350`) e pontuação real.
* **Tradução 100% Offline com `NLLB-200` 3.3B (`CTranslate2`):**
  * Substitui scrapers online (Google/MyMemory) pelo modelo local `OpenNMT/nllb-200-3.3B-ct2-int8` rodando na CPU via `CTranslate2`.
  * Preserva automaticamente mais de 120 termos técnicos de programação (`config_ptbr.py` + `glossario.txt`).
* **Voz Neural PT-BR 100% Offline com `Coqui XTTS-v2`:**
  * Substitui o Edge TTS (sujeito a `HTTP 403`) pelo modelo neural local **Coqui XTTS-v2**, com speakers nativos de alta naturalidade e suporte a **clonagem zero-shot da voz do instrutor** (`--voice clone` ou `--speaker-wav`).
* **Política Fail-Fast Rigorosa (`MAX_FAILURE_RATIO = 0.10`):**
  * Nunca gera um `*_dubbed.mp4` falsamente "dublado" com áudio em inglês: se mais de **10%** dos segmentos falharem na tradução ou síntese TTS (ou se `0` segmentos TTS forem montados), o vídeo é abortado preservando o cache do que já deu certo.
* **Experiência de Filme Dublado (Modo Filme 0%/0%):**
  * Substituição total da voz original (`DUCKING_VOLUME = 0.0`, `ORIGINAL_VOLUME = 0.0`) + **Room Tone** vetorizado + **Smart Overflow** (`85% / 1500ms`).

---

## 💡 Exemplos de Uso

### 1. Dublar todos os vídeos da pasta (Modo Filme — Padrão)
```powershell
python run.py
```

### 2. Dublar um vídeo específico e forçar reprocessamento
```powershell
python run.py -i "input_videos\aula01.mp4" --force
```

### 3. Escolher outra voz PT-BR ou clonar a voz original
```powershell
# Voz feminina clara
python run.py --voice claribel

# Clonagem automática usando ~10s limpos da voz do instrutor do próprio vídeo
python run.py --voice clone

# Clonagem usando um arquivo WAV de referência customizado (6–12s)
python run.py --speaker-wav "minha_voz_ref.wav"
```

### 4. Ativar Voice-Over (manter áudio original ao fundo)
```powershell
# Preset Voice-Over TI (25% nos gaps / 6% durante a fala)
python run.py --keep-audio

# Voice-Over genérico (50% nos gaps / 15% durante a fala)
python run.py --keep-audio --not-tech
```

### 5. Modo Semi-Automático (Revisar Legendas Antes de Narrar)
```powershell
# Passo 1: Gera apenas as legendas traduzidas para revisão
python run.py --srt-only

# Passo 2: Revise o arquivo .srt gerado em output_videos/

# Passo 3: Dubla usando o SRT revisado (requer -i apontando para o vídeo)
python run.py -i "input_videos\aula01.mp4" --from-srt "output_videos\aula01_dubbed.srt"
```

---

## 🎭 Vozes Disponíveis (Coqui XTTS-v2)

| Alias (`--voice`) | Speaker XTTS-v2 | Perfil |
|---|---|---|
| `marcos` ⭐ | `Marcos Rudaski` | Masculino — natural e claro (padrão) |
| `damien` | `Damien Black` | Masculino — tom mais grave |
| `claribel` | `Claribel Dervla` | Feminino — didática clara |
| `ana` | `Ana Florence` | Feminino — tom suave |
| `clone` | `auto_clone` | Clona automaticamente a voz do próprio vídeo |

---

## 📚 Glossário Técnico Personalizado

Crie um arquivo `glossario.txt` na raiz do projeto com um termo por linha para protegê-los contra tradução junto com os +120 termos já presentes em `config_ptbr.py`:

```text
# glossario.txt
LangChain
LlamaIndex
FastAPI
ChromaDB
```

---

## 🔧 Diagnóstico do Ambiente

```powershell
.venv\Scripts\python.exe config_ptbr.py
```
Valida `ffmpeg`, `ffprobe`, `faster-whisper`, `onnxruntime`, `ctranslate2`, `transformers`, `sentencepiece`, `huggingface_hub`, `torch`, `TTS`, `soundfile`, `numpy` e `tqdm`.
