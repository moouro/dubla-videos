# 🎬 LexiVox — Sistema de Dublagem Automática de Alta Fidelidade (v3.2)

**Dubla vídeos, cursos e screencasts de tecnologia de qualquer idioma para português brasileiro (PT-BR) com máxima fidelidade acústica, inteligência artificial neural e sincronização precisa estilo filme dublado.**

[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![Status](https://img.shields.io/badge/status-v3.2_est%C3%A1vel-brightgreen)]()
[![Whisper](https://img.shields.io/badge/faster--whisper-Large--v3-purple)]()
[![TTS](https://img.shields.io/badge/Edge--TTS-Neural-cyan)]()

---

## ⚡ Início Rápido

### Pré-requisitos
* **Python 3.11+** (validado no Windows 11 com Python 3.14)
* **FFmpeg e FFprobe** instalados e acessíveis no `PATH` (ou via Winget):
  ```powershell
  winget install --id=yt-dlp.FFmpeg -e
  ```

### Instalação e Execução
```powershell
# 1. Instalar dependências Python
pip install -r requirements.txt

# 2. Verificar ambiente (FFmpeg, FFprobe, faster-whisper, onnxruntime, etc.)
python config_ptbr.py

# 3. Colocar seus vídeos na pasta input_videos/

# 4. Dublar!
python run.py
# (ou python dublar.py)
```

Os vídeos dublados (`*_dubbed.mp4`) e as legendas sincronizadas (`*_dubbed.srt`) serão salvos em `output_videos/`.

---

## 🚀 Principais Recursos da v3.2

* **`faster-whisper` com Large-v3 & Silero VAD:**
  * O modelo **Large-v3** é baixado e armazenado automaticamente em `models/faster-whisper/`.
  * **Silero VAD integrado:** remove silêncios e ruídos antes da transcrição, reduzindo alucinações em pausas longas.
  * **Word-Level Timestamps + Clamping Seguro:** alinhamento por palavra sem descarte acidental de falas com pequena sobreposição de fronteira.
  * **Merge Inteligente de Segmentos:** agrupa segmentos contínuos da mesma frase por pausa (`<= 600ms`), duração (`<= 14s`), teto de caracteres (`450`) e pontuação real (sem quebrar abreviações/extensões como `e.g.` ou `v3.`).
* **Experiência de Filme Dublado (Modo Filme 0%/0%):**
  * **Substituição Total:** por padrão (`DUCKING_VOLUME = 0.0`, `ORIGINAL_VOLUME = 0.0`), a voz original em inglês é **100% removida** — apenas o narrador em português é ouvido, como um filme dublado ou a dublagem automática do YouTube.
  * **Room Tone Contínuo (Vetorizado):** preenche pausas silenciosas com o ruído ambiente natural extraído do próprio vídeo, evitando cortes digitais artificiais.
  * **Smart Overflow:** aproveita até `85%` (máximo `1500ms`) do intervalo até a próxima fala, permitindo dicção natural (`+0%`) sem aceleração excessiva e sem sobrescrever o trecho seguinte (`+=`).
* **Tradução e Síntese Resilientes:**
  * **Preservação de Termos Técnicos:** protege mais de 120 termos de programação (`config_ptbr.py` + `glossario.txt`) via placeholders antes de traduzir.
  * **Fallback Automático:** alterna automaticamente para `MyMemoryTranslator` com *chunking* (`<= 450` caracteres) caso o Google atinja rate-limit (429), usando o idioma real detectado pelo Whisper.
  * **Edge TTS Concorrente:** pool assíncrono com semáforo, backoff exponencial com jitter e limpeza automática de arquivos `.mp3` parciais/corrompidos.
* **Sistema de Cache Inteligente:**
  * Baseado no hash MD5 do conteúdo do vídeo (independente do nome do arquivo), do modelo Whisper e do idioma destino em `.cache/`.
  * Cache de áudio TTS validado por voz, velocidade e hash MD5 do texto traduzido.
* **Ordenação Alfanumérica Natural:**
  * Processa aulas em sequência humana (`1.-...`, `2.-...`, ..., `10.-...`).

---

## 💡 Exemplos de Uso

### 1. Dublar todos os vídeos da pasta (Modo Filme — Padrão)
```powershell
python run.py
```

### 2. Dublar um vídeo específico
```powershell
python run.py -i "input_videos\aula01.mp4"
```

### 3. Forçar reprocessamento de um vídeo já dublado
```powershell
python run.py -i "input_videos\aula01.mp4" --force
```

### 4. Usar voz feminina profissional
```powershell
python run.py --voice pt-BR-FranciscaNeural
```

### 5. Ativar Voice-Over (manter áudio original ao fundo)
```powershell
# Preset Voice-Over TI (25% nos gaps / 6% durante a fala)
python run.py --keep-audio

# Voice-Over genérico (50% nos gaps / 15% durante a fala)
python run.py --keep-audio --not-tech
```

### 6. Preservar música/efeitos separando vocais via ONNX (opcional)
Requer instalar `audio-separator` separadamente:
```powershell
python run.py --separate-audio
```

### 7. Modo Semi-Automático (Revisar Legendas Antes de Narrar)
```powershell
# Passo 1: Gera apenas as legendas traduzidas para revisão
python run.py --srt-only

# Passo 2: Revise o arquivo .srt gerado em output_videos/ se desejar

# Passo 3: Dubla usando o SRT revisado (requer -i apontando para o vídeo correspondente)
python run.py -i "input_videos\aula01.mp4" --from-srt "output_videos\aula01_dubbed.srt"
```

### 8. Usar outro modelo Whisper
```powershell
python run.py --model large-v3-turbo
# Opções: base, small, medium, large-v3 (padrão), large-v3-turbo
```

---

## 🎭 Vozes Neurais Disponíveis

| Voz | Tom | Recomendado Para |
|---|---|---|
| `pt-BR-AntonioNeural` ⭐ | Masculino | Uso geral, cursos técnicos, tutoriais |
| `pt-BR-DonatoNeural` | Masculino (grave) | Documentários, apresentações formais |
| `pt-BR-FranciscaNeural` | Feminino | Cursos corporativos, didática clara |
| `pt-BR-BrendaNeural` | Feminino (suave) | Narrativas, tutoriais tranquilos |

---

## 📚 Glossário Técnico Personalizado

Para adicionar termos específicos da sua área que **nunca devem ser traduzidos** (ex: bibliotecas novas, ferramentas internas), basta criar um arquivo `glossario.txt` na raiz do projeto com um termo por linha:

```text
# glossario.txt
LangChain
LlamaIndex
FastAPI
ChromaDB
```
O sistema carregará automaticamente esses termos junto com os mais de 120 termos técnicos já pré-configurados em `config_ptbr.py`.

---

## 🔧 Configuração e Diagnóstico

Para verificar se todo o ambiente está pronto antes de dublar:
```powershell
python config_ptbr.py
```
Isso validará `ffmpeg`, `ffprobe`, `faster-whisper`, `onnxruntime`, `edge-tts`, `deep-translator`, `soundfile`, `numpy`, `tqdm` e exibirá a configuração ativa.
