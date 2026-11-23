# 🎬 LexiVox — Sistema de Dublagem Automática de Alta Fidelidade (v3.0)

**Dubla vídeos, cursos e screencasts de tecnologia de qualquer idioma para português brasileiro (PT-BR) com máxima fidelidade acústica, inteligência artificial neural e sincronização precisa.**

[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![Status](https://img.shields.io/badge/status-otimizado_v3.0-brightgreen)]()
[![Whisper](https://img.shields.io/badge/faster--whisper-Large--v3-purple)]()
[![TTS](https://img.shields.io/badge/Edge--TTS-Neural-cyan)]()

---

## ⚡ Início Rápido

```powershell
# 1. Instalar dependências no Python
pip install -r requirements.txt

# 2. Colocar seus vídeos na pasta input_videos/

# 3. Dublar!
python run.py
# (ou python dublar.py)
```

Vídeos dublados + legendas sincronizadas `.srt` serão salvos automaticamente em `output_videos/`.

---

## 🚀 Principais Recursos da v3.0

* **`faster-whisper` com Large-v3 & Silero VAD:**
  * O modelo **Large-v3** é baixado e gerenciado automaticamente na pasta `models/faster-whisper/`.
  * **Silero VAD integrado:** remove silêncios, ruídos de fundo e barulho de digitação antes da transcrição, **eliminando 100% das alucinações**.
  * **Word-Level Timestamps:** alinhamento milimétrico da voz ao vídeo.
* **Otimização Especial para Tutoriais de Tela (Screencasts / Cursos de TI):**
  * **Smart Overflow:** sem a restrição de boca humana na tela, o sistema utiliza as pausas naturais do professor para permitir fala em velocidade 100% humana (`+0%`) sem acelerações robóticas.
  * **Preset Voice-Over Premium:** voz em português a 100%, com a fala original do professor mantida a 5% ao fundo e subindo para 25% nas pausas (preservando o ritmo e os sons de digitação/cliques).
  * **Room Tone Contínuo:** preenchimento inteligente de gaps silenciosos, eliminando cortes abruptos de áudio.
* **Sistema de Cache Inteligente:**
  * Baseado no hash MD5 do vídeo em `.cache/`. Se você trocar a voz ou velocidade, a transcrição e tradução são restauradas instantaneamente sem reprocessamento.
  * O cache de áudio TTS é validado pelo hash do texto, garantindo que frases editadas recebam novo áudio automaticamente.
* **Ordenação Alfanumérica Natural:**
  * Processa cursos e aulas na ordem correta (`Aula 1`, `Aula 2`, ..., `Aula 10`).

---

## 💡 Exemplos de Uso

### 1. Dublar todos os vídeos da pasta (Padrão Otimizado)
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

### 5. Modo Semi-Automático (Revisar Legendas Antes de Narrar)
```powershell
# Passo 1: Gera apenas as legendas traduzidas para revisão
python run.py --srt-only

# Passo 2: Revise o arquivo .srt gerado em output_videos/ se desejar

# Passo 3: Dubla usando o SRT revisado
python run.py -i "input_videos\aula01.mp4" --from-srt "output_videos\aula01_dubbed.srt"
```

### 6. Usar modelo mais leve (opcional)
```powershell
python run.py --model large-v3-turbo
# ou --model medium, --model small
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

Para verificar se o seu ambiente está totalmente configurado:
```powershell
python config_ptbr.py
```
Isso validará o FFmpeg, o faster-whisper e todas as bibliotecas necessárias.
# dubla-videos
