# -*- coding: utf-8 -*-
"""
Gerenciamento de Configurações e Ambiente para o Sistema de Dublagem
"""

import os
import sys
from pathlib import Path
import logging

# Garante UTF-8 no console Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Adiciona o diretório do config_ptbr.py ao PATH de busca do Python se necessário
SCRIPT_DIR = Path(__file__).parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Importa o arquivo original de configurações do usuário (também garante UTF-8 e FFmpeg no PATH)
try:
    import config_ptbr as user_config
except ImportError:
    # Fallback seguro caso seja executado fora da pasta raiz
    sys.path.insert(0, os.getcwd())
    import config_ptbr as user_config

# --- DIRETÓRIOS ---
INPUT_DIR = user_config.INPUT_DIR
OUTPUT_DIR = user_config.OUTPUT_DIR
MODELS_DIR = user_config.MODELS_DIR
CACHE_DIR = SCRIPT_DIR / ".cache"

# Garante a criação de pastas essenciais
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- MODELO WHISPER ---
WHISPER_MODEL = getattr(user_config, "WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = getattr(user_config, "WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = getattr(user_config, "WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = getattr(user_config, "WHISPER_CPU_THREADS", 10)
WHISPER_VAD_FILTER = getattr(user_config, "WHISPER_VAD_FILTER", True)
WHISPER_VAD_MIN_SILENCE_MS = getattr(user_config, "WHISPER_VAD_MIN_SILENCE_MS", 400)
WHISPER_CONDITION_ON_PREVIOUS_TEXT = getattr(user_config, "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)
WHISPER_REPETITION_PENALTY = getattr(user_config, "WHISPER_REPETITION_PENALTY", 1.1)
WHISPER_NO_REPEAT_NGRAM_SIZE = getattr(user_config, "WHISPER_NO_REPEAT_NGRAM_SIZE", 3)
WHISPER_HALLUCINATION_SILENCE_THRESHOLD = getattr(user_config, "WHISPER_HALLUCINATION_SILENCE_THRESHOLD", 2.0)

# --- TRADUÇÃO OFFLINE E IDIOMA ---
IDIOMA_DESTINO = getattr(user_config, "IDIOMA_DESTINO", "pt")
TERMOS_TECNICOS = getattr(user_config, "TERMOS_TECNICOS", [])
MAX_TEXT_CHUNK_SIZE = getattr(user_config, "MAX_TEXT_CHUNK_SIZE", 450)
MAX_FAILURE_RATIO = getattr(user_config, "MAX_FAILURE_RATIO", 0.10)
TRANSLATOR_MODEL = getattr(user_config, "TRANSLATOR_MODEL", "OpenNMT/nllb-200-3.3B-ct2-int8")
TRANSLATOR_TOKENIZER = getattr(user_config, "TRANSLATOR_TOKENIZER", "facebook/nllb-200-distilled-600M")
TRANSLATOR_CPU_THREADS = getattr(user_config, "TRANSLATOR_CPU_THREADS", 10)

# --- VOZ E VELOCIDADE (XTTS-v2) ---
TTS_ENGINE = getattr(user_config, "TTS_ENGINE", "xtts")
TTS_MODEL = getattr(user_config, "TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
TTS_CPU_THREADS = getattr(user_config, "TTS_CPU_THREADS", 10)
VOZES = user_config.VOZES
VOZ_PADRAO = user_config.VOZ_PADRAO
VELOCIDADE_PADRAO = user_config.VELOCIDADE_PADRAO
MIN_SPEED_FACTOR = getattr(user_config, "MIN_SPEED_FACTOR", 0.7)
MAX_SPEED_FACTOR = getattr(user_config, "MAX_SPEED_FACTOR", 1.15)

# --- CONFIGURAÇÕES DE DUCKING E AUDIO ---
AUDIO_SAMPLE_RATE = getattr(user_config, "AUDIO_SAMPLE_RATE", 44100)
DUCKING_VOLUME = getattr(user_config, "DUCKING_VOLUME", 0.0)
ORIGINAL_VOLUME = getattr(user_config, "ORIGINAL_VOLUME", 0.0)
DUCKING_FADE_MS = getattr(user_config, "DUCKING_FADE_MS", 150)
OVERFLOW_GAP_RATIO = getattr(user_config, "OVERFLOW_GAP_RATIO", 0.85)
OVERFLOW_MAX_MS = getattr(user_config, "OVERFLOW_MAX_MS", 1500)

# --- SEPARAÇÃO DE ÁUDIO ---
USE_AUDIO_SEPARATOR = getattr(user_config, "USE_AUDIO_SEPARATOR", False)
AUDIO_SEPARATOR_MODEL = getattr(user_config, "AUDIO_SEPARATOR_MODEL", "UVR-MDX-NET-Inst_HQ_3.onnx")
INSTRUMENTAL_VOLUME = getattr(user_config, "INSTRUMENTAL_VOLUME", 1.0)
TRANSCRIBE_ORIGINAL_VOCALS = getattr(user_config, "TRANSCRIBE_ORIGINAL_VOCALS", True)

# --- MERGE DE SEGMENTOS ---
MERGE_MAX_GAP_MS = getattr(user_config, "MERGE_MAX_GAP_MS", 600)
MERGE_MAX_DURATION_MS = getattr(user_config, "MERGE_MAX_DURATION_MS", 14000)
MERGE_MAX_CHARS = getattr(user_config, "MERGE_MAX_CHARS", 350)

# --- RETRY E TIMEOUTS ---
MAX_RETRIES = getattr(user_config, "MAX_RETRIES", 3)
RETRY_DELAY = getattr(user_config, "RETRY_DELAY", 2)
TTS_TIMEOUT = getattr(user_config, "TTS_TIMEOUT", 60)
TRANSLATION_TIMEOUT = getattr(user_config, "TRANSLATION_TIMEOUT", 30)

# --- FORMATOS SUPORTADOS ---
FORMATOS_VIDEO = user_config.FORMATOS_VIDEO
MSG = user_config.MSG

# --- LOGGING SETUP ---
def setup_logging():
    """Configura o sistema de logging global"""
    log_file = OUTPUT_DIR / 'dubbing.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True # Garante que o logging seja configurado
    )
