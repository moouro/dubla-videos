# -*- coding: utf-8 -*-
"""
Utilitários de Cache e Hashing para o Sistema de Dublagem
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional
from dublagem.config import CACHE_DIR


def _sanitize_cache_key(value: str) -> str:
    """Normaliza valores usados em nomes de arquivo de cache no Windows."""
    cleaned = value.lower().strip().replace("/", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_.]", "_", cleaned) or "default"


def calculate_video_hash(video_path: Path) -> str:
    """
    Calcula um hash MD5 rápido do vídeo baseado nos primeiros e últimos 10MB
    e no tamanho exato do arquivo (preserva cache mesmo se o arquivo for renomeado).
    """
    chunk_size = 1024 * 1024  # 1MB
    hash_md5 = hashlib.md5()
    file_size = video_path.stat().st_size

    with open(video_path, "rb") as f:
        if file_size <= 20 * chunk_size:
            hash_md5.update(f.read())
        else:
            for _ in range(10):
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_md5.update(chunk)

            f.seek(file_size - 10 * chunk_size)
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_md5.update(chunk)

    hash_md5.update(str(file_size).encode("utf-8"))
    return hash_md5.hexdigest()


def get_cache_paths(
    video_hash: str,
    model_name: str = "large-v3",
    dest_lang: str = "pt"
) -> Dict[str, Path]:
    """
    Retorna os caminhos para o cache associado ao hash do vídeo, modelo e idioma destino.
    """
    video_cache_dir = CACHE_DIR / video_hash
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    safe_model = _sanitize_cache_key(model_name)
    safe_dest = _sanitize_cache_key(dest_lang)

    return {
        "dir": video_cache_dir,
        "transcription": video_cache_dir / f"transcription_{safe_model}.json",
        "translation": video_cache_dir / f"translation_{safe_model}_{safe_dest}.json",
    }


def load_cache(file_path: Path) -> Optional[Any]:
    """Carrega dados JSON do cache se ele existir."""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logging.info(f"💾 Cache carregado com sucesso: {file_path.name}")
                return data
        except Exception as e:
            logging.warning(f"⚠️ Falha ao ler cache {file_path.name}: {e}. O processamento será refeito.")
    return None


def save_cache(file_path: Path, data: Any) -> None:
    """Salva dados em formato JSON no cache."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"💾 Progresso salvo em cache: {file_path.name}")
    except Exception as e:
        logging.warning(f"⚠️ Falha ao salvar cache {file_path.name}: {e}")
