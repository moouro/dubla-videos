# -*- coding: utf-8 -*-
"""
Módulo de Tradução com Preservação de Termos Técnicos, Chunking e Fallback Resiliente
"""

import concurrent.futures
import logging
import random
import re
import time
from typing import List, Dict, Tuple

from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.exceptions import TooManyRequests
from tqdm import tqdm

from dublagem.config import (
    TERMOS_TECNICOS, MAX_RETRIES, RETRY_DELAY, MSG,
    MAX_TEXT_CHUNK_SIZE, TRANSLATION_TIMEOUT
)

_CASE_SENSITIVE_TERMS = {"Go", "Rust", "Swift", "Java", "Node", "Rails"}

_MYMEMORY_LANG_MAP = {
    "en": "en-US",
    "pt": "pt-BR",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "ja": "ja-JP",
    "zh": "zh-CN",
    "ru": "ru-RU",
    "ko": "ko-KR",
}


def _to_mymemory_lang(lang: str) -> str:
    """Converte código de idioma para o padrão regional exigido pela API do MyMemory."""
    if not lang or lang.lower() == "auto":
        return "en-US"
    cleaned = lang.strip().replace("_", "-")
    if "-" in cleaned:
        parts = cleaned.split("-", 1)
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return _MYMEMORY_LANG_MAP.get(cleaned.lower(), cleaned.lower())


def _protect_tech_terms(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Substitui termos técnicos por placeholders (ex: TECHTERM000) antes da tradução.
    Respeita case-sensitivity para termos curtos/palavras comuns (como Go, Rust).
    """
    placeholders: Dict[str, str] = {}
    protected = text

    sorted_terms = sorted(TERMOS_TECNICOS, key=len, reverse=True)

    idx = 0
    for term in sorted_terms:
        pattern_str = rf"(?<!\w){re.escape(term)}(?!\w)"
        flags = 0 if term in _CASE_SENSITIVE_TERMS else re.IGNORECASE
        pattern = re.compile(pattern_str, flags)

        matches = list(pattern.finditer(protected))
        for match in reversed(matches):
            original = match.group()
            placeholder = f"TECHTERM{idx:03d}"
            placeholders[placeholder] = original
            protected = protected[:match.start()] + placeholder + protected[match.end():]
            idx += 1

    return protected, placeholders


def _restore_tech_terms(text: str, placeholders: Dict[str, str]) -> str:
    """
    Restaura os termos originais substituindo os placeholders de volta.
    Tolera espaços inseridos por tradutores (ex: 'TECHTERM 001') e caracteres especiais como '\\'.
    """
    restored = text
    for placeholder, original in placeholders.items():
        match = re.match(r"^TECHTERM(\d+)$", placeholder)
        if match:
            digits = match.group(1)
            pattern = re.compile(rf"TECHTERM\s*{re.escape(digits)}", re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
        restored = pattern.sub(lambda _: original, restored)
    return restored


def _split_text_chunks(text: str, max_chars: int) -> List[str]:
    """
    Divide textos longos respeitando limites de frase e palavras,
    sem nunca quebrar placeholders TECHTERM ou palavras ao meio.
    """
    stripped = text.strip()
    if not stripped or max_chars <= 0 or len(stripped) <= max_chars:
        return [stripped] if stripped else []

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) <= max_chars:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
        else:
            if current:
                chunks.append(current)
                current = ""

            # Divide frases muito longas por espaços (preservando vírgulas anexadas)
            words = re.split(r"\s+", sentence)
            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = word

    if current:
        chunks.append(current)

    return chunks


def _translate_with_timeout(translator, text: str, timeout: int) -> str:
    """Executa translator.translate com timeout real sem bloquear em caso de expiração."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(translator.translate, text)
        return future.result(timeout=timeout)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _translate_chunked_google(translator: GoogleTranslator, text: str) -> str:
    chunks = _split_text_chunks(text, MAX_TEXT_CHUNK_SIZE)
    translated_parts = []
    for chunk in chunks:
        res = _translate_with_timeout(translator, chunk, TRANSLATION_TIMEOUT)
        if not res:
            raise RuntimeError("Resposta vazia do GoogleTranslator")
        translated_parts.append(res.strip())
    return " ".join(translated_parts)


def _translate_chunked_mymemory(translator: MyMemoryTranslator, text: str) -> str:
    # MyMemory possui limite rígido de 500 caracteres por requisição
    mymemory_limit = min(MAX_TEXT_CHUNK_SIZE, 450)
    chunks = _split_text_chunks(text, mymemory_limit)
    translated_parts = []
    for chunk in chunks:
        res = _translate_with_timeout(translator, chunk, TRANSLATION_TIMEOUT)
        if not res:
            raise RuntimeError("Resposta vazia do MyMemoryTranslator")
        upper_res = res.upper()
        if "QUERY LENGTH LIMIT EXCEEDED" in upper_res or "MYMEMORY WARNING" in upper_res:
            raise RuntimeError(f"Erro/aviso retornado pelo MyMemory: {res[:120]}")
        translated_parts.append(res.strip())
    return " ".join(translated_parts)


def translate_segments(segments: List[Dict], src_lang: str = "auto", dest_lang: str = "pt") -> List[Dict]:
    """
    Traduz uma lista de segmentos de texto preservando a pontuação e os termos técnicos.
    Inclui chunking, timeout, retry exponencial com jitter e fallback dinâmico para MyMemory.
    """
    if src_lang != "auto" and src_lang.split("-")[0].lower() == dest_lang.split("-")[0].lower():
        logging.info(f"ℹ️ Idioma de origem ({src_lang}) e destino ({dest_lang}) equivalentes. Pulando tradução.")
        return [seg.copy() for seg in segments]

    logging.info(MSG["traduzindo"])
    google_translator = GoogleTranslator(
        source=src_lang if src_lang != "auto" else "auto",
        target=dest_lang
    )

    mymemory_translator = None
    try:
        mm_src = _to_mymemory_lang(src_lang)
        mm_dest = _to_mymemory_lang(dest_lang)
        if mm_src.lower() != mm_dest.lower():
            mymemory_translator = MyMemoryTranslator(source=mm_src, target=mm_dest)
    except Exception as e:
        logging.debug(f"MyMemoryTranslator indisponível ({src_lang} -> {dest_lang}): {e}")

    translated = []
    terms_preserved = 0
    use_fallback_primary = False

    for i, seg in enumerate(tqdm(segments, desc="Traduzindo", unit="seg")):
        new_seg = seg.copy()
        text = seg["text"].strip()

        if not text:
            translated.append(new_seg)
            continue

        protected_text, placeholders = _protect_tech_terms(text)
        if placeholders:
            terms_preserved += len(placeholders)

        translated_text = ""
        success = False

        # Se Google já bloqueou na sessão atual, prioriza MyMemory
        if use_fallback_primary and mymemory_translator:
            try:
                res = _translate_chunked_mymemory(mymemory_translator, protected_text)
                if res:
                    translated_text = _restore_tech_terms(res, placeholders)
                    success = True
            except Exception as e:
                logging.warning(f"⚠️ Fallback MyMemory falhou no segmento {i}: {e}")

        if not success:
            for attempt in range(MAX_RETRIES):
                try:
                    result = _translate_chunked_google(google_translator, protected_text)
                    if result:
                        translated_text = _restore_tech_terms(result, placeholders)
                        success = True
                        break
                except Exception as e:
                    err_lower = str(e).lower()
                    is_rate_limit = (
                        isinstance(e, TooManyRequests)
                        or "too many requests" in err_lower
                        or "429" in err_lower
                    )
                    if is_rate_limit and mymemory_translator:
                        logging.info(f"🔄 Rate-limit do Google detectado. Ativando fallback MyMemory (seg {i})...")
                        use_fallback_primary = True
                        try:
                            alt_res = _translate_chunked_mymemory(mymemory_translator, protected_text)
                            if alt_res:
                                translated_text = _restore_tech_terms(alt_res, placeholders)
                                success = True
                                break
                        except Exception as ex2:
                            logging.warning(f"⚠️ MyMemory fallback também falhou: {ex2}")

                    wait_time = RETRY_DELAY * (2 ** attempt) + random.uniform(0.2, 1.0)
                    if attempt < MAX_RETRIES - 1:
                        logging.warning(
                            f"⚠️ Falha na tradução do segmento {i} (tentativa {attempt + 1}/{MAX_RETRIES}). "
                            f"Aguardando {wait_time:.1f}s... Erro: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logging.error(f"❌ Erro ao traduzir o segmento {i} no Google após {MAX_RETRIES} tentativas: {e}")

        # Última tentativa no MyMemory antes de manter o texto original
        if not success and mymemory_translator and not use_fallback_primary:
            try:
                alt_res = _translate_chunked_mymemory(mymemory_translator, protected_text)
                if alt_res:
                    translated_text = _restore_tech_terms(alt_res, placeholders)
                    success = True
            except Exception:
                pass

        if not success:
            logging.warning(f"⚠️ Mantendo texto original para o segmento {i} como fallback de emergência.")
            new_seg["text"] = text
        else:
            new_seg["text"] = translated_text

        translated.append(new_seg)

        if i < len(segments) - 1:
            time.sleep(0.6)

    logging.info(f"✅ Tradução concluída: {len(translated)} segmentos processados ({terms_preserved} termos preservados).")
    return translated
