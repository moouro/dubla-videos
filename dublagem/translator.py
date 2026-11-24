# -*- coding: utf-8 -*-
"""
Módulo de Tradução Offline via NLLB-200 (CTranslate2) com Preservação de Termos Técnicos
Zero dependência de APIs externas/pagas após o download inicial do modelo.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import ctranslate2
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer

from dublagem.config import (
    MODELS_DIR, TERMOS_TECNICOS, MSG,
    MAX_TEXT_CHUNK_SIZE, TRANSLATOR_MODEL,
    TRANSLATOR_TOKENIZER, TRANSLATOR_CPU_THREADS
)

_CASE_SENSITIVE_TERMS = {"Go", "Rust", "Swift", "Java", "Node", "Rails"}

_NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "pt": "por_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
}

_CACHED_TRANSLATOR: Optional[ctranslate2.Translator] = None
_CACHED_TOKENIZER = None


def _to_nllb_lang(lang: str) -> str:
    """Converte códigos ISO 639-1 (ex: 'en', 'pt') para códigos Flores-200 do NLLB-200."""
    if not lang or lang.lower() == "auto":
        return "eng_Latn"
    cleaned = lang.strip()
    if "_" in cleaned and len(cleaned) >= 8:
        return cleaned
    base = cleaned.replace("_", "-").split("-")[0].lower()
    if base not in _NLLB_LANG_MAP:
        raise ValueError(f"Idioma '{lang}' não mapeado para NLLB-200. Mapeados: {sorted(_NLLB_LANG_MAP.keys())}")
    return _NLLB_LANG_MAP[base]


def _get_translator_and_tokenizer():
    """Carrega e mantém em cache o modelo CTranslate2 NLLB-200 e seu tokenizer offline."""
    global _CACHED_TRANSLATOR, _CACHED_TOKENIZER

    if _CACHED_TRANSLATOR is not None and _CACHED_TOKENIZER is not None:
        return _CACHED_TRANSLATOR, _CACHED_TOKENIZER

    nllb_root = MODELS_DIR / "nllb-200"
    nllb_root.mkdir(parents=True, exist_ok=True)

    safe_repo = re.sub(r"[^a-zA-Z0-9._-]", "_", TRANSLATOR_MODEL)
    model_dir = nllb_root / safe_repo
    tokenizer_dir = nllb_root / "tokenizer"

    if not (model_dir / "model.bin").exists():
        logging.info(f"📥 Baixando modelo de tradução offline {TRANSLATOR_MODEL} (apenas na 1ª execução)...")
        snapshot_download(repo_id=TRANSLATOR_MODEL, local_dir=str(model_dir))

    if not (tokenizer_dir / "tokenizer_config.json").exists():
        logging.info(f"📥 Baixando tokenizer NLLB-200 ({TRANSLATOR_TOKENIZER})...")
        tok = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
        tok.save_pretrained(str(tokenizer_dir))
        _CACHED_TOKENIZER = tok
    else:
        _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    logging.info(f"🧠 Carregando NLLB-200 CTranslate2 ({TRANSLATOR_MODEL} | Threads: {TRANSLATOR_CPU_THREADS})...")
    _CACHED_TRANSLATOR = ctranslate2.Translator(
        str(model_dir),
        device="cpu",
        compute_type="int8",
        intra_threads=TRANSLATOR_CPU_THREADS
    )

    return _CACHED_TRANSLATOR, _CACHED_TOKENIZER


def _protect_tech_terms(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Substitui termos técnicos compostos ou CamelCase/com símbolos por marcadores curtos ([0], [1], ...).
    Siglas puramente maiúsculas de até 6 letras (CSS, HTML, SQL, API, JSON, etc.) são preservadas
    diretamente pois o NLLB-200 as mantém intactas com melhor fluidez gramatical.
    """
    placeholders: Dict[str, str] = {}
    protected = text
    sorted_terms = sorted(TERMOS_TECNICOS, key=len, reverse=True)

    idx = 0
    for term in sorted_terms:
        if term.isalpha() and term.isupper() and len(term) <= 6:
            continue

        pattern_str = rf"(?<!\w){re.escape(term)}(?!\w)"
        flags = 0 if term in _CASE_SENSITIVE_TERMS else re.IGNORECASE
        pattern = re.compile(pattern_str, flags)

        matches = list(pattern.finditer(protected))
        for match in reversed(matches):
            original = match.group()
            placeholder = f"[{idx}]"
            placeholders[str(idx)] = original
            protected = protected[:match.start()] + placeholder + protected[match.end():]
            idx += 1

    return protected, placeholders


def _restore_tech_terms(text: str, placeholders: Dict[str, str]) -> str:
    """
    Restaura os termos originais substituindo os marcadores [N] de volta.
    Tolera espaços internos inseridos por tokenizadores (ex: '[ 0 ]') e barras invertidas.
    """
    restored = text
    for num, original in placeholders.items():
        pattern = re.compile(rf"\[\s*{re.escape(num)}\s*\]")
        restored = pattern.sub(lambda _: original, restored)
    return restored


def _split_text_chunks(text: str, max_chars: int) -> List[str]:
    """Divide textos longos respeitando limites de frase e palavras sem quebrar placeholders."""
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

            for word in re.split(r"\s+", sentence):
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


def _translate_single_text(
    translator: ctranslate2.Translator,
    tokenizer,
    text: str,
    src_nllb: str,
    tgt_nllb: str
) -> str:
    """Traduz uma string dividindo em chunks seguros dentro da janela do NLLB-200."""
    chunks = _split_text_chunks(text, MAX_TEXT_CHUNK_SIZE)
    translated_parts: List[str] = []

    tokenizer.src_lang = src_nllb

    for chunk in chunks:
        token_ids = tokenizer.encode(chunk)
        source_tokens = tokenizer.convert_ids_to_tokens(token_ids)

        results = translator.translate_batch(
            [source_tokens],
            target_prefix=[[tgt_nllb]],
            beam_size=4,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            max_decoding_length=512
        )

        output_tokens = list(results[0].hypotheses[0])
        if output_tokens and output_tokens[0] == tgt_nllb:
            output_tokens = output_tokens[1:]

        out_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        decoded = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        if not decoded:
            raise RuntimeError("NLLB-200 retornou tradução vazia para o chunk.")
        translated_parts.append(decoded)

    return " ".join(translated_parts)


def translate_segments(
    segments: List[Dict],
    src_lang: str = "en",
    dest_lang: str = "pt"
) -> Tuple[List[Dict], int]:
    """
    Traduz todos os segmentos offline via NLLB-200 (CTranslate2) preservando termos técnicos.
    Retorna (segmentos_traduzidos, quantidade_de_falhas).
    """
    if src_lang != "auto" and src_lang.split("-")[0].lower() == dest_lang.split("-")[0].lower():
        logging.info(f"ℹ️ Idioma de origem ({src_lang}) e destino ({dest_lang}) equivalentes. Pulando tradução.")
        return [seg.copy() for seg in segments], 0

    logging.info(MSG["traduzindo"])
    src_nllb = _to_nllb_lang(src_lang)
    tgt_nllb = _to_nllb_lang(dest_lang)

    translator, tokenizer = _get_translator_and_tokenizer()

    translated: List[Dict] = []
    terms_preserved = 0
    failed_count = 0

    for i, seg in enumerate(tqdm(segments, desc="Traduzindo (NLLB)", unit="seg", dynamic_ncols=True)):
        new_seg = seg.copy()
        text = seg.get("text", "").strip()

        if not text:
            new_seg["translation_failed"] = False
            translated.append(new_seg)
            continue

        protected_text, placeholders = _protect_tech_terms(text)
        if placeholders:
            terms_preserved += len(placeholders)

        try:
            raw_translated = _translate_single_text(
                translator, tokenizer, protected_text, src_nllb, tgt_nllb
            )
            restored = _restore_tech_terms(raw_translated, placeholders)
            new_seg["text"] = restored
            new_seg["translation_failed"] = False
        except Exception as e:
            failed_count += 1
            new_seg["text"] = text
            new_seg["translation_failed"] = True
            logging.error(f"❌ Falha na tradução offline do segmento {i}: {e}")

        translated.append(new_seg)

    logging.info(
        f"✅ Tradução offline concluída: {len(translated) - failed_count}/{len(translated)} segmentos "
        f"({terms_preserved} termos técnicos preservados, {failed_count} falhas)."
    )
    return translated, failed_count
