# -*- coding: utf-8 -*-
"""
Configuração para Dublagem em Português Brasileiro
Todas as configurações são importadas e utilizadas pelo dublar.py
"""

from pathlib import Path
import sys

# Força UTF-8 no console Windows (cp1252 não suporta emojis)
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

import os
import shutil

# Detecção dinâmica do FFmpeg (garante compatibilidade com Winget e PATH do sistema)
if not shutil.which("ffmpeg"):
    winget_packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_packages.exists():
        for ffmpeg_candidate in winget_packages.glob("**/ffmpeg.exe"):
            bin_dir = str(ffmpeg_candidate.parent)
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.path.pathsep + os.environ.get("PATH", "")
            break

# =====================================================
# CAMINHOS
# =====================================================

SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "input_videos"
OUTPUT_DIR = SCRIPT_DIR / "output_videos"
MODELS_DIR = SCRIPT_DIR / "models"

# =====================================================
# MODELO WHISPER (faster-whisper)
# =====================================================

# Padrão ouro: large-v3 para precisão técnica máxima em termos de programação
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"               # Otimizado para Intel Core i5 com AVX2
WHISPER_CPU_THREADS = 10                   # 10 threads para equilibrar com os núcleos do i5-13400F
WHISPER_VAD_FILTER = True                  # Silero VAD para filtrar silêncios e ruídos de teclado
WHISPER_VAD_MIN_SILENCE_MS = 400           # Pausa mínima para considerar corte sem alucinações
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False # Evita loops de repetição em pausas longas
WHISPER_REPETITION_PENALTY = 1.1           # Penaliza tokens repetidos em sequência
WHISPER_NO_REPEAT_NGRAM_SIZE = 3           # Bloqueia n-grams repetidos consecutivos
WHISPER_HALLUCINATION_SILENCE_THRESHOLD = 2.0

# =====================================================
# IDIOMA E TRADUÇÃO OFFLINE (NLLB-200 via CTranslate2)
# =====================================================

IDIOMA_DESTINO = "pt"
TRANSLATOR_MODEL = "OpenNMT/nllb-200-3.3B-ct2-int8"
TRANSLATOR_TOKENIZER = "facebook/nllb-200-distilled-600M"
TRANSLATOR_CPU_THREADS = 10

# =====================================================
# VOZES TTS OFFLINE (Coqui XTTS-v2 — Neural PT-BR)
# =====================================================

TTS_ENGINE = "xtts"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_CPU_THREADS = 10

VOZES = {
    "marcos": "Marcos Rudaski",     # Masculina - natural e clara (padrão)
    "damien": "Damien Black",       # Masculina - tom mais grave
    "claribel": "Claribel Dervla",  # Feminina - profissional e clara
    "ana": "Ana Florence",          # Feminina - tom suave
    "clone": "auto_clone",          # Clona automaticamente ~10s da voz do instrutor original
}

VOZ_PADRAO = VOZES["marcos"]

# =====================================================
# VELOCIDADE DE FALA
# =====================================================

VELOCIDADE_PADRAO = "+0%"   # Velocidade normal para naturalidade

# =====================================================
# PROCESSAMENTO DE ÁUDIO
# =====================================================

AUDIO_SAMPLE_RATE = 44100
MIN_SPEED_FACTOR = 0.7
MAX_SPEED_FACTOR = 1.15

# =====================================================
# CONFIGURAÇÕES DE RETRY E TIMEOUT
# =====================================================

MAX_RETRIES = 3
RETRY_DELAY = 2       # segundos (com backoff exponencial)
TTS_TIMEOUT = 60       # segundos
TRANSLATION_TIMEOUT = 30

# =====================================================
# TRADUÇÃO E RESILIÊNCIA (FAIL-FAST)
# =====================================================

MAX_TEXT_CHUNK_SIZE = 450
MAX_FAILURE_RATIO = 0.10  # Aborta o vídeo se >10% dos segmentos falharem na tradução ou TTS

# Termos técnicos que NÃO devem ser traduzidos (web dev / full stack)
# Adicione ou remova termos conforme necessário
TERMOS_TECNICOS = [
    # Frameworks e Libs
    "React", "React.js", "Vue", "Vue.js", "Svelte", "SvelteKit", "Angular",
    "Next.js", "Nuxt", "Nuxt.js", "Astro", "Remix", "Gatsby",
    "Node.js", "Node", "Express", "Express.js", "Fastify", "Nest.js", "NestJS",
    "Django", "Flask", "Laravel", "Rails", "Ruby on Rails", "Spring Boot",
    "Tailwind", "Tailwind CSS", "Bootstrap", "Sass", "SCSS", "Less",
    "jQuery", "Alpine.js", "HTMX", "Solid", "SolidJS", "Qwik",
    "Three.js", "D3.js", "Chart.js",
    # Linguagens e tecnologias core
    "JavaScript", "TypeScript", "Python", "HTML", "CSS", "SQL", "NoSQL",
    "PHP", "Ruby", "Rust", "Go", "Golang", "Java", "Kotlin", "Swift", "C#",
    "JSX", "TSX", "JSON", "XML", "YAML", "Markdown",
    # Ferramentas e plataformas
    "Git", "GitHub", "GitLab", "Bitbucket",
    "Docker", "Kubernetes", "Terraform",
    "npm", "yarn", "pnpm", "Bun", "Deno",
    "Webpack", "Vite", "Rollup", "esbuild", "Turbopack", "Parcel",
    "VS Code", "Visual Studio Code", "ESLint", "Prettier",
    "Vercel", "Netlify", "AWS", "Azure", "Heroku", "Cloudflare",
    "Firebase", "Supabase", "PlanetScale", "Prisma", "Drizzle",
    # Bancos de dados e APIs
    "MongoDB", "PostgreSQL", "MySQL", "SQLite", "Redis",
    "API", "REST", "RESTful", "GraphQL", "gRPC", "WebSocket",
    # Específicos e Siglas Técnicas que não devem ser traduzidas
    "SSR", "SSG", "ISR", "SPA", "MPA", "DOM", "ORM", "CRUD", "MVC", "MVVM",
    "CI/CD", "DevOps", "JWT", "OAuth", "CDN", "DNS", "localhost",
    # Serviços de Terceiros
    "Stripe", "PayPal", "Twilio", "SendGrid",
]

# Carrega termos extras de glossario.txt se existir na raiz
GLOSSARIO_PATH = SCRIPT_DIR / "glossario.txt"
if GLOSSARIO_PATH.exists():
    try:
        with open(GLOSSARIO_PATH, "r", encoding="utf-8") as f:
            for line in f:
                termo = line.strip()
                if termo and not termo.startswith("#") and termo not in TERMOS_TECNICOS:
                    TERMOS_TECNICOS.append(termo)
    except Exception:
        pass

# =====================================================
# MERGE DE SEGMENTOS (Otimizado para Tutoriais de Tela)
# =====================================================

# Gap máximo (ms) entre segmentos para considerar merge de frases
MERGE_MAX_GAP_MS = 600
# Duração máxima (ms) de um segmento merged
MERGE_MAX_DURATION_MS = 14000
# Limite máximo de caracteres por segmento merged (ótima prosódia no XTTS-v2 e precisão no NLLB-200)
MERGE_MAX_CHARS = 350

# =====================================================
# SEPARAÇÃO DE ÁUDIO (Voz vs Música/Efeitos)
# =====================================================

# Habilita ou desabilita o uso de audio-separator (MDX-Net ONNX)
USE_AUDIO_SEPARATOR = False

# Modelo padrão ONNX para separação de vocais/instrumental
AUDIO_SEPARATOR_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"

# Volume do instrumental (música e efeitos de fundo) na mixagem final (0.0 a 1.0)
INSTRUMENTAL_VOLUME = 1.0

# Transcreve a partir da faixa de voz limpa isolada em vez do áudio completo do vídeo
TRANSCRIBE_ORIGINAL_VOCALS = True

# =====================================================
# AUDIO DUCKING (Modo Filme — Dublagem Total)
# =====================================================

# Volume do áudio original DURANTE a dublagem (0.0 a 1.0)
# 0.0 = dublagem estilo filme: voz original completamente substituída
DUCKING_VOLUME = 0.0

# Volume do áudio original onde NÃO há dublagem (0.0 a 1.0)
# 0.0 = silêncio total do original (room tone preenche gaps automaticamente)
ORIGINAL_VOLUME = 0.0

# Fade in/out em ms para transição de ducking
DUCKING_FADE_MS = 150

# Fração do gap até o próximo segmento que pode ser usada para overflow
# Em tutoriais de tela sem rosto humano, 0.85 permite fala 100% natural sem aceleração
OVERFLOW_GAP_RATIO = 0.85

# Overflow máximo em ms
OVERFLOW_MAX_MS = 1500

# =====================================================
# FORMATOS SUPORTADOS
# =====================================================

FORMATOS_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")

# =====================================================
# MENSAGENS EM PORTUGUÊS
# =====================================================

MSG = {
    "iniciando": "🎬 Iniciando dublagem automática...",
    "detectando": "🔍 Detectando idioma original...",
    "transcrevendo": "📝 Transcrevendo áudio com faster-whisper...",
    "traduzindo": "🌐 Traduzindo offline com NLLB-200 (CTranslate2)...",
    "sintetizando": "🎤 Gerando dublagem neural offline (Coqui XTTS-v2)...",
    "sincronizando": "⚡ Sincronizando com vídeo original...",
    "finalizando": "🎯 Finalizando vídeo dublado...",
    "concluido": "✅ Dublagem concluída com sucesso!",
    "erro": "❌ Erro durante o processamento:",
}

# =====================================================
# VALIDAÇÃO
# =====================================================

def validar():
    """Valida se a configuração, Python 3.11 (.venv) e dependências v4.0 estão prontas"""
    import importlib
    import platform
    erros = []

    if not shutil.which("ffmpeg"):
        erros.append(
            "FFmpeg não encontrado no PATH.\n"
            "   Instale com: winget install --id=yt-dlp.FFmpeg -e"
        )
    if not shutil.which("ffprobe"):
        erros.append(
            "FFprobe não encontrado no PATH (geralmente incluído no pacote do FFmpeg).\n"
            "   Instale com: winget install --id=yt-dlp.FFmpeg -e"
        )

    dependencias = [
        ("faster_whisper", "faster-whisper"),
        ("onnxruntime", "onnxruntime (Silero VAD)"),
        ("ctranslate2", "ctranslate2 (Whisper + NLLB-200)"),
        ("transformers", "transformers (Tokenizer NLLB-200)"),
        ("sentencepiece", "sentencepiece (Tokenizer NLLB-200)"),
        ("huggingface_hub", "huggingface-hub"),
        ("torch", "torch (PyTorch CPU para XTTS-v2)"),
        ("TTS.api", "coqui-tts (XTTS-v2)"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]
    for modulo, pacote in dependencias:
        try:
            importlib.import_module(modulo)
        except ImportError:
            erros.append(f"{pacote} não instalado no Python atual ({platform.python_version()}). Use .venv\\Scripts\\python.exe")

    if erros:
        print("❌ Problemas encontrados:")
        for e in erros:
            print(f"   • {e}")
        return False

    modo_audio = (
        "Modo Filme (0%/0% — substituição total)"
        if DUCKING_VOLUME == 0.0 and ORIGINAL_VOLUME == 0.0
        else f"Ducking customizado ({int(ORIGINAL_VOLUME * 100)}% gaps / {int(DUCKING_VOLUME * 100)}% fala)"
    )

    print(f"🔧 Configuração (LexiVox v4.0 — Python {platform.python_version()}):")
    print(f"   • Transcrição:   faster-whisper ({WHISPER_MODEL}) [Device: {WHISPER_DEVICE} | Threads: {WHISPER_CPU_THREADS}]")
    print(f"   • Tradutor:      {TRANSLATOR_MODEL} (offline CTranslate2)")
    print(f"   • Voz TTS:       {TTS_MODEL} -> '{VOZ_PADRAO}'")
    print(f"   • Velocidade:    {VELOCIDADE_PADRAO}")
    print(f"   • Idioma:        auto -> {IDIOMA_DESTINO}")
    print(f"   • Modo de áudio: {modo_audio}")
    print(f"   • Fail-Fast:     aborta se falha > {int(MAX_FAILURE_RATIO * 100)}% dos segmentos")
    print("✅ Todas as dependências e configurações estão prontas!")
    return True


def listar_vozes():
    """Lista todas as vozes disponíveis"""
    print("🎭 Vozes disponíveis:")
    for nome, voz in VOZES.items():
        marcador = " ⭐" if voz == VOZ_PADRAO else ""
        print(f"   • {nome}: {voz}{marcador}")


if __name__ == "__main__":
    validar()
    print()
    listar_vozes()
