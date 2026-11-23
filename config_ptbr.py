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
WHISPER_COMPUTE_TYPE = "int8"       # Otimizado para Intel Core i5 com AVX2
WHISPER_CPU_THREADS = 10           # 10 threads para equilibrar com os núcleos do i5-13400F
WHISPER_VAD_FILTER = True          # Silero VAD para filtrar silêncios e ruídos de teclado
WHISPER_VAD_MIN_SILENCE_MS = 400   # Pausa mínima para considerar corte sem alucinações

# =====================================================
# IDIOMA E TRADUÇÃO
# =====================================================

IDIOMA_DESTINO = "pt"

# =====================================================
# VOZES TTS (Microsoft Edge Neural)
# =====================================================

VOZES = {
    "antonio": "pt-BR-AntonioNeural",    # Masculina - natural e clara
    "donato": "pt-BR-DonatoNeural",      # Masculina - tom mais grave
    "francisca": "pt-BR-FranciscaNeural",  # Feminina - profissional
    "brenda": "pt-BR-BrendaNeural",      # Feminina - tom mais suave
}

VOZ_PADRAO = VOZES["antonio"]

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
# TRADUÇÃO
# =====================================================

MAX_TEXT_CHUNK_SIZE = 800

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
# Limite máximo de caracteres por segmento merged (mantém compatibilidade com MyMemory <= 500 e SRT legível)
MERGE_MAX_CHARS = 450

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
    "traduzindo": "🌐 Traduzindo para português brasileiro...",
    "sintetizando": "🎤 Gerando nova dublagem...",
    "sincronizando": "⚡ Sincronizando com vídeo original...",
    "finalizando": "🎯 Finalizando vídeo dublado...",
    "concluido": "✅ Dublagem concluída com sucesso!",
    "erro": "❌ Erro durante o processamento:",
}

# =====================================================
# VALIDAÇÃO
# =====================================================

def validar():
    """Valida se a configuração e dependências estão prontas"""
    erros = []

    # Testa FFmpeg e FFprobe
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

    # Testa bibliotecas Python obrigatórias
    dependencias = [
        ("faster_whisper", "faster-whisper"),
        ("onnxruntime", "onnxruntime (necessário para Silero VAD)"),
        ("edge_tts", "edge-tts"),
        ("deep_translator", "deep-translator"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]
    for modulo, pacote in dependencias:
        try:
            __import__(modulo)
        except ImportError:
            erros.append(f"{pacote} não instalado. Execute: pip install -r requirements.txt")

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

    print("🔧 Configuração (LexiVox v3.2):")
    print(f"   • Motor Whisper: faster-whisper ({WHISPER_MODEL}) [Device: {WHISPER_DEVICE} | Threads: {WHISPER_CPU_THREADS}]")
    print(f"   • Voz padrão:    {VOZ_PADRAO}")
    print(f"   • Velocidade:    {VELOCIDADE_PADRAO}")
    print(f"   • Idioma:        auto -> {IDIOMA_DESTINO}")
    print(f"   • Modo de áudio: {modo_audio}")
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
