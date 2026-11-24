#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ponto de Entrada para o Sistema de Dublagem Automática de Vídeos v4.0 (100% Offline)
Encaminha a execução para o pacote modular 'dublagem' dentro da .venv (Python 3.11).
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

_VENV_DIR = Path(__file__).resolve().parent / ".venv"
_VENV_PYTHON = _VENV_DIR / "Scripts" / "python.exe"
if (
    _VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != _VENV_DIR.resolve()
    and os.environ.get("LEXIVOX_VENV_BOOTSTRAPPED") != "1"
):
    _env = os.environ.copy()
    _env["LEXIVOX_VENV_BOOTSTRAPPED"] = "1"
    _res = subprocess.run([str(_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env=_env)
    sys.exit(_res.returncode)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dublagem.cli import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Operação cancelada pelo usuário.")
    except Exception as e:
        print(f"❌ Erro fatal durante a execução: {e}")
        sys.exit(1)
