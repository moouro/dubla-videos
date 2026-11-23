#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ponto de Entrada para o Sistema de Dublagem Automática de Vídeos v3.2
"""

import asyncio
import sys

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
