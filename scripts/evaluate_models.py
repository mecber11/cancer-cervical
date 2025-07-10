#!/usr/bin/env python3
"""
Script para evaluar modelos entrenados
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import *
from src.evaluation.evaluator import evaluate_all_models

def main():
    print("Iniciando evaluacion de modelos...")
    evaluate_all_models()

if __name__ == "__main__":
    main()
