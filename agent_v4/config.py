from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_NAMES = {
    "q4": "qwen3.5-deepseek-q4",
    "q8": "qwen3.5-deepseek-q8",
}

MODEL_FILES = {
    "q4": BASE_DIR / "Modelfile_q4",
    "q8": BASE_DIR / "Modelfile_q8",
}
