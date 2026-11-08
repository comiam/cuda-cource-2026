"""Trusted course policy; no configuration is loaded from submissions."""

import re
from pathlib import PurePosixPath

LAB_TITLES: dict[int, str] = {
    1: "Vector Operations and Activations",
    2: "Matrix Multiplication",
    3: "Sobel Operator",
    4: "cuDNN Perceptron",
    5: "AlexNet-style CNN on CIFAR-100",
}

REQUIRED_README_SECTIONS: tuple[str, ...] = (
    "Описание",
    "Окружение",
    "Компиляция",
    "Запуск",
    "Проверка корректности",
    "Результаты",
    "Выводы",
    "Воспроизводимость",
)
PROFILE_SECTION = "Профилирование и оптимизация"
REPRODUCTION_SUBSECTIONS: tuple[str, ...] = ("Данные", "Полный запуск")
TRAINING_REPRODUCTION_SUBSECTIONS: tuple[str, ...] = (
    "Параметры обучения",
    "Загрузка весов",
)

MAX_FILE_BYTES = 10 * 1024**2
MAX_SUBMISSION_BYTES = 25 * 1024**2
MAX_CHANGED_FILES = 1000
MAX_SUBMISSION_FILES = 1000

CPP_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}
)
SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".cu"})
PYTHON_SUFFIXES = frozenset({".py"})
CMAKE_SUFFIXES = frozenset({".cmake"})
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".o", ".obj", ".out", ".exe", ".a", ".so", ".dll", ".dylib", ".lib",
        ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".onnx", ".pb", ".safetensors",
        ".ncu-rep", ".nsys-rep", ".prof", ".bin", ".pyc", ".pyo", ".pkl", ".pickle",
    }
)
FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {"build", "CMakeFiles", "__pycache__", ".venv", "venv", ".git", "node_modules"}
)

_LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")


def login_is_valid(value: str) -> bool:
    """Check public GitHub login syntax, not the student's identity."""
    return isinstance(value, str) and "--" not in value and _LOGIN.fullmatch(value) is not None


def lint_group(path: str, submission_path: str) -> str | None:
    """Select only explicitly allowed sources, CMake files and the main README."""
    candidate = PurePosixPath(path)
    root = PurePosixPath(submission_path)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
        return None
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    if relative == PurePosixPath("README.md"):
        return "markdown"
    if candidate.name == "CMakeLists.txt" or candidate.suffix.lower() in CMAKE_SUFFIXES:
        return "cmake"
    if candidate.suffix.lower() in CPP_SUFFIXES:
        return "cpp"
    if candidate.suffix.lower() in PYTHON_SUFFIXES:
        return "python"
    return None
