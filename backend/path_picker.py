from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT


class NativePathPickerError(RuntimeError):
    pass


def _resolve_initial_directory(initial_path: str, kind: str) -> Path:
    raw_value = str(initial_path or "").strip()
    if raw_value:
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.exists():
            if candidate.is_dir():
                return candidate
            if kind == "file":
                return candidate.parent
            return candidate.parent
        if candidate.suffix:
            return candidate.parent
        return candidate
    return PROJECT_ROOT


def _pick_with_osascript(kind: str, title: str, initial_directory: Path) -> Optional[str]:
    choose_target = "folder" if kind == "directory" else "file"
    script = [
        f'set initialLocation to POSIX file "{initial_directory.as_posix()}"',
        f'set chosenItem to choose {choose_target} with prompt "{title.replace(chr(34), chr(39))}" default location initialLocation',
        "return POSIX path of chosenItem",
    ]
    result = subprocess.run(
        ["osascript", *sum([["-e", line] for line in script], [])],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        selected_path = result.stdout.strip()
        return selected_path or None

    stderr = (result.stderr or "").strip().lower()
    if "user canceled" in stderr or "cancelled" in stderr:
        return None
    raise NativePathPickerError(result.stderr.strip() or "AppleScript path picker failed.")


def _pick_with_tkinter(kind: str, title: str, initial_directory: Path) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - platform dependent
        raise NativePathPickerError("Tkinter is not available in this Python environment.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        if kind == "directory":
            selected_path = filedialog.askdirectory(
                title=title,
                initialdir=str(initial_directory),
                parent=root,
                mustexist=False,
            )
        else:
            selected_path = filedialog.askopenfilename(
                title=title,
                initialdir=str(initial_directory),
                parent=root,
            )
        return str(selected_path).strip() or None
    except Exception as exc:  # pragma: no cover - GUI runtime safeguard
        raise NativePathPickerError(str(exc)) from exc
    finally:
        root.destroy()


def open_native_path_picker(kind: str, title: str, initial_path: str = "") -> Optional[str]:
    normalized_kind = str(kind or "directory").strip().lower()
    if normalized_kind not in {"directory", "file"}:
        raise ValueError("kind must be 'directory' or 'file'.")

    initial_directory = _resolve_initial_directory(initial_path, normalized_kind)
    dialog_title = str(title or "Choisir un chemin").strip() or "Choisir un chemin"

    if sys.platform == "darwin":
        return _pick_with_osascript(normalized_kind, dialog_title, initial_directory)

    return _pick_with_tkinter(normalized_kind, dialog_title, initial_directory)
