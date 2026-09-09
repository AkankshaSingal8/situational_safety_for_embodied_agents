from pathlib import Path


def load_prompt(prompt_path: Path | str,
                return_stem: bool = False) -> str | tuple[str, str]:
    """Read a .md prompt file and return its contents.

    If return_stem=True, returns (content, stem) where stem is the
    filename without extension — used to name the output directory.
    """
    prompt_path = Path(prompt_path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    content = prompt_path.read_text(encoding="utf-8")
    if return_stem:
        return content, prompt_path.stem
    return content
