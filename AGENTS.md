# Repository Agent Instructions

## PowerShell and Encoding

- When reading or writing text files with PowerShell, always specify encoding explicitly.
- Prefer UTF-8: use `Get-Content -Encoding UTF8`, `Set-Content -Encoding UTF8`, and `Add-Content -Encoding UTF8`.
- Japanese project documents must not be summarized or edited from garbled output. If text appears garbled, retry with the correct encoding first.

## Project Context

- This repository is for the shogi AI **爆裂玉登太郎**.
- The top-level design overview is `爆裂玉登太郎_設計概要.md`.
- Detailed specifications live under `docs/`.
- Environment setup, quality checks, and workspace operation are defined in `docs/16_environment.md`.
- When adding or renaming detailed specification documents under `docs/`, update `爆裂玉登太郎_設計概要.md` in the same change.

## Implementation Split

- Rust owns shogi rules, legal move generation, terminal detection, declaration-win detection, move labels, search, and the USI engine.
- Python owns network code, dataset handling, training, checkpoints, evaluation, and experiment management.
- Keep shared version/schema concepts explicit across Rust, Python, logs, datasets, and checkpoints.

## Quality Checks

- Before handing off code changes, run:

```powershell
.\scripts\check_all.cmd
```

- Rust quality gates are:

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

- Python quality gates are:

```powershell
uv run ruff format --check python tests/python
uv run ruff check python tests/python
uv run ty check python tests/python
uv run pytest
```

## Workspace and Experiment Logs

- `workspace/logs/` is for human-written experiment notes: intent, conditions, results, adoption decisions, failures, and next actions.
- Keep `workspace/README.md` and `workspace/logs/**` tracked.
- Do not commit generated artifacts under:

```text
workspace/runs/
workspace/datasets/
workspace/checkpoints/
workspace/eval_artifacts/
workspace/reanalyse/
workspace/scratch/
```

## Commit Messages

- Follow `.agents/skills/emoji-commit-messages/SKILL.md`.
- Every commit subject must start with exactly one visible emoji, then one space, then a concise Japanese summary.
- Example: `📝 玉登太郎の設計概要を追加`
