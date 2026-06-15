---
name: emoji-commit-messages
description: Enforce a local Git commit-message convention where every commit subject starts with exactly one visible emoji followed by a space and a concise Japanese summary. Use when Codex creates, amends, squashes, or suggests local commit messages in this repository, especially after staging files or before running git commit.
---

# Emoji Commit Messages

## Rule

Start every commit subject with exactly one emoji character, then one space, then a concise Japanese summary.

Examples:

- `✨ 実験ログ用ワークスペースを追加`
- `📝 玉登太郎の設計概要を更新`
- `🔧 ローカルコミット規約を更新`

## Workflow

1. Before committing or amending, choose one emoji that describes the change.
2. Write the rest of the subject in concise Japanese.
3. Keep the subject readable as plain text; do not add multiple emojis or emoji prefixes in the body.
4. When amending an existing local commit whose subject lacks an emoji, preserve the meaning and prepend one emoji.

## Emoji Hints

- `✨` for new features, new skills, or new capabilities.
- `📝` for documentation, instructions, or text-only guidance.
- `🔧` for configuration, tooling, or maintenance.
- `🐛` for bug fixes.
- `✅` for tests or validation changes.

Prefer the clearest emoji over novelty. Use Japanese for the subject unless the change is only meaningful as an identifier, command, or version string.
