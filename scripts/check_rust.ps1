$ErrorActionPreference = "Stop"

. "$PSScriptRoot\_lib.ps1"

Invoke-Checked { cargo fmt --all --check }
Invoke-Checked { cargo clippy --workspace --all-targets -- -D warnings }
Invoke-Checked { cargo test --workspace }
