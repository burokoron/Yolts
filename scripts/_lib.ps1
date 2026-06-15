function Invoke-Checked {
  param([scriptblock]$Command)

  & $Command
  $success = $?
  $exitCode = $LASTEXITCODE
  if (-not $success) {
    if ($exitCode -ne 0) {
      exit $exitCode
    }
    exit 1
  }
  if ($exitCode -ne 0) {
    exit $exitCode
  }
}

