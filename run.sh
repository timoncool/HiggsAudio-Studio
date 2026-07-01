#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_FILE="$SCRIPT_DIR/app.py"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
VENV_PY="$VENV_DIR/bin/python"
REQ_HASH_FILE="$VENV_DIR/.requirements.sha256"

# uv provisions both the interpreter and the venv (no system python3 required).
UV_DIR="${UV_DIR:-$SCRIPT_DIR/.uv}"
UV_BIN="$UV_DIR/uv"
UV=""
PY_VERSION="${PY_VERSION:-3.12}"

RUN_AFTER_SETUP=1
DO_UPDATE=0
FORCE_REINSTALL=0
DO_DOWNLOAD=0

if [[ -t 1 ]]; then
  C_RESET='\033[0m'
  C_BOLD='\033[1m'
  C_BLUE='\033[34m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_RED='\033[31m'
else
  C_RESET=''
  C_BOLD=''
  C_BLUE=''
  C_GREEN=''
  C_YELLOW=''
  C_RED=''
fi

print_banner() {
  echo -e "${C_BOLD}${C_BLUE}========================================${C_RESET}"
  echo -e "${C_BOLD}${C_BLUE}  Higgs Audio Studio (Linux Launcher)${C_RESET}"
  echo -e "${C_BOLD}${C_BLUE}========================================${C_RESET}"
}

info() {
  echo -e "${C_BLUE}[INFO]${C_RESET} $*"
}

ok() {
  echo -e "${C_GREEN}[ OK ]${C_RESET} $*"
}

warn() {
  echo -e "${C_YELLOW}[WARN]${C_RESET} $*"
}

fail() {
  echo -e "${C_RED}[ERR ]${C_RESET} $*" >&2
}

usage() {
  cat <<'EOF'
Usage:
  ./run.sh [options]

Options:
  --update         Pull latest git changes and refresh dependencies, then run app.
  --update-only    Pull latest git changes and refresh dependencies, then exit.
  --reinstall      Force reinstall Python dependencies.
  --download       Pre-download all models with resume support, then run app.
  --download-only  Pre-download all models, then exit.
  --help, -h       Show this help.

Examples:
  ./run.sh
  ./run.sh --download       # first-time setup: download models then start
  ./run.sh --update
  ./run.sh --update-only
  ./run.sh --reinstall
EOF
}

on_error() {
  local line="$1"
  fail "Command failed on line ${line}."
  fail "Possible reasons: missing dependencies, incompatible CUDA/ROCm, or low VRAM."
}
trap 'on_error $LINENO' ERR

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command '$1' is not installed."
    exit 1
  fi
}

hash_requirements() {
  sha256sum "$REQ_FILE" | awk '{print $1}'
}

setup_runtime_dirs() {
  export TEMP="$SCRIPT_DIR/temp"
  export TMP="$SCRIPT_DIR/temp"
  export GRADIO_TEMP_DIR="$SCRIPT_DIR/temp"
  mkdir -p "$TEMP"

  export HF_HOME="$SCRIPT_DIR/models"
  export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/models"
  export TRANSFORMERS_CACHE="$SCRIPT_DIR/models"
  mkdir -p "$HF_HOME"

  export TORCH_HOME="$SCRIPT_DIR/models/torch"
  mkdir -p "$TORCH_HOME"

  export XDG_CACHE_HOME="$SCRIPT_DIR/cache"
  mkdir -p "$XDG_CACHE_HOME"

  # HF_TOKEN: из переменной окружения или из файла .hf_token рядом со скриптом
  local token_file="$SCRIPT_DIR/.hf_token"
  if [[ -z "${HF_TOKEN:-}" && -f "$token_file" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$token_file")"
    export HF_TOKEN
    info "HF_TOKEN loaded from .hf_token"
  elif [[ -n "${HF_TOKEN:-}" ]]; then
    info "HF_TOKEN loaded from environment"
  else
    warn "HF_TOKEN not set — downloads may be slow or rate-limited."
    warn "Put your token in .hf_token or export HF_TOKEN=hf_xxx before running."
  fi

  # Ускорённая загрузка моделей с HuggingFace через Xet
  export HF_XET_HIGH_PERFORMANCE=1

  if [[ -d "$SCRIPT_DIR/ffmpeg" ]]; then
    export PATH="$SCRIPT_DIR/ffmpeg:$PATH"
  fi

  export PYTHONIOENCODING="utf-8"
  export PYTHONUNBUFFERED="1"
}

ensure_uv() {
  if [[ -n "$UV" ]]; then
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
    info "Using uv: $UV"
    return
  fi
  if [[ -x "$UV_BIN" ]]; then
    UV="$UV_BIN"
    info "Using uv: $UV"
    return
  fi

  require_cmd curl
  require_cmd tar

  local arch tarball
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) tarball="uv-x86_64-unknown-linux-gnu.tar.gz" ;;
    aarch64|arm64) tarball="uv-aarch64-unknown-linux-gnu.tar.gz" ;;
    *) fail "Unsupported architecture for uv: $arch"; exit 1 ;;
  esac

  info "Downloading uv ($tarball)"
  mkdir -p "$UV_DIR"
  curl -LsSf "https://github.com/astral-sh/uv/releases/latest/download/$tarball" -o "$UV_DIR/uv.tar.gz"
  tar -xzf "$UV_DIR/uv.tar.gz" -C "$UV_DIR" --strip-components=1
  rm -f "$UV_DIR/uv.tar.gz"

  if [[ ! -x "$UV_BIN" ]]; then
    fail "Failed to install uv"
    exit 1
  fi
  UV="$UV_BIN"
  ok "uv installed"
}

create_or_activate_venv() {
  ensure_uv

  if [[ ! -x "$VENV_PY" ]]; then
    info "Creating virtual environment in $VENV_DIR (uv, Python $PY_VERSION)"
    "$UV" venv --python "$PY_VERSION" "$VENV_DIR"
    ok "Virtual environment created"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
}

install_or_update_deps() {
  if [[ ! -f "$REQ_FILE" ]]; then
    fail "requirements.txt not found: $REQ_FILE"
    exit 1
  fi

  local current_hash installed_hash
  current_hash="$(hash_requirements)"
  installed_hash="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

  ensure_uv

  info "Updating pip tooling"
  "$UV" pip install --python "$VENV_PY" --upgrade pip setuptools wheel >/dev/null

  if [[ "$FORCE_REINSTALL" -eq 1 || "$DO_UPDATE" -eq 1 || "$current_hash" != "$installed_hash" ]]; then
    info "Installing Python dependencies"
    "$UV" pip install --python "$VENV_PY" -r "$REQ_FILE"
    printf '%s\n' "$current_hash" > "$REQ_HASH_FILE"
    ok "Dependencies are up to date"
  else
    ok "Dependencies unchanged, skipping install"
  fi
}

update_code() {
  require_cmd git

  if [[ -d "$SCRIPT_DIR/.git" ]]; then
    info "Pulling latest changes"
    git pull --ff-only || git pull
    ok "Repository updated"
  else
    warn "No .git folder found, skipping git pull"
  fi
}

download_models() {
  info "Downloading models (verbose debug enabled)..."
  "$VENV_PY" - <<'PYEOF'
import os, sys, logging, socket

# Debug logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# Set socket timeout to catch hanging connections
socket.setdefaulttimeout(30)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[download] huggingface_hub not installed — run with --reinstall first", flush=True)
    sys.exit(1)

cache_dir = os.environ.get("HF_HOME", "models")
token = os.environ.get("HF_TOKEN") or None

if token:
    print(f"[download] Using HF_TOKEN (first 10 chars: {token[:10]}...)", flush=True)
else:
    print("[download] No HF_TOKEN — using anonymous (may be rate-limited)", flush=True)

repos = [
    "multimodalart/higgs-audio-v3-tts-4b-transformers",
    "UsefulSensors/moonshine-base",
]

for repo in repos:
    print(f"\n[download] Starting: {repo}", flush=True)
    try:
        path = snapshot_download(
            repo_id=repo,
            cache_dir=cache_dir,
            token=token,
            ignore_patterns=["*.bin"],
            user_agent="higgs-audio-studio/linux",
        )
        print(f"[download] ✓ Done: {repo} → {path}", flush=True)
    except Exception as e:
        print(f"[download] ✗ ERROR {repo}: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n[download] All models downloaded successfully!", flush=True)
PYEOF
}

run_app() {
  if [[ ! -f "$APP_FILE" ]]; then
    fail "app.py not found: $APP_FILE"
    exit 1
  fi

  info "Starting application"
  "$VENV_PY" "$APP_FILE"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --update)
        DO_UPDATE=1
        shift
        ;;
      --update-only)
        DO_UPDATE=1
        RUN_AFTER_SETUP=0
        shift
        ;;
      --reinstall)
        FORCE_REINSTALL=1
        shift
        ;;
      --download)
        DO_DOWNLOAD=1
        shift
        ;;
      --download-only)
        DO_DOWNLOAD=1
        RUN_AFTER_SETUP=0
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        fail "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

main() {
  print_banner
  parse_args "$@"

  setup_runtime_dirs
  create_or_activate_venv

  if [[ "$DO_UPDATE" -eq 1 ]]; then
    update_code
  fi

  install_or_update_deps

  if [[ "$DO_DOWNLOAD" -eq 1 ]]; then
    download_models
  fi

  if [[ "$RUN_AFTER_SETUP" -eq 1 ]]; then
    run_app
  else
    ok "Update completed"
  fi
}

main "$@"
