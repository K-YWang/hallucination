## useless！！


#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# download_data.sh – One‑click fetcher for Stage‑1/2 datasets (mainland‑friendly)
# -----------------------------------------------------------------------------
# ▸ Muse‑512 on PartiPrompts (P2)               – Hugging Face dataset repo
# ▸ T2I‑CompBench & CompBench++ sample images   – GitHub repo samples folder
# ▸ T2I‑FactualBench concept images             – GitHub repo + HF tarball via
#                                                project script
# -----------------------------------------------------------------------------
# 
#   1. 设置 Hugging Face 镜像:
#         export HF_ENDPOINT=https://hf-mirror.com
#   2. 设置 GitHub 代理前缀 (可换为 ghproxy.com, ghproxy.cc 等):
#         export GH_PROXY=https://ghproxy.com/
# -----------------------------------------------------------------------------
export HF_ENDPOINT=https://hf-mirror.com
set -Eeuo pipefail

ARIA2_THREADS="${ARIA2_THREADS:-16}"
DATA_ROOT="${DATA_ROOT:-./data/raw}"
mkdir -p "$DATA_ROOT"

# ------------------------------
# helper: download HF dataset
# ------------------------------
function download_hf_dataset() {
  local repo_id="$1"  # e.g. diffusers-parti-prompts/muse512
  local dest_dir="$2"
  echo -e "\n>>> [HF] $repo_id → $dest_dir"
  if ! command -v huggingface-cli &>/dev/null; then
    echo "huggingface-cli not found. Run:  pip install -U huggingface-hub  \n" >&2; exit 1
  fi
  huggingface-cli download "$repo_id" --repo-type dataset \
    --local-dir "$dest_dir" --local-dir-use-symlinks False --resume-download
}

# ------------------------------
# helper: download GitHub repo (tarball via proxy + aria2c)
# ------------------------------
function download_github_repo() {
  local repo_url="$1"   # e.g. https://github.com/Karine-Huang/T2I-CompBench
  local dest_dir="$2"
  local branch="${3:-main}"
  echo -e "\n>>> [GitHub] $repo_url ($branch) → $dest_dir"
  mkdir -p "$dest_dir"
  local tar_link="${repo_url}/archive/refs/heads/${branch}.tar.gz"
  local proxy_prefix="${GH_PROXY:-}"   # allow empty (direct)
  local final_link="${proxy_prefix}${tar_link}"
  echo "Fetching tarball via: $final_link"
  aria2c -x "$ARIA2_THREADS" -s "$ARIA2_THREADS" -c "$final_link" -o repo.tar.gz || {
    echo "aria2c download failed. Trying curl..." >&2
    curl -L "$final_link" -o repo.tar.gz
  }
  tar -xzf repo.tar.gz --strip-components=1 -C "$dest_dir"
  rm repo.tar.gz
}

# ============================================================================
# 1) Muse‑512 ▸ PartiPrompts (P2)
# ============================================================================
MUSE_DIR="$DATA_ROOT/p2_muse512"
[ -d "$MUSE_DIR/.completed" ] || {
  download_hf_dataset "diffusers-parti-prompts/muse512" "$MUSE_DIR"
  touch "$MUSE_DIR/.completed"
}

# ============================================================================
# 2) T2I‑CompBench & CompBench++ baseline images
# ============================================================================
COMPBENCH_DIR="$DATA_ROOT/t2i_compbench"
[ -d "$COMPBENCH_DIR/.completed" ] || {
  download_github_repo "https://github.com/Karine-Huang/T2I-CompBench" "$COMPBENCH_DIR" "main"
  echo "Extracting example images only → $COMPBENCH_DIR/examples/samples …"
  # 可按需移动或过滤图片; 这里只留下 samples 文件夹
  find "$COMPBENCH_DIR" -maxdepth 1 -type f ! -name "LICENSE" -exec rm -f {} +
  touch "$COMPBENCH_DIR/.completed"
}

# ============================================================================
# 3) T2I‑FactualBench concept images (via provided download.py)
# ============================================================================
FACT_DIR="$DATA_ROOT/t2i_factualbench"
[ -d "$FACT_DIR/.completed" ] || {
  download_github_repo "https://github.com/Safeoffellow/T2I-FactualBench" "$FACT_DIR" "main"
  echo -e "\n>>> Running project download script (may re‑hit HF) …"
  cd "$FACT_DIR/data" && python download.py --output_dir "$FACT_DIR/concept_images" && cd -
  touch "$FACT_DIR/.completed"
}

echo -e "\n✅  All datasets downloaded to $DATA_ROOT"
