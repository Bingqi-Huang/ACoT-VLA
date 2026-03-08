#!/usr/bin/env bash

set -euo pipefail

SRC_ROOT="${1:-$HOME/Datasets/hf_data/Reasoning2Action-Sim}"
DST_ROOT="${2:-$HOME/Datasets/lerobot/Reasoning2Action-Sim}"

extract_stream() {
  local src_dir="$1"
  local prefix="$2"
  local dst_dir="$3"

  mapfile -t parts < <(find "$src_dir" -maxdepth 1 -type f -name "${prefix}.tar.gz.*" | sort)
  if [[ "${#parts[@]}" -eq 0 ]]; then
    echo "Missing ${prefix}.tar.gz.* under ${src_dir}" >&2
    return 1
  fi

  echo "Extracting ${prefix} for $(basename "$src_dir")"
  if cat "${parts[@]}" | tar -xzf - -C "$dst_dir"; then
    echo "Successfully extracted ${prefix}, now deleting source files..."
    rm "${parts[@]}"
  else
    echo "Error extracting ${prefix} from ${src_dir}. Source files will not be deleted." >&2
    return 1
  fi
}

mkdir -p "$DST_ROOT"

for task_dir in "$SRC_ROOT"/*; do
  [[ -d "$task_dir" ]] || continue

  task_name="$(basename "$task_dir")"
  out_dir="$DST_ROOT/$task_name"
  mkdir -p "$out_dir"

  if [[ -f "$out_dir/meta/info.json" && -d "$out_dir/data" && -d "$out_dir/videos" ]]; then
    echo "Skipping ${task_name}: already extracted at ${out_dir}"
    continue
  fi

  extract_stream "$task_dir" "meta" "$out_dir"
  extract_stream "$task_dir" "data" "$out_dir"
  extract_stream "$task_dir" "videos" "$out_dir"
done

echo
echo "Extraction complete."
echo "LeRobot datasets are under: $DST_ROOT"
