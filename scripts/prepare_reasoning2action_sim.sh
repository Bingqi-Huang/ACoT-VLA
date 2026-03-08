#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 022
shopt -s nullglob

# Usage:
#   ./prepare_reasoning2action_sim.sh [SRC_ROOT] [DST_ROOT] [START_INDEX] [MAX_TASKS]
#
# Examples:
#   ./prepare_reasoning2action_sim.sh
#   ./prepare_reasoning2action_sim.sh "$HOME/Datasets/hf_data/Reasoning2Action-Sim" "$HOME/SharedData/lerobot/Reasoning2Action-Sim"
#   ./prepare_reasoning2action_sim.sh "$HOME/Datasets/hf_data/Reasoning2Action-Sim" "$HOME/SharedData/lerobot/Reasoning2Action-Sim" 0 2
#   ./prepare_reasoning2action_sim.sh "$HOME/Datasets/hf_data/Reasoning2Action-Sim" "/media/bingqi/SSD/Research/AGIBOT_DATA/lerobot/Reasoning2Action-Sim" 2 2

SRC_ROOT="${1:-$HOME/Datasets/hf_data/Reasoning2Action-Sim}"
DST_ROOT="${2:-$HOME/SharedData/lerobot/Reasoning2Action-Sim}"
START_INDEX="${3:-0}"
MAX_TASKS="${4:--1}"   # -1 means no limit

# For meta/data pre-check only
MIN_MARGIN_BYTES=$((10 * 1024 * 1024 * 1024))

mkdir -p -- "$DST_ROOT"
TMP_ROOT="$(mktemp -d "${DST_ROOT%/}/.extract_tmp.XXXXXX")"

cleanup() {
  rm -rf -- "$TMP_ROOT"
}
trap cleanup EXIT

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf '[%s] ERROR: %s\n' "$(date '+%F %T')" "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_cmd bash
require_cmd find
require_cmd sort
require_cmd tar
require_cmd python3
require_cmd mktemp
require_cmd rm
require_cmd mv
require_cmd cat
require_cmd df
require_cmd awk

[[ "$START_INDEX" =~ ^[0-9]+$ ]] || die "START_INDEX must be a non-negative integer"
[[ "$MAX_TASKS" =~ ^-?[0-9]+$ ]] || die "MAX_TASKS must be an integer (-1 means no limit)"
(( MAX_TASKS == -1 || MAX_TASKS >= 0 )) || die "MAX_TASKS must be -1 or a non-negative integer"

[[ -d "$SRC_ROOT" ]] || die "SRC_ROOT does not exist or is not a directory: $SRC_ROOT"

bytes_to_human() {
  python3 - "$1" <<'PY'
import sys
n = int(sys.argv[1])
units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
v = float(n)
for u in units:
    if v < 1024.0 or u == units[-1]:
        if u == "B":
            print(f"{int(v)} {u}")
        else:
            print(f"{v:.2f} {u}")
        break
    v /= 1024.0
PY
}

get_fs_avail_bytes() {
  local path="$1"
  local avail
  avail="$(df -B1 --output=avail "$path" | awk 'NR==2 {print $1}')"
  [[ "$avail" =~ ^[0-9]+$ ]] || die "Failed to parse available space for $path"
  echo "$avail"
}

# Collect split parts and require contiguous numeric suffixes:
#   prefix.tar.gz.000, .001, .002, ...
get_parts() {
  local src_dir="$1"
  local prefix="$2"
  local -n _out_ref="$3"

  mapfile -t _out_ref < <(
    find "$src_dir" -maxdepth 1 -type f -name "${prefix}.tar.gz.*" -print | sort -V
  )

  if [[ "${#_out_ref[@]}" -eq 0 ]]; then
    return 1
  fi

  local i part suffix n
  for i in "${!_out_ref[@]}"; do
    part="${_out_ref[$i]}"
    suffix="${part##*.}"
    [[ "$suffix" =~ ^[0-9]+$ ]] || die "Invalid part suffix: $part"
    n=$((10#$suffix))
    if (( n != i )); then
      die "Missing or non-contiguous parts for ${prefix} in ${src_dir}. Expected index $(printf '%03d' "$i"), got ${suffix}"
    fi
  done
}

validate_existing_target() {
  local out_dir="$1"
  local prefix="$2"

  case "$prefix" in
    meta)
      [[ -d "$out_dir/meta" && -f "$out_dir/meta/info.json" ]]
      ;;
    data)
      [[ -d "$out_dir/data" ]]
      ;;
    videos)
      [[ -d "$out_dir/videos" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

# Pre-analysis for small prefixes only (meta/data):
#   - stream-read concatenated split parts
#   - validate tar members for safety
#   - sum sizes of regular files
# Output: prints total payload bytes to stdout
analyze_archive_stream() {
  local prefix="$1"
  shift

  python3 - "$prefix" "$@" <<'PY'
import posixpath
import sys
import tarfile

prefix = sys.argv[1]
parts = sys.argv[2:]

class ConcatReader:
    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        self.fp = None
        self._open_next()

    def _open_next(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None
        while self.idx < len(self.paths):
            self.fp = open(self.paths[self.idx], 'rb')
            self.idx += 1
            return
        self.fp = None

    def read(self, size=-1):
        if self.fp is None:
            return b''
        if size is None or size < 0:
            chunks = []
            while self.fp is not None:
                data = self.fp.read()
                if data:
                    chunks.append(data)
                self._open_next()
            return b''.join(chunks)

        out = bytearray()
        remain = size
        while remain > 0 and self.fp is not None:
            chunk = self.fp.read(remain)
            if chunk:
                out.extend(chunk)
                remain -= len(chunk)
            else:
                self._open_next()
        return bytes(out)

    def close(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None

def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

reader = ConcatReader(parts)
seen = set()
total_regular_bytes = 0
member_count = 0

try:
    tf = tarfile.open(fileobj=reader, mode='r|gz')
except Exception as e:
    reader.close()
    fail(f"Unable to open concatenated archive stream: {e}")

try:
    for m in tf:
        member_count += 1
        name = m.name

        if not name:
            fail("Archive contains an empty member name")
        if name.startswith('/'):
            fail(f"Archive contains an absolute path: {name}")

        norm = posixpath.normpath(name)
        if norm in ("", ".", ".."):
            fail(f"Archive contains invalid normalized path: {name!r} -> {norm!r}")

        if any(p == '..' for p in norm.split('/')):
            fail(f"Archive contains path traversal: {name}")

        if norm != prefix and not norm.startswith(prefix + '/'):
            fail(f"Archive member is outside expected top-level directory '{prefix}/': {name}")

        if m.issym():
            fail(f"Archive contains a symbolic link, rejected for safety: {name}")
        if m.islnk():
            fail(f"Archive contains a hard link, rejected for safety: {name}")
        if m.isdev():
            fail(f"Archive contains a device file, rejected for safety: {name}")
        if m.isfifo():
            fail(f"Archive contains a FIFO, rejected for safety: {name}")

        if not (m.isfile() or m.isdir()):
            fail(f"Archive contains unsupported member type: {name}")

        if norm in seen:
            fail(f"Archive contains duplicate normalized path: {name}")
        seen.add(norm)

        if m.isfile():
            if m.size < 0:
                fail(f"Archive contains a file with negative size: {name}")
            total_regular_bytes += m.size

    if member_count == 0:
        fail("Archive is empty")

    if prefix not in seen and not any(p.startswith(prefix + '/') for p in seen):
        fail(f"Archive does not contain expected top-level directory '{prefix}/'")

    print(total_regular_bytes)
finally:
    try:
        tf.close()
    except Exception:
        pass
    reader.close()
PY
}

extract_archive_stream() {
  local stage_root="$1"
  shift
  cat -- "$@" | tar \
    --extract \
    --gzip \
    --file - \
    --directory "$stage_root" \
    --no-same-owner \
    --no-same-permissions \
    --delay-directory-restore \
    --numeric-owner
}

is_task_complete() {
  local out_dir="$1"
  [[ -f "$out_dir/meta/info.json" && -d "$out_dir/data" && -d "$out_dir/videos" ]]
}

required_margin_bytes() {
  local payload_bytes="$1"
  local five_percent=$(( payload_bytes / 20 ))
  if (( five_percent > MIN_MARGIN_BYTES )); then
    echo "$five_percent"
  else
    echo "$MIN_MARGIN_BYTES"
  fi
}

extract_stream() {
  local src_dir="$1"
  local task_name="$2"
  local prefix="$3"
  local out_dir="$4"

  local target_path="$out_dir/$prefix"
  local parts=()

  if get_parts "$src_dir" "$prefix" parts; then
    :
  else
    if validate_existing_target "$out_dir" "$prefix"; then
      log "Task ${task_name}: ${prefix} already present at ${target_path}, and no source parts remain. Skipping."
      return 0
    else
      die "Missing ${prefix}.tar.gz.* under ${src_dir}, and target ${target_path} is not already valid."
    fi
  fi

  if [[ -e "$target_path" ]]; then
    die "Both source parts and existing target are present for ${task_name}/${prefix}. Refusing to overwrite: ${target_path}"
  fi

  # Keep pre-analysis for small prefixes only.
  # Skip videos pre-analysis to avoid reading the whole archive twice.
  if [[ "$prefix" != "videos" ]]; then
    log "Task ${task_name}: analyzing ${prefix} from ${#parts[@]} part(s)"
    local payload_bytes margin_bytes need_bytes avail_bytes
    payload_bytes="$(analyze_archive_stream "$prefix" "${parts[@]}")"
    [[ "$payload_bytes" =~ ^[0-9]+$ ]] || die "Internal error: invalid payload size for ${task_name}/${prefix}: $payload_bytes"

    margin_bytes="$(required_margin_bytes "$payload_bytes")"
    need_bytes=$(( payload_bytes + margin_bytes ))
    avail_bytes="$(get_fs_avail_bytes "$DST_ROOT")"
    [[ "$avail_bytes" =~ ^[0-9]+$ ]] || die "Failed to determine free space for $DST_ROOT"

    log "Task ${task_name}: ${prefix} payload estimate = $(bytes_to_human "$payload_bytes"), required free = $(bytes_to_human "$need_bytes"), available = $(bytes_to_human "$avail_bytes")"

    if (( avail_bytes < need_bytes )); then
      die "Not enough free space for ${task_name}/${prefix}. Need at least $(bytes_to_human "$need_bytes"), have $(bytes_to_human "$avail_bytes"). Refusing to start extraction."
    fi
  else
    local avail_bytes
    avail_bytes="$(get_fs_avail_bytes "$DST_ROOT")"
    log "Task ${task_name}: skipping full pre-analysis for videos to save time; available free space = $(bytes_to_human "$avail_bytes")"
  fi

  local stage_dir
  stage_dir="$(mktemp -d "${TMP_ROOT}/${task_name}.${prefix}.stage.XXXXXX")"
  mkdir -p -- "$stage_dir/root"

  log "Task ${task_name}: extracting ${prefix} to staging"
  extract_archive_stream "$stage_dir/root" "${parts[@]}"

  [[ -e "$stage_dir/root/$prefix" ]] || die "Archive extracted but expected path missing: $stage_dir/root/$prefix"

  case "$prefix" in
    meta)
      [[ -f "$stage_dir/root/meta/info.json" ]] || die "meta extracted, but meta/info.json is missing"
      ;;
    data)
      [[ -d "$stage_dir/root/data" ]] || die "data extracted, but data/ is missing"
      ;;
    videos)
      [[ -d "$stage_dir/root/videos" ]] || die "videos extracted, but videos/ is missing"
      ;;
    *)
      die "Internal error: unsupported prefix ${prefix}"
      ;;
  esac

  log "Task ${task_name}: moving ${prefix} into final destination"
  mv -- "$stage_dir/root/$prefix" "$target_path"

  validate_existing_target "$out_dir" "$prefix" || die "Post-move validation failed for ${target_path}"

  log "Task ${task_name}: ${prefix} completed successfully; deleting source parts"
  rm -f -- "${parts[@]}"

  rm -rf -- "$stage_dir"
  log "Task ${task_name}: ${prefix} done"
}

main() {
  local task_dirs=()
  local task_dir task_name out_dir
  local idx=0
  local selected=0

  mapfile -t task_dirs < <(
    find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d -print | sort
  )

  log "Found ${#task_dirs[@]} task(s) under $SRC_ROOT"
  log "Selection window: START_INDEX=$START_INDEX, MAX_TASKS=$MAX_TASKS"

  for task_dir in "${task_dirs[@]}"; do
    if (( idx < START_INDEX )); then
      ((idx += 1))
      continue
    fi

    if (( MAX_TASKS >= 0 && selected >= MAX_TASKS )); then
      break
    fi

    task_name="$(basename "$task_dir")"
    out_dir="$DST_ROOT/$task_name"
    mkdir -p -- "$out_dir"

    if is_task_complete "$out_dir"; then
      log "Skipping ${task_name}: already complete at ${out_dir}"
      ((idx += 1))
      ((selected += 1))
      continue
    fi

    log "===== Processing task [$idx]: ${task_name} ====="

    extract_stream "$task_dir" "$task_name" "meta"   "$out_dir"
    extract_stream "$task_dir" "$task_name" "data"   "$out_dir"
    extract_stream "$task_dir" "$task_name" "videos" "$out_dir"

    is_task_complete "$out_dir" || die "Task ${task_name} finished stream extraction, but final structure is incomplete"

    log "===== Task complete: ${task_name} ====="

    ((idx += 1))
    ((selected += 1))
  done

  echo
  log "Processed $selected task(s) starting from index $START_INDEX"
  log "LeRobot datasets are under: $DST_ROOT"
}

main "$@"