#!/usr/bin/env bash

set -euo pipefail

API_BASE="${API_BASE:-https://api.wqyunpan.com/yp}"
ORIGIN="${ORIGIN:-https://read.wqyunpan.com}"
REFERER="${REFERER:-https://read.wqyunpan.com/}"
USER_AGENT="${USER_AGENT:-Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36}"

TOKEN="${WQYP_TOKEN:-}"
BOOK_ID=""
RESOURCE_TYPE="kz"
QRCODE_ID=""
FILE_ID=""
OUTDIR="materials/videos"
DRY_RUN=0
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  download_wqyunpan_videos.sh --token TOKEN [options]

Required:
  --token TOKEN           WQYunpan yp-token

Selection:
  --book-id ID            Download all videos from book/{bookId}/{type}
  --qrcode-id ID          Download all videos from book/{qrcodeId}/file
  --file-id ID            Download a single file via qrcode/file/{fileId}/preview

Options:
  --type TYPE             Resource type for book listing, default: kz
  --outdir DIR            Output directory, default: materials/videos
  --dry-run               Print resolved files and URLs without downloading
  --force                 Re-download even if target file exists
  -h, --help              Show this help

Examples:
  download_wqyunpan_videos.sh --token "$WQYP_TOKEN" --book-id 1246181 --type kz
  download_wqyunpan_videos.sh --token "$WQYP_TOKEN" --qrcode-id 442171 --outdir materials/videos
  download_wqyunpan_videos.sh --token "$WQYP_TOKEN" --file-id 515322
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

fail() {
  log "ERROR: $*"
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

sanitize_name() {
  local name="$1"
  name="${name//$'\r'/}"
  name="${name//$'\n'/ }"
  name="${name//\//_}"
  name="${name//:/_}"
  name="${name//\\/_}"
  printf '%s' "$name"
}

pick_unique_path() {
  local dir="$1"
  local raw_name="$2"
  local name stem ext candidate idx

  name="$(sanitize_name "$raw_name")"
  candidate="$dir/$name"

  if [[ ! -e "$candidate" || "$FORCE" -eq 1 ]]; then
    printf '%s\n' "$candidate"
    return
  fi

  stem="$name"
  ext=""
  if [[ "$name" == *.* ]]; then
    stem="${name%.*}"
    ext=".${name##*.}"
  fi

  idx=1
  while :; do
    candidate="$dir/${stem}_${idx}${ext}"
    if [[ ! -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
    idx=$((idx + 1))
  done
}

api_get() {
  local path="$1"
  local response

  response="$(curl \
    --silent \
    --show-error \
    --fail \
    -H "accept: application/json, text/plain, */*" \
    -H "origin: ${ORIGIN}" \
    -H "referer: ${REFERER}" \
    -H "user-agent: ${USER_AGENT}" \
    -H "yp-token: ${TOKEN}" \
    "${API_BASE}/${path}")"

  if [[ "$(jq -r '.code // empty' <<<"$response")" != "200" ]]; then
    fail "API request failed for ${path}: $(jq -r '.message // "unknown error"' <<<"$response")"
  fi

  printf '%s\n' "$response"
}

download_file() {
  local file_id="$1"
  local preview_response filename url output_path

  preview_response="$(api_get "qrcode/file/${file_id}/preview")"
  filename="$(jq -r '.data.filename // empty' <<<"$preview_response")"
  url="$(jq -r '.data.url // empty' <<<"$preview_response")"

  [[ -n "$filename" ]] || fail "No filename returned for fileId=${file_id}"
  [[ -n "$url" ]] || fail "No preview URL returned for fileId=${file_id}"

  mkdir -p "$OUTDIR"
  output_path="$(pick_unique_path "$OUTDIR" "$filename")"

  if [[ "$FORCE" -eq 0 && -e "$output_path" ]]; then
    log "Skip existing file: $output_path"
    return
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\t%s\n' "$filename" "$url"
    return
  fi

  log "Downloading fileId=${file_id} -> ${output_path}"
  curl -L --fail --retry 3 --retry-delay 2 --output "$output_path" "$url"
}

download_qrcode_videos() {
  local qrcode_id="$1"
  local file_response file_ids count

  log "Fetching file list for qrcodeId=${qrcode_id}"
  file_response="$(api_get "book/${qrcode_id}/file")"
  file_ids="$(jq -r '.data[] | select(.type == "video") | .id' <<<"$file_response")"

  if [[ -z "$file_ids" ]]; then
    log "No video files found for qrcodeId=${qrcode_id}"
    return
  fi

  count=0
  while IFS= read -r current_file_id; do
    [[ -n "$current_file_id" ]] || continue
    download_file "$current_file_id"
    count=$((count + 1))
  done <<<"$file_ids"

  log "Processed ${count} video file(s) for qrcodeId=${qrcode_id}"
}

download_book_videos() {
  local book_id="$1"
  local list_response qrcode_ids count

  log "Fetching resource list for bookId=${book_id}, type=${RESOURCE_TYPE}"
  list_response="$(api_get "book/${book_id}/${RESOURCE_TYPE}")"
  qrcode_ids="$(jq -r '.data[] | .id' <<<"$list_response")"

  if [[ -z "$qrcode_ids" ]]; then
    log "No resource entries found for bookId=${book_id}, type=${RESOURCE_TYPE}"
    return
  fi

  count=0
  while IFS= read -r current_qrcode_id; do
    [[ -n "$current_qrcode_id" ]] || continue
    download_qrcode_videos "$current_qrcode_id"
    count=$((count + 1))
  done <<<"$qrcode_ids"

  log "Processed ${count} qrcode resource entrie(s)"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --token)
        [[ $# -ge 2 ]] || fail "--token requires a value"
        TOKEN="$2"
        shift 2
        ;;
      --book-id)
        [[ $# -ge 2 ]] || fail "--book-id requires a value"
        BOOK_ID="$2"
        shift 2
        ;;
      --qrcode-id)
        [[ $# -ge 2 ]] || fail "--qrcode-id requires a value"
        QRCODE_ID="$2"
        shift 2
        ;;
      --file-id)
        [[ $# -ge 2 ]] || fail "--file-id requires a value"
        FILE_ID="$2"
        shift 2
        ;;
      --type)
        [[ $# -ge 2 ]] || fail "--type requires a value"
        RESOURCE_TYPE="$2"
        shift 2
        ;;
      --outdir)
        [[ $# -ge 2 ]] || fail "--outdir requires a value"
        OUTDIR="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --force)
        FORCE=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        fail "Unknown argument: $1"
        ;;
    esac
  done
}

main() {
  need_cmd curl
  need_cmd jq

  parse_args "$@"

  [[ -n "$TOKEN" ]] || fail "Missing --token or WQYP_TOKEN"

  local selectors=0
  [[ -n "$BOOK_ID" ]] && selectors=$((selectors + 1))
  [[ -n "$QRCODE_ID" ]] && selectors=$((selectors + 1))
  [[ -n "$FILE_ID" ]] && selectors=$((selectors + 1))

  [[ "$selectors" -eq 1 ]] || fail "Choose exactly one of --book-id, --qrcode-id, or --file-id"

  if [[ -n "$FILE_ID" ]]; then
    download_file "$FILE_ID"
  elif [[ -n "$QRCODE_ID" ]]; then
    download_qrcode_videos "$QRCODE_ID"
  else
    download_book_videos "$BOOK_ID"
  fi
}

main "$@"
