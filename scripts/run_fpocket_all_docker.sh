#!/usr/bin/env bash
set -euo pipefail

STRUCT_DIR="structures"

echo "=== Running fpocket in Docker on all PDBs in ${STRUCT_DIR} ==="

for pdb in "${STRUCT_DIR}"/*.pdb; do
    [ -e "$pdb" ] || continue

    base=$(basename "$pdb" .pdb)
    out_dir="${STRUCT_DIR}/${base}_out"

    if [ -d "$out_dir" ]; then
        echo "[SKIP] ${base}: ${out_dir} already exists."
        continue
    fi

    echo "[RUN] fpocket on ${base}..."
    docker run --rm \
        -v "$PWD/${STRUCT_DIR}":/data \
        fpocket/fpocket \
        fpocket -f "/data/${base}.pdb"

    echo "[OK] finished ${base}"
done

echo "=== All done running fpocket ==="