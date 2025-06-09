#!/bin/bash

logdir=""
params=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --logdir)
            logdir="$2/RTX-6000-Ada"
            shift 2
            ;;
        --xyz_files)
            IFS=',' read -r -a xyz_files <<< "$2"
            shift 2
            ;;
        *)
            # Directly forward other parametrs to run-all-tests.sh
            params+=("$1" "$2")
            shift 2
            ;;
    esac
done

if [[ -z "$logdir" ]]; then
    echo "Error: --logdir parameter is required."
    exit 1
fi

# Default xyz files if not provided
if [ ${#xyz_files[@]} -eq 0 ]; then
    xyz_files=(
        "data/alkane/alkane_62.xyz"
        "data/alkane/alkane_122.xyz"
        "data/alkane/alkane_182.xyz"
        "data/graphene/graphene_36.xyz"
        "data/graphene/graphene_72.xyz"
        "data/water_clusters/005.xyz"
        "data/water_clusters/007.xyz"
        "data/organic/057_Tamoxifen.xyz"
        "data/organic/084_Sphingomyelin.xyz"
    )
fi

# Pass logdir and xyz_files to run-all-tests.sh
params+=("--logdir" "$logdir")
params+=("--xyz_files" "$(IFS=,; echo "${xyz_files[*]}")")

./run-all-tests.sh "${params[@]}"
