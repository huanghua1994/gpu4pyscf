#!/bin/bash

time_now=$(date +%Y%m%d_%H%M%S)
logdir="logs_$time_now"
declare -a xyz_files

declare -a basis_set=("6-311gss" "def2-tzvpp")
declare -a calc_types=("DFT")
declare -a mxp_df_levels=(0 1 2)
max_cycle=50
conv_tol=1e-9
run_verbose=5
xc="b3lyp"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --logdir)
            logdir="$2"
            shift 2
            ;;
        --xyz_files)
            IFS=',' read -r -a xyz_files <<< "$2"
            shift 2
            ;;
        --max_cycles)
            max_cycle="$2"
            shift 2
            ;;
        --conv_tol)
            conv_tol="$2"
            shift 2
            ;;
        --run_verbose)
            run_verbose="$2"
            shift 2
            ;;
        --xc)
            xc="$2"
            shift 2
            ;;
        --basis)
            IFS=',' read -r -a basis_set <<< "$2"
            shift 2
            ;;
        --calc_type)
            IFS=',' read -r -a calc_types <<< "$2"
            shift 2
            ;;
        --mxp_df_level)
            IFS=',' read -r -a mxp_df_levels <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$logdir"
echo "Logs will be stored in directory $logdir"

lscpu       > "$logdir/test-machine.txt"
free -h    >> "$logdir/test-machine.txt"
nvidia-smi >> "$logdir/test-machine.txt"

if [ ${#xyz_files[@]} -eq 0 ]; then
    xyz_files=(
        "data/alkane/alkane_32.xyz"
        "data/alkane/alkane_62.xyz"
        "data/alkane/alkane_122.xyz"
        "data/alkane/alkane_182.xyz"
        "data/graphene/graphene_36.xyz"
        "data/graphene/graphene_72.xyz"
        "data/graphene/graphene_120.xyz"
        "data/water_clusters/005.xyz"
        "data/water_clusters/007.xyz"
        "data/water_clusters/008.xyz"
        "data/organic/057_Tamoxifen.xyz"
        "data/organic/095_Azadirachtin.xyz"
        "data/organic/168_Valinomycin.xyz"
    )
fi

# Warn-up running
echo "Warm-up running..."
for mxp_df_level in "${mxp_df_levels[@]}"; do
    python3 test-mxp-df.py data/alkane/alkane_32.xyz 6-311gss DFT "--mxp_df_level" "$mxp_df_level" &> /tmp/mxp-df-warmup.log
done

# Run all tests
echo "Start running tests..."
for xyz_file in "${xyz_files[@]}"; do
    base_name=$(basename "$xyz_file" .xyz)
    for basis in "${basis_set[@]}"; do
        for calc_type in "${calc_types[@]}"; do
            for mxp_df_level in "${mxp_df_levels[@]}"; do
                log_file="$logdir/${base_name}_${basis}_${calc_type}_mxp-df-level${mxp_df_level}.log"

                echo Running: python3 test-mxp-df.py "$xyz_file" "$basis" "$calc_type" \
                    "--xc" $xc "--mxp_df_level" "$mxp_df_level" "--max_cycle" "$max_cycle" \
                    "--run_verbose" $run_verbose "--conv_tol" $conv_tol, log to "$log_file"

                python3 test-mxp-df.py "$xyz_file" "$basis" "$calc_type" \
                    "--xc" $xc "--mxp_df_level" "$mxp_df_level" "--max_cycle" "$max_cycle" \
                    "--run_verbose" $run_verbose "--conv_tol" $conv_tol &> "$log_file"
            done
        done
    done
done
