#!/bin/bash

# default values
build_type=Release
looped=false
pattern=
negative=

# Simple argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            build_type=Debug
            echo "Debug mode enabled"
            shift
            ;;
        -l|--looped)
            looped=true
            echo "Looped mode enabled"
            shift
            ;;
        -p|--pattern)
            if [[ -n "$2" ]]; then
                pattern="$2"  # Store pattern without -path prefix
                echo "Pattern set to: $2"
                shift 2
            else
                echo "Error: Pattern argument requires a value" >&2
                exit 1
            fi
            ;;
        --negative)
            negative="!"
            echo "Negative mode enabled"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            shift
            ;;
    esac
done

echo "Final values:"
echo "build_type: $build_type"
echo "looped: $looped"
echo "pattern: $pattern"
echo "negative: $negative"

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $build_type == "Debug" ];
then
    echo "Will run the samples using the debug version of shards"
else
    echo "Will run the samples using the release version of shards"
fi

# Create a temporary directory for status files
tmp_dir=$(mktemp -d)
declare -a failed_samples

# Function to run a single sample
run_sample() {
    local sample=$1
    local sample_base=$(basename "$sample")
    echo -e "\n\033[1;34mRunning sample: $sample\033[0m"
    rm -f "$sample.log"
    if $script_dir/build/$build_type/shards new $script_dir/docs/samples/run-sample.shs looped:$looped file:"$sample" 2>"$sample.log"; then
        echo -e "\033[1;32m✓ Success: $sample\033[0m"
        echo "success" > "$tmp_dir/${sample//\//_}.status"
    else
        echo -e "\033[1;31m✗ Failed: $sample\033[0m"
        echo "failed" > "$tmp_dir/${sample//\//_}.status"
        echo "$sample" >> "$tmp_dir/failed_list"
        if [ -f "$sample.log" ]; then
            echo -e "\033[1;31mError output:\033[0m"
            cat "$sample.log"
        fi
    fi
}
export -f run_sample
export looped build_type script_dir tmp_dir

# execute commands
pushd $script_dir/docs/samples
echo -e "\n\033[1;36mStarting sample execution...\033[0m\n"

# Find samples and run them in parallel (4 at a time)
find shards -name '*.shs' $negative -path "$pattern" | xargs -P 4 -I {} bash -c 'run_sample "{}"'

# Collect results
echo -e "\n\033[1;36mExecution Summary:\033[0m"
total=0
failed=0

# Count total tests
total=$(find "$tmp_dir" -name "*.status" | wc -l)

# Get failed tests
if [ -f "$tmp_dir/failed_list" ]; then
    mapfile -t failed_samples < "$tmp_dir/failed_list"
    failed=${#failed_samples[@]}
fi

# Print summary
echo -e "\nTotal samples: $total"
if [ $failed -eq 0 ]; then
    echo -e "\033[1;32mAll samples completed successfully!\033[0m"
else
    echo -e "\033[1;31mFailed samples: $failed\033[0m"
    echo -e "\033[1;31mFailed tests:\033[0m"
    for sample in "${failed_samples[@]}"; do
        echo -e "\033[1;31m  - $sample\033[0m"
    done
    exit 1
fi

find . -name '*.log' -size 0 -delete
popd

# Cleanup
rm -rf "$tmp_dir"
