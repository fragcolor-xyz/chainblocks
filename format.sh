#!/bin/bash

FIND=${FIND:=find}

if [[ "$OSTYPE" == "darwin"* ]]; then
    FIND_ARG0=-E
else
    FIND_ARG1="-regextype posix-extended"
fi

PIDCounter=0
function FormatFolderRecursiveAsync() {
    if [[ ! -d "$1" ]]; then
        echo "Folder not found: $1"
        return
    fi

    while read file; do
        if [[ $file == shards/gfx/rust/wgpu-native* ]]; then
            echo "Ignoring $file"
            continue
        fi

        echo "Formatting $file"
        clang-format -i $file &
        local PID=$!
        PIDs[${PIDCounter}]=$PID
        PIDCounter=$(($PIDCounter+1))
    done <<<"$($FIND $FIND_ARG0 "$1" $FIND_ARG1 -regex '.*\.(cpp|hpp|h)')"
}

function AsyncWait() {
    for pid in ${PIDs[*]}; do
        wait $pid
    done
}

FormatFolderRecursiveAsync include
FormatFolderRecursiveAsync shards
AsyncWait
echo "Formatted ${#PIDs[@]} files"
