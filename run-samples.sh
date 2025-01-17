#!/usr/bin/bash

opt_short="dlp:"
opt_long="debug,looped,pattern:,negative"

OPTS=$(getopt -o "$opt_short" -l "$opt_long" -- "$@")

eval set -- "$OPTS"

# default values
build_type=Release
looped=false
pattern=
negative=

# get args
while true
do
    case "$1" in
        -d|--debug)
            build_type=Debug
            shift ;;
        -l|--looped)
            looped=true
            shift;;
		-p|--pattern)
            [[ ! "$2" =~ ^- ]] && pattern="-path $2"
            shift 2 ;;
		--negative)
            negative="!"
            shift;;
        --) # End of input reading
            shift; break ;;
    esac
done

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $build_type == "Debug" ];
then
    echo "Will run the samples using the debug version of shards"
else
    echo "Will run the samples using the release version of shards"
fi

# execute commands
pushd $script_dir/docs/samples
for i in $(find shards -name '*.shs' $negative $pattern);
do
    echo "running sample $i";
    rm -f "$i.log"
    $script_dir/build/$build_type/shards new $script_dir/docs/samples/run-sample.shs looped:$looped file:"$i" 2>"$i.log"
    find . -name '*.log' -size 0 -delete
done
popd
