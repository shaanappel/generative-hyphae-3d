#!/bin/bash

set -e

blender=""

if [ "$(uname)" == "Darwin" ]; then
  blender="/Applications/Blender/blender.app/Contents/MacOS/blender"
else
  blender="/usr/share/blender/blender"
fi

if [ ! -e "$blender" ]; then
  echo "Expected to find blender at: $blender"
  echo "Make sure blender is installed, or fix the path inside this script"
  exit 1
fi

here=$(pwd)
generate="$here/make_path.py"
thing=$(pwd)

res_folder="$here/blend/"

if [ ! -d "$res_folder" ]; then
  mkdir $res_folder
fi

if [ $# -gt 0 ]; then
  echo "too many parameters, expected 0"
  exit 1
fi


"$blender" -b -P "$generate"



