#!/bin/zsh
# Vendors the MiniMax-H3 engine (h3) from a checkout into the places
# runtime/video/VENDORED lists, rewriting its include paths for them, and
# prints the commit to record there. Then rebuild and run make
# test-engine-metal and make test-video-real.
#
#   dev/tools/vendor_video.sh H3_CHECKOUT
set -eu
SRC=${1:?usage: dev/tools/vendor_video.sh H3_CHECKOUT}
SRC=${SRC:A}
cd ${0:A:h:h:h}
hosts=(engine media model ops)
groups=(audio common dit prep preview shared vae)
rm -rf runtime/video/h3 runtime/metal/abi/video runtime/metal/kernels/video/${^groups} install/video
mkdir -p runtime/video/h3/metal runtime/metal/abi/video runtime/metal/kernels/video install/video/h3
for host in $hosts; do cp -R $SRC/runtime/$host runtime/video/h3/$host; done
cp $SRC/runtime/metal/MetalBackend.* runtime/video/h3/metal/
cp $SRC/include/h3/h3.h runtime/video/h3/h3.h
cp $SRC/runtime/metal/abi/*.h runtime/metal/abi/video/
for group in $groups; do cp -R $SRC/runtime/metal/kernels/$group runtime/metal/kernels/video/$group; done
cp $SRC/tools/convert_*.py install/video/
cp $SRC/h3/__init__.py $SRC/h3/common.py $SRC/h3/config.py $SRC/h3/safetensors.py install/video/h3/
rewrite=()
for host in $hosts metal; do rewrite+=(-e "s#\"$host/#\"video/h3/$host/#g"); done
sed -i '' -E $rewrite \
  -e 's#"video/h3/metal/abi/#"metal/abi/video/#g' \
  -e 's#"video/h3/metal/kernels/common/#"metal/kernels/video/common/#g' \
  -e 's#"h3/h3.h"#"video/h3/h3.h"#g' \
  $(find runtime/video/h3 runtime/metal/abi/video runtime/metal/kernels/video/${^groups} -type f)
echo "vendored h3 $(git -C $SRC rev-parse --short HEAD)"
