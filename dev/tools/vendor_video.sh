#!/bin/zsh
# Brings an upstream change of the MiniMax-H3 engine (h3) to Splash's copy of
# it (runtime/video/VENDORED). The kernels, their ABI headers and the weight
# converters stay as h3 has them: they are copied again, their include paths
# rewritten. The host code in runtime/video/h3 is Splash's since the import:
# the script prints h3's changes to its host code since the commit VENDORED
# records, to port by hand. Then record the new commit there, rebuild and run
# make test-engine-metal and make test-video-real.
#
#   dev/tools/vendor_video.sh H3_CHECKOUT
set -eu
SRC=${1:?usage: dev/tools/vendor_video.sh H3_CHECKOUT}
SRC=${SRC:A}
cd ${0:A:h:h:h}
base=$(sed -n 's/^Imported from h3 commit \([0-9a-f]*\).*/\1/p' runtime/video/VENDORED)
groups=(audio common dit prep preview shared vae)
rm -rf runtime/metal/abi/video runtime/metal/kernels/video/${^groups} install/video
mkdir -p runtime/metal/abi/video runtime/metal/kernels/video install/video/h3
cp $SRC/runtime/metal/abi/*.h runtime/metal/abi/video/
for group in $groups; do cp -R $SRC/runtime/metal/kernels/$group runtime/metal/kernels/video/$group; done
cp $SRC/tools/convert_*.py install/video/
cp $SRC/h3/__init__.py $SRC/h3/common.py $SRC/h3/config.py $SRC/h3/safetensors.py install/video/h3/
sed -i '' -E -e 's#"metal/abi/#"metal/abi/video/#g' -e 's#"metal/kernels/common/#"metal/kernels/video/common/#g' \
  $(find runtime/metal/abi/video runtime/metal/kernels/video/${^groups} -type f)
git -C $SRC diff --stat $base HEAD -- runtime/engine runtime/media runtime/model runtime/ops \
  runtime/metal/MetalBackend.hpp runtime/metal/MetalBackend.mm include/h3/h3.h
echo "h3 $base..$(git -C $SRC rev-parse --short HEAD): port the host code changes above into runtime/video/h3"
