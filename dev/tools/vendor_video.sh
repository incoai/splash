#!/bin/zsh
# Brings an upstream change of the MiniMax-H3 engine (h3) to Splash's video
# engine (runtime/video/VENDORED). Of h3, Splash still copies its kernels and
# their ABI headers, and its weight converters: they are copied again, their
# include paths rewritten (h3's synthetic-weight kernel is not: Splash's is
# its test's, dev/tests/engine/video_synthetic.metal). The host code in
# runtime/video is Splash's own: the script prints h3's changes to its host
# code since the commit VENDORED records, to port by hand. Then record the new
# commit there, rebuild and run make test-engine-metal and make test-video-real.
#
#   dev/tools/vendor_video.sh H3_CHECKOUT
set -eu
SRC=${1:?usage: dev/tools/vendor_video.sh H3_CHECKOUT}
SRC=${SRC:A}
cd ${0:A:h:h:h}
base=$(sed -n 's/^Copied from h3 commit \([0-9a-f]*\).*/\1/p' runtime/video/VENDORED)
groups=(audio common dit prep preview shared vae)
own=(TextEncoderABI.h)
find runtime/metal/abi/video -type f ! -name ${^own} -delete
rm -rf runtime/metal/kernels/video/${^groups} install/video
mkdir -p runtime/metal/kernels/video install/video/h3
cp $SRC/runtime/metal/abi/*.h runtime/metal/abi/video/
rm runtime/metal/abi/video/SyntheticABI.h
for group in $groups; do cp -R $SRC/runtime/metal/kernels/$group runtime/metal/kernels/video/$group; done
rm runtime/metal/kernels/video/shared/synthetic.metal
cp $SRC/tools/convert_*.py install/video/
cp $SRC/h3/__init__.py $SRC/h3/common.py $SRC/h3/config.py $SRC/h3/safetensors.py install/video/h3/
sed -i '' -E -e 's#"metal/abi/#"metal/abi/video/#g' -e 's#"metal/kernels/common/#"metal/kernels/video/common/#g' \
  $(find runtime/metal/abi/video runtime/metal/kernels/video/${^groups} -type f ! -name ${^own})
git -C $SRC diff --stat $base HEAD -- runtime/engine runtime/media runtime/model runtime/ops \
  runtime/metal/MetalBackend.hpp runtime/metal/MetalBackend.mm
echo "h3 $base..$(git -C $SRC rev-parse --short HEAD): port the host code changes above into runtime/video"
