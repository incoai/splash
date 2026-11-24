# Native tests, tools and benchmarks of the Omni engine (included by
# dev/Makefile). Tests link the engine library and run against the
# production metallib under Metal shader validation.
ENGINE_TEST_BUILD := $(BUILD)/engine-tests
ENGINE_TEST_CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror -Iruntime -Idev \
	$(MACOS_TARGET_FLAG)
# Production flags less -O3, so Metal optimizes at its default -O2.
TEST_METALFLAGS := $(filter-out -O3,$(PROD_METALFLAGS))
TEST_CONFIG_DIGEST := $(shell printf '%s\0' $(CONFIG_DIGEST) \
	$(call shell-quote,$(ENGINE_TEST_CXXFLAGS)) \
	$(call shell-quote,$(TEST_METALFLAGS)) | shasum -a 256 | cut -c1-16)

TEST_WEIGHT_SOURCE := $(ENGINE_TEST_BUILD)/weight-source
TEST_AFFINE_CHECKPOINT := $(ENGINE_TEST_BUILD)/affine-checkpoint
TEST_OMNI_PROTOCOL := $(ENGINE_TEST_BUILD)/omni-protocol
TEST_METAL_BACKEND_TEST := $(ENGINE_TEST_BUILD)/metal-backend
TEST_METAL_BACKEND_AIR := $(ENGINE_TEST_BUILD)/metal-backend.air
TEST_METAL_BACKEND_LIB := $(ENGINE_TEST_BUILD)/metal-backend.metallib
TEST_PRODUCTION_LIB := $(ENGINE_TEST_BUILD)/production-and-test.metallib
TEST_RESIDENCY_AIR := $(METAL_BUILD)/shared/residency.air
TEST_Q4_SGMATRIX_TEST := $(ENGINE_TEST_BUILD)/q4-sgmatrix
TEST_OMNI_KERNELS := $(ENGINE_TEST_BUILD)/omni-kernels
TEST_OMNI_DECODER := $(ENGINE_TEST_BUILD)/omni-decoder
TEST_OMNI_CODEC := $(ENGINE_TEST_BUILD)/omni-codec
TEST_OMNI_AUDIO_ENCODER := $(ENGINE_TEST_BUILD)/omni-audio-encoder
TEST_OMNI_ASR_SPECULATION := $(ENGINE_TEST_BUILD)/omni-asr-speculation
TEST_ANE_FFN := $(ENGINE_TEST_BUILD)/ane-ffn
TEST_OMNI_PARITY := $(ENGINE_TEST_BUILD)/omni-model-parity
TEST_OMNI_TTS_RUN := $(ENGINE_TEST_BUILD)/omni-tts-run
TEST_OMNI_ASR_RUN := $(ENGINE_TEST_BUILD)/omni-asr-run
TEST_OMNI_STEP_PROFILE := $(ENGINE_TEST_BUILD)/omni-step-profile
TEST_OMNI_PIPELINE_PROFILE := $(ENGINE_TEST_BUILD)/omni-pipeline-profile
TEST_OMNI_DQ4_BENCH := $(ENGINE_TEST_BUILD)/omni-dq4-bench
TEST_OMNI_ANE_PROBE := $(ENGINE_TEST_BUILD)/omni-ane-probe
TEST_OMNI_CODEC_BENCH := $(ENGINE_TEST_BUILD)/omni-codec-bench
TEST_OMNI_ANE_CODEC_PROBE := $(ENGINE_TEST_BUILD)/omni-ane-codec-probe
TEST_OMNI_ANE_STACK_PROBE := $(ENGINE_TEST_BUILD)/omni-ane-stack-probe
TEST_OMNI_ANE_FFN_BENCH := $(ENGINE_TEST_BUILD)/omni-ane-ffn-bench
TEST_OMNI_DENSE_BENCH := $(ENGINE_TEST_BUILD)/omni-dense-bench
TEST_OMNI_DENSE_VARIANTS_AIR := $(ENGINE_TEST_BUILD)/omni-dense-variants.air
TEST_OMNI_DENSE_VARIANTS_LIB := $(ENGINE_TEST_BUILD)/omni-dense-variants.metallib
TEST_OMNI_ASR_VERIFY_BENCH := $(ENGINE_TEST_BUILD)/omni-asr-verify-bench

TEST_CPU_TARGETS := $(TEST_WEIGHT_SOURCE) $(TEST_AFFINE_CHECKPOINT) $(TEST_OMNI_PROTOCOL)
TEST_METAL_TARGETS := $(TEST_METAL_BACKEND_TEST) $(TEST_Q4_SGMATRIX_TEST) \
	$(TEST_OMNI_KERNELS) $(TEST_OMNI_DECODER) $(TEST_OMNI_CODEC) \
	$(TEST_OMNI_AUDIO_ENCODER) $(TEST_ANE_FFN) $(TEST_OMNI_ASR_SPECULATION)
TEST_TOOL_TARGETS := $(TEST_OMNI_PARITY) $(TEST_OMNI_TTS_RUN) $(TEST_OMNI_ASR_RUN) \
	$(TEST_OMNI_STEP_PROFILE) $(TEST_OMNI_PIPELINE_PROFILE) $(TEST_OMNI_DQ4_BENCH) \
	$(TEST_OMNI_ANE_PROBE) $(TEST_OMNI_CODEC_BENCH) $(TEST_OMNI_ANE_CODEC_PROBE) \
	$(TEST_OMNI_ANE_STACK_PROBE) $(TEST_OMNI_ANE_FFN_BENCH) $(TEST_OMNI_DENSE_BENCH) \
	$(TEST_OMNI_ASR_VERIFY_BENCH)
# The test kernels' AIR takes the Makefile's TEST_KERNEL_CONFIG; the libraries
# and binaries the test configuration.
TEST_CONFIG_TARGETS := $(TEST_CPU_TARGETS) $(TEST_METAL_TARGETS) $(TEST_TOOL_TARGETS) \
	$(TEST_METAL_BACKEND_LIB) $(TEST_PRODUCTION_LIB) $(TEST_OMNI_DENSE_VARIANTS_LIB)
TEST_INPUTS = $(filter-out %.h %.hpp %.metallib,$(BUILD_INPUTS))
METAL_TEST_ENV := MTL_SHADER_VALIDATION=1

$(ENGINE_TEST_BUILD): | $(BUILD)
	mkdir -p $@

# CPU tests, built from the sources they check.
$(TEST_WEIGHT_SOURCE): dev/tests/engine/weight_source_test.cpp runtime/model/WeightSource.cpp
$(TEST_WEIGHT_SOURCE): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@
$(TEST_AFFINE_CHECKPOINT): dev/tests/engine/affine_checkpoint_test.cpp $(ENGINE_LIBRARY)
$(TEST_AFFINE_CHECKPOINT): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) $(ENGINE_LINKFLAGS) -o $@
# The speech request frame, the TTS model types and their prompts.
$(TEST_OMNI_PROTOCOL): dev/tests/engine/omni_protocol_test.cpp $(ENGINE_LIBRARY)
$(TEST_OMNI_PROTOCOL): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) $(ENGINE_LINKFLAGS) -o $@

# Metal tests and tools, linked against the engine library.
$(TEST_METAL_BACKEND_TEST): dev/tests/engine/metal_backend_test.mm \
		$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_LIBRARY) $(TEST_METAL_BACKEND_LIB)
$(TEST_Q4_SGMATRIX_TEST): dev/tests/engine/q4_sgmatrix_metal_test.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_KERNELS): dev/tests/engine/omni_kernels_test.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_DECODER): dev/tests/engine/omni_decoder_test.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_CODEC): dev/tests/engine/omni_codec_test.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_AUDIO_ENCODER): dev/tests/engine/omni_audio_encoder_test.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_ASR_SPECULATION): dev/tests/engine/omni_asr_speculation_test.mm $(ENGINE_LIBRARY) $(LIB)
# The ANE FFN test links the instrumented Program for its fault checks.
$(TEST_ANE_FFN): dev/tests/engine/ane_ffn_test.mm $(ENGINE_INSTRUMENTED_ANE_OBJECT) $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_PARITY): dev/tests/engine/omni_model_parity_test.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_TTS_RUN): dev/tools/omni_tts_run.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_ASR_RUN): dev/tools/omni_asr_run.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_STEP_PROFILE): dev/benchmarks/omni_step_profile.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_DQ4_BENCH): dev/benchmarks/omni_dq4_bench.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_ANE_PROBE): dev/benchmarks/omni_ane_probe.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_CODEC_BENCH): dev/benchmarks/omni_codec_bench.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_ANE_CODEC_PROBE): dev/benchmarks/omni_ane_codec_probe.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_ANE_STACK_PROBE): dev/benchmarks/omni_ane_stack_probe.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_ANE_FFN_BENCH): dev/benchmarks/omni_ane_ffn_bench.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_DENSE_BENCH): dev/benchmarks/omni_dense_bench.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_OMNI_ASR_VERIFY_BENCH): dev/benchmarks/omni_asr_verify_bench.mm $(ENGINE_LIBRARY) $(LIB)
# The pipeline profiler links the instrumented backend for its per-dispatch timings.
$(TEST_OMNI_PIPELINE_PROFILE): dev/benchmarks/omni_pipeline_profile.mm \
		$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_LIBRARY) $(LIB)

$(TEST_METAL_TARGETS) $(TEST_TOOL_TARGETS): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

# The Metal backend test's kernels, compiled with the test flags.
$(TEST_METAL_BACKEND_AIR): dev/tests/engine/metal_backend_test.metal | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(METAL) $(TEST_METALFLAGS) -c $< -o $@
$(TEST_METAL_BACKEND_LIB): $(TEST_METAL_BACKEND_AIR) $(TEST_RESIDENCY_AIR)
# The production kernels and the test kernels of metal_backend_test.metal in
# one library, for the Q4 tile test, which orders a test copy between
# production dispatches. It relinks whenever the production library does.
$(TEST_PRODUCTION_LIB): $(PRODUCTION_AIRS) $(TEST_METAL_BACKEND_AIR) $(LIB)
# The production kernels and the audio tower GEMM's variants for
# omni-dense-bench --tower: make omni-dense-variants. The variants take the
# production flags, so they compare with the production tiles.
$(TEST_OMNI_DENSE_VARIANTS_AIR): dev/benchmarks/omni_dense_variants.metal $(KERNEL_HEADERS) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(METAL) $(PROD_METALFLAGS) -c $< -o $@
$(TEST_OMNI_DENSE_VARIANTS_LIB): $(PRODUCTION_AIRS) $(TEST_OMNI_DENSE_VARIANTS_AIR)
$(TEST_METAL_BACKEND_LIB) $(TEST_PRODUCTION_LIB) $(TEST_OMNI_DENSE_VARIANTS_LIB):
	$(RUN_CONFIGURED) $(METALLIB) $(filter %.air,$^) -o $@
.PHONY: omni-dense-variants
omni-dense-variants: $(TEST_OMNI_DENSE_BENCH) $(TEST_OMNI_DENSE_VARIANTS_LIB)

.PHONY: test-engine test-engine-cpu test-engine-metal
test-engine: test-engine-cpu test-engine-metal

test-engine-cpu: $(TEST_CPU_TARGETS)
	$(TEST_AFFINE_CHECKPOINT)
	$(TEST_WEIGHT_SOURCE)
	$(TEST_OMNI_PROTOCOL)

test-engine-metal: $(TEST_METAL_TARGETS) $(LIB) $(TEST_PRODUCTION_LIB)
	$(METAL_TEST_ENV) $(TEST_METAL_BACKEND_TEST) $(TEST_METAL_BACKEND_LIB)
	$(METAL_TEST_ENV) $(TEST_Q4_SGMATRIX_TEST) $(TEST_PRODUCTION_LIB)
	$(METAL_TEST_ENV) $(TEST_OMNI_KERNELS) $(LIB)
	$(METAL_TEST_ENV) $(TEST_OMNI_DECODER) $(LIB)
	$(METAL_TEST_ENV) $(TEST_OMNI_CODEC) $(LIB)
	$(TEST_OMNI_CODEC) $(LIB) --ane
	$(METAL_TEST_ENV) $(TEST_OMNI_AUDIO_ENCODER) $(LIB) dev/tests/fixtures/omni
	$(METAL_TEST_ENV) $(TEST_OMNI_ASR_SPECULATION) $(LIB)
	$(METAL_TEST_ENV) $(TEST_ANE_FFN) $(LIB) kernels
	$(TEST_ANE_FFN) $(LIB) split
	$(TEST_ANE_FFN) $(LIB) faults

# The Neural Engine split's checks alone. Metal's validation layer wraps the
# shared event the ANE waits on in one it cannot share, so only the kernel
# checks run under it.
.PHONY: test-ane-ffn
test-ane-ffn: $(TEST_ANE_FFN) $(LIB)
	$(METAL_TEST_ENV) $(TEST_ANE_FFN) $(LIB) kernels
	$(TEST_ANE_FFN) $(LIB) split
	$(TEST_ANE_FFN) $(LIB) faults

.PHONY: test-omni-kernels test-omni-decoder test-omni-codec test-omni-audio-encoder test-omni-asr-speculation
test-omni-kernels: $(TEST_OMNI_KERNELS) $(LIB)
	$(METAL_TEST_ENV) $(TEST_OMNI_KERNELS) $(LIB)
test-omni-decoder: $(TEST_OMNI_DECODER) $(LIB)
	$(METAL_TEST_ENV) $(TEST_OMNI_DECODER) $(LIB)
test-omni-codec: $(TEST_OMNI_CODEC) $(LIB)
	$(METAL_TEST_ENV) $(TEST_OMNI_CODEC) $(LIB)
	$(TEST_OMNI_CODEC) $(LIB) --ane
test-omni-audio-encoder: $(TEST_OMNI_AUDIO_ENCODER) $(LIB)
	$(METAL_TEST_ENV) $(TEST_OMNI_AUDIO_ENCODER) $(LIB) dev/tests/fixtures/omni
test-omni-asr-speculation: $(TEST_OMNI_ASR_SPECULATION) $(LIB)
	$(METAL_TEST_ENV) $(TEST_OMNI_ASR_SPECULATION) $(LIB)

# Real-weight parity against mlx-audio (dev/tools/omni_parity_reference.py):
# make test-omni-parity-real OMNI_KIND=asr|tts|tts-prompt|tts-force OMNI_MODEL=DIR OMNI_REFERENCE=FILE
.PHONY: test-omni-parity-real
test-omni-parity-real: $(TEST_OMNI_PARITY) $(LIB)
	$(TEST_OMNI_PARITY) $(LIB) $(OMNI_KIND) "$(OMNI_MODEL)" "$(OMNI_REFERENCE)"

# The engine run tools and the step profiler: make omni-tools.
.PHONY: omni-tools
omni-tools: $(TEST_TOOL_TARGETS)
