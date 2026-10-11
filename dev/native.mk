ENGINE_TEST_BUILD := $(BUILD)/engine-tests
ENGINE_TEST_CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror -Iruntime -Idev \
	$(MACOS_TARGET_FLAG)
# Production flags less -O3, so Metal optimizes at its default -O2. With
# metal 32023 the kernels built with them get the same AIR as at -O3, but
# three production kernels do not (shared/gguf_linear, moe_gguf,
# normalization): a test that recompiles those needs PROD_METALFLAGS.
TEST_METALFLAGS := $(filter-out -O3,$(PROD_METALFLAGS))
# Include the shared tools and production flags: some tests also link the
# production library or metallib, and all share ENGINE_LINKFLAGS.
TEST_CONFIG_DIGEST := $(shell printf '%s\0' $(CONFIG_DIGEST) \
	$(call shell-quote,$(ENGINE_TEST_CXXFLAGS)) \
	$(call shell-quote,$(TEST_METALFLAGS)) | shasum -a 256 | cut -c1-16)
ENGINE_SANITIZER_BUILD := $(BUILD)/sanitizers
ENGINE_SANITIZER_CXXFLAGS := -std=c++20 -O1 -g -fno-omit-frame-pointer \
	-Wall -Wextra -Werror -Iruntime -Idev $(MACOS_TARGET_FLAG)
SANITIZER_CONFIG_DIGEST := $(shell printf '%s\0' $(CONFIG_DIGEST) \
	$(call shell-quote,$(ENGINE_SANITIZER_CXXFLAGS)) \
	| shasum -a 256 | cut -c1-16)
# Control-plane tests compile the runtime sources they check: their sanitizer
# builds cannot use the unsanitized engine library, and none links a framework.
CACHE_SOURCES := \
	runtime/engine/KvPool.cpp \
	runtime/engine/KvCache.cpp \
	runtime/engine/StateCache.cpp \
	runtime/engine/Cache.cpp \
	runtime/engine/WriteBehind.cpp
BACKEND_CONTROL_SOURCES := \
	runtime/engine/Scheduler.cpp \
	runtime/model/DraftContextPlan.cpp \
	$(CACHE_SOURCES) \
	runtime/engine/Engine.cpp
NATIVE_RUNTIME_SOURCES := $(BACKEND_CONTROL_SOURCES) \
	runtime/engine/Protocol.cpp \
	runtime/engine/NativeRuntime.cpp
TEST_SLOT_FILE_ASAN := $(ENGINE_SANITIZER_BUILD)/slot-file-asan-ubsan
TEST_SLOT_FILE_TSAN := $(ENGINE_SANITIZER_BUILD)/slot-file-tsan
TEST_CACHE_DIRECTORY_ASAN := $(ENGINE_SANITIZER_BUILD)/cache-directory-asan-ubsan
TEST_CACHE_DIRECTORY_TSAN := $(ENGINE_SANITIZER_BUILD)/cache-directory-tsan
TEST_PERSISTENT_CACHE_ASAN := $(ENGINE_SANITIZER_BUILD)/persistent-cache-asan-ubsan
TEST_PERSISTENT_CACHE_TSAN := $(ENGINE_SANITIZER_BUILD)/persistent-cache-tsan
TEST_ENGINE_CACHE_ASAN := $(ENGINE_SANITIZER_BUILD)/engine-cache-asan-ubsan
TEST_ENGINE_CACHE_TSAN := $(ENGINE_SANITIZER_BUILD)/engine-cache-tsan
TEST_PROTOCOL_ASAN := $(ENGINE_SANITIZER_BUILD)/protocol-asan-ubsan
TEST_PROTOCOL_TSAN := $(ENGINE_SANITIZER_BUILD)/protocol-tsan
TEST_ENGINE_ASAN := $(ENGINE_SANITIZER_BUILD)/engine-asan-ubsan
TEST_ENGINE_TSAN := $(ENGINE_SANITIZER_BUILD)/engine-tsan
TEST_FD_TRANSPORT_ASAN := $(ENGINE_SANITIZER_BUILD)/native-fd-asan-ubsan
TEST_FD_TRANSPORT_TSAN := $(ENGINE_SANITIZER_BUILD)/native-fd-tsan
TEST_ANE_FFN_CALIBRATION_ASAN := $(ENGINE_SANITIZER_BUILD)/ane-ffn-calibration-asan-ubsan
TEST_ANE_FFN_CALIBRATION_TSAN := $(ENGINE_SANITIZER_BUILD)/ane-ffn-calibration-tsan
TEST_MEMORY_TEST := $(ENGINE_TEST_BUILD)/engine-memory-plan
TEST_VISION_PREPARATION := $(ENGINE_TEST_BUILD)/vision-preparation
TEST_DRAFT_PREPARATION := $(ENGINE_TEST_BUILD)/draft-preparation
TEST_SAFETENSORS_CHECKPOINT := $(ENGINE_TEST_BUILD)/safetensors-checkpoint
TEST_MODEL_CONFIGURATION := $(ENGINE_TEST_BUILD)/model-configuration
TEST_WEIGHT_SOURCE := $(ENGINE_TEST_BUILD)/weight-source
TEST_GGUF_FILE := $(ENGINE_TEST_BUILD)/gguf-file
TEST_GGUF_PROJECTION := $(ENGINE_TEST_BUILD)/gguf-projection
TEST_ANE_FFN := $(ENGINE_TEST_BUILD)/ane-ffn
TEST_ANE_FFN_CALIBRATION := $(ENGINE_TEST_BUILD)/ane-ffn-calibration
TEST_ANE_FFN_STARTUP := $(ENGINE_TEST_BUILD)/ane-ffn-startup
TEST_ANE_PROGRAM_FAULTS := $(ENGINE_TEST_BUILD)/ane-program-faults
TEST_GGUF_DEQUANT := $(ENGINE_TEST_BUILD)/gguf-dequant
TEST_GGUF_ROTATION := $(ENGINE_TEST_BUILD)/gguf-rotation
TEST_GGUF_MOE := $(ENGINE_TEST_BUILD)/gguf-moe
TEST_GGUF_REFERENCE := $(ENGINE_TEST_BUILD)/gguf-reference
TEST_GGUF_PLANNER := $(ENGINE_TEST_BUILD)/gguf-planner
TEST_GGUF_PREPARATION := $(ENGINE_TEST_BUILD)/gguf-preparation
TEST_GGUF_DEQUANT_AIR := $(ENGINE_TEST_BUILD)/gguf-dequant.air
TEST_GGUF_DEQUANT_LIB := $(ENGINE_TEST_BUILD)/gguf-dequant.metallib
# Every hash the weight tests compare against, and how to update them.
WEIGHT_GOLDENS := dev/tests/fixtures/weight-goldens/goldens.json
MLX_QUANTIZATION_FIXTURE := dev/tests/fixtures/mlx-quantization/fixture.json
# An MLX config.json and a DFlash2 draft config built like each family's.
MODEL_CONFIGS := dev/tests/fixtures/model-configs
TEST_KV_PAGE_CACHE_TEST := $(ENGINE_TEST_BUILD)/kv-page-cache
TEST_ENGINE_CACHE_TEST := $(ENGINE_TEST_BUILD)/engine-cache
TEST_DRAFT_CONTEXT_PLAN_TEST := $(ENGINE_TEST_BUILD)/draft-context-plan
TEST_RAGGED_SCHEDULER_TEST := $(ENGINE_TEST_BUILD)/ragged-scheduler
TEST_ENGINE_TEST := $(ENGINE_TEST_BUILD)/engine
TEST_ELASTIC_KV_TEST := $(ENGINE_TEST_BUILD)/elastic-kv-pool
TEST_PROTOCOL_TEST := $(ENGINE_TEST_BUILD)/protocol
TEST_NATIVE_LOOP_TEST := $(ENGINE_TEST_BUILD)/native-engine-loop
TEST_FD_TRANSPORT_TEST := $(ENGINE_TEST_BUILD)/native-fd-transport
TEST_BOOTSTRAP_TEST := $(ENGINE_TEST_BUILD)/runtime-bootstrap
TEST_NATIVE_ARGUMENTS_TEST := $(ENGINE_TEST_BUILD)/native-arguments
TEST_RESOURCES_TEST := $(ENGINE_TEST_BUILD)/runtime-resources
TEST_KV_PAGE_TIER_TEST := $(ENGINE_TEST_BUILD)/kv-page-tier
TEST_METRICS_TEST := $(ENGINE_TEST_BUILD)/runtime-metrics
TEST_MODEL_LOADING_TEST := $(ENGINE_TEST_BUILD)/model-loading
TEST_MEMORY_AUDIT_TEST := $(ENGINE_TEST_BUILD)/memory-audit
TEST_MEMORY_GOVERNOR_TEST := $(ENGINE_TEST_BUILD)/memory-governor
TEST_QWEN_STATE_TEST := $(ENGINE_TEST_BUILD)/qwen-state-storage
TEST_STATUS_TEST := $(ENGINE_TEST_BUILD)/runtime-status
TEST_Q8_CPU_TEST := $(ENGINE_TEST_BUILD)/q8-paged-kv
TEST_Q8_METAL_TEST := $(ENGINE_TEST_BUILD)/q8-paged-kv-metal
TEST_Q8_STORAGE_TEST := $(ENGINE_TEST_BUILD)/q8-page-storage
TEST_Q8_ATTENTION_TEST := $(ENGINE_TEST_BUILD)/q8-flash-attention
TEST_Q8_PREFILL_TEST := $(ENGINE_TEST_BUILD)/q8-chunked-prefill
TEST_GDN_METAL_TEST := $(ENGINE_TEST_BUILD)/gdn-metal
TEST_OPERATOR_WORKSPACE := $(ENGINE_TEST_BUILD)/operator-workspace
TEST_EXECUTION_PLANS := $(ENGINE_TEST_BUILD)/execution-plans
TEST_MODEL_EXECUTION_PLANS := $(ENGINE_TEST_BUILD)/model-execution-plans
TEST_ATTENTION_PLAN := $(ENGINE_TEST_BUILD)/paged-attention-plan
TEST_LINEAR_PLAN := $(ENGINE_TEST_BUILD)/linear-plan
TEST_DFLASH_BATCH_CONTROL_TEST := $(ENGINE_TEST_BUILD)/dflash-batch-control
TEST_DRAFT_ATTENTION_TEST := $(ENGINE_TEST_BUILD)/draft-attention
TEST_GDN_DECODE_TEST := $(ENGINE_TEST_BUILD)/gdn-decode
TEST_DRAFT_SELECTOR_TEST := $(ENGINE_TEST_BUILD)/draft-selector
TEST_LIVE_VERIFY_ROWS := $(ENGINE_TEST_BUILD)/live-verify-rows
TEST_TARGET_SAMPLING_TEST := $(ENGINE_TEST_BUILD)/target-sampling
TEST_VISION_METAL_TEST := $(ENGINE_TEST_BUILD)/vision-metal
TEST_BACKEND_BENCHMARK := $(ENGINE_TEST_BUILD)/backend-benchmark
TEST_DECODE_PROFILE := $(ENGINE_TEST_BUILD)/decode-profile
TEST_ATTENTION_SWEEP := $(ENGINE_TEST_BUILD)/attention-sweep
TEST_GGUF_PROJECTION_BENCHMARK := $(ENGINE_TEST_BUILD)/gguf-projection-benchmark
TEST_GGUF_MOE_BENCHMARK := $(ENGINE_TEST_BUILD)/gguf-moe-benchmark
TEST_MODEL_RUNTIME_ORACLE := $(ENGINE_TEST_BUILD)/model-runtime-oracle
WEIGHT_DIGESTS := $(ENGINE_TEST_BUILD)/weight-digests
TEST_VISION_ENCODER_TEST := $(ENGINE_TEST_BUILD)/vision-encoder
TEST_Q8_AIR := $(ENGINE_TEST_BUILD)/q8-paged-kv.air
TEST_Q8_LIB := $(ENGINE_TEST_BUILD)/q8-paged-kv.metallib
# Both Q8 tests share the attention and store kernels of both phases.
TEST_Q8_KERNEL_SOURCES := \
	runtime/metal/kernels/prefill/paged_attention.metal \
	runtime/metal/kernels/decode/paged_attention.metal \
	runtime/metal/kernels/prefill/paged_attention_store.metal \
	runtime/metal/kernels/decode/paged_attention_store.metal
TEST_Q8_KERNEL_AIRS := $(patsubst runtime/metal/kernels/%.metal,$(ENGINE_TEST_BUILD)/kernels/%.air,$(TEST_Q8_KERNEL_SOURCES))
# Every library a MetalBackend loads has the kernel that ends its residency;
# the test libraries built without the production kernels link it in.
TEST_RESIDENCY_AIR := $(ENGINE_TEST_BUILD)/kernels/shared/residency.air
TEST_Q8_ATTENTION_LIB := $(ENGINE_TEST_BUILD)/q8-attention.metallib
TEST_METAL_BACKEND_TEST := $(ENGINE_TEST_BUILD)/metal-backend
TEST_HANDOFF_TEST := $(ENGINE_TEST_BUILD)/handoff
TEST_METAL_BACKEND_AIR := $(ENGINE_TEST_BUILD)/metal-backend.air
TEST_METAL_BACKEND_LIB := $(ENGINE_TEST_BUILD)/metal-backend.metallib
TEST_PRODUCTION_LIB := $(ENGINE_TEST_BUILD)/production-and-test.metallib

TEST_SLOT_FILE := $(ENGINE_TEST_BUILD)/slot-file
TEST_CACHE_DIRECTORY := $(ENGINE_TEST_BUILD)/cache-directory
TEST_PERSISTENT_CACHE := $(ENGINE_TEST_BUILD)/persistent-cache

TEST_CPU_TARGETS := $(TEST_SLOT_FILE) $(TEST_CACHE_DIRECTORY) $(TEST_VISION_PREPARATION) $(TEST_SAFETENSORS_CHECKPOINT) $(TEST_WEIGHT_SOURCE) $(TEST_OPERATOR_WORKSPACE) \
	$(TEST_MODEL_CONFIGURATION) \
	$(TEST_GGUF_FILE) \
	$(TEST_GGUF_REFERENCE) $(TEST_GGUF_PLANNER) \
	$(TEST_LINEAR_PLAN) \
	$(TEST_ANE_FFN_CALIBRATION) \
	$(TEST_ANE_FFN_STARTUP) \
	$(TEST_EXECUTION_PLANS) \
	$(TEST_MODEL_EXECUTION_PLANS) \
	$(TEST_MEMORY_TEST) \
	$(TEST_KV_PAGE_CACHE_TEST) \
	$(TEST_ENGINE_CACHE_TEST) \
	$(TEST_PERSISTENT_CACHE) \
	$(TEST_DRAFT_CONTEXT_PLAN_TEST) \
	$(TEST_RAGGED_SCHEDULER_TEST) \
	$(TEST_ENGINE_TEST) \
	$(TEST_ELASTIC_KV_TEST) \
	$(TEST_PROTOCOL_TEST) \
	$(TEST_NATIVE_LOOP_TEST) \
	$(TEST_FD_TRANSPORT_TEST) \
	$(TEST_BOOTSTRAP_TEST) \
	$(TEST_NATIVE_ARGUMENTS_TEST) \
	$(TEST_METRICS_TEST) \
	$(TEST_MEMORY_AUDIT_TEST) \
	$(TEST_MEMORY_GOVERNOR_TEST) \
	$(TEST_STATUS_TEST) \
	$(TEST_Q8_CPU_TEST)

TEST_METAL_TARGETS := $(TEST_DRAFT_PREPARATION) \
	$(TEST_GGUF_PROJECTION) \
	$(TEST_ANE_FFN) \
	$(TEST_ANE_PROGRAM_FAULTS) \
	$(TEST_GGUF_DEQUANT) \
	$(TEST_GGUF_ROTATION) \
	$(TEST_GGUF_MOE) \
	$(TEST_GGUF_PREPARATION) \
	$(TEST_ATTENTION_PLAN) \
	$(TEST_LINEAR_PLAN) \
	$(TEST_RESOURCES_TEST) \
	$(TEST_KV_PAGE_TIER_TEST) \
	$(TEST_MODEL_LOADING_TEST) \
	$(TEST_QWEN_STATE_TEST) \
	$(TEST_Q8_METAL_TEST) \
	$(TEST_Q8_STORAGE_TEST) \
	$(TEST_Q8_ATTENTION_TEST) \
	$(TEST_Q8_PREFILL_TEST) \
	$(TEST_GDN_METAL_TEST) \
	$(TEST_DFLASH_BATCH_CONTROL_TEST) \
	$(TEST_DRAFT_ATTENTION_TEST) \
	$(TEST_GDN_DECODE_TEST) \
	$(TEST_DRAFT_SELECTOR_TEST) \
	$(TEST_LIVE_VERIFY_ROWS) \
	$(TEST_TARGET_SAMPLING_TEST) \
	$(TEST_VISION_METAL_TEST) \
	$(TEST_METAL_BACKEND_TEST) \
	$(TEST_HANDOFF_TEST) \
	$(LIB) $(TEST_METAL_BACKEND_LIB) $(TEST_PRODUCTION_LIB) $(TEST_Q8_LIB) $(TEST_Q8_ATTENTION_LIB) \
	$(TEST_GGUF_DEQUANT_LIB)

# Keep every output that uses a flag set together, including standalone
# benchmarks, real-model tests and intermediate test AIRs/metallibs.
TEST_CONFIG_TARGETS := $(filter-out $(LIB),$(sort $(TEST_CPU_TARGETS) $(TEST_METAL_TARGETS))) \
	$(TEST_MODEL_RUNTIME_ORACLE) $(TEST_VISION_ENCODER_TEST) \
	$(TEST_DECODE_PROFILE) $(TEST_ATTENTION_SWEEP) \
	$(TEST_GGUF_PROJECTION_BENCHMARK) $(TEST_GGUF_MOE_BENCHMARK) \
	$(TEST_Q8_AIR) $(TEST_Q8_KERNEL_AIRS) $(TEST_RESIDENCY_AIR) $(TEST_METAL_BACKEND_AIR) \
	$(TEST_GGUF_DEQUANT_AIR)
# Benchmarks and tools that build with the production flags.
PRODUCTION_FLAG_TOOLS := $(TEST_BACKEND_BENCHMARK) $(WEIGHT_DIGESTS)
PRODUCTION_CONFIG_TARGETS += $(PRODUCTION_FLAG_TOOLS)
SANITIZER_CONFIG_TARGETS := $(TEST_SLOT_FILE_ASAN) $(TEST_SLOT_FILE_TSAN) \
	$(TEST_CACHE_DIRECTORY_ASAN) $(TEST_CACHE_DIRECTORY_TSAN) \
	$(TEST_PERSISTENT_CACHE_ASAN) $(TEST_PERSISTENT_CACHE_TSAN) \
	$(TEST_ENGINE_CACHE_ASAN) $(TEST_ENGINE_CACHE_TSAN) \
	$(TEST_PROTOCOL_ASAN) $(TEST_PROTOCOL_TSAN) \
	$(TEST_ENGINE_ASAN) $(TEST_ENGINE_TSAN) \
	$(TEST_FD_TRANSPORT_ASAN) $(TEST_FD_TRANSPORT_TSAN) \
	$(TEST_ANE_FFN_CALIBRATION_ASAN) $(TEST_ANE_FFN_CALIBRATION_TSAN)

# Every native test, benchmark and tool depends on the engine's and the tests'
# headers, and compiles and links its other prerequisites in the order it
# lists them: sources, objects and the engine library. A Metal library it
# loads at run time and the force dependency are not compiler inputs either.
ENGINE_TEST_HEADERS := $(filter %.h %.hpp,$(PRODUCTION_ENGINE_INPUTS)) \
	$(wildcard dev/tests/engine/*.hpp)
TEST_INPUTS = $(filter-out %.h %.hpp %.metallib,$(BUILD_INPUTS))
$(filter-out %.air %.metallib,$(TEST_CONFIG_TARGETS)) $(PRODUCTION_FLAG_TOOLS) \
	$(SANITIZER_CONFIG_TARGETS): $(ENGINE_TEST_HEADERS)

$(ENGINE_TEST_BUILD) $(ENGINE_SANITIZER_BUILD):
	mkdir -p $@

# CPU tests built from the runtime sources they check, without the engine
# library or a framework. Those that also run under the sanitizers list their
# sources with the sanitizer builds.
$(TEST_WEIGHT_SOURCE): dev/tests/engine/weight_source_test.cpp runtime/model/WeightSource.cpp
$(TEST_GGUF_FILE): dev/tests/engine/gguf_file_test.cpp runtime/model/GgufFile.cpp \
		runtime/model/WeightSource.cpp
$(TEST_MEMORY_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp \
		dev/tests/engine/engine_memory_plan_test.cpp
$(TEST_KV_PAGE_CACHE_TEST): runtime/engine/KvPool.cpp \
		runtime/engine/KvCache.cpp \
		dev/tests/engine/kv_page_cache_test.cpp
$(TEST_DRAFT_CONTEXT_PLAN_TEST): runtime/model/DraftContextPlan.cpp \
		dev/benchmarks/PrefillWork.hpp \
		dev/tests/engine/draft_context_plan_test.cpp
$(TEST_RAGGED_SCHEDULER_TEST): runtime/engine/Scheduler.cpp \
		dev/tests/engine/ragged_scheduler_test.cpp
$(TEST_ELASTIC_KV_TEST): runtime/engine/KvPool.cpp \
		dev/tests/engine/elastic_kv_pool_test.cpp
$(TEST_NATIVE_LOOP_TEST): $(NATIVE_RUNTIME_SOURCES) \
		runtime/engine/MemoryGovernor.cpp runtime/engine/MemoryControl.cpp \
		dev/tests/engine/ProtocolPeer.cpp \
		dev/tests/engine/native_engine_loop_test.cpp
$(TEST_METRICS_TEST): dev/tests/engine/runtime_metrics_test.cpp
$(TEST_MEMORY_AUDIT_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryAudit.cpp \
		dev/tests/engine/memory_audit_test.cpp
$(TEST_MEMORY_GOVERNOR_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryGovernor.cpp \
		dev/tests/engine/memory_governor_test.cpp
$(TEST_STATUS_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryAudit.cpp \
		runtime/engine/Status.cpp \
		dev/tests/engine/runtime_status_test.cpp
$(TEST_Q8_CPU_TEST): dev/tests/engine/q8_paged_kv_test.cc

$(TEST_WEIGHT_SOURCE) $(TEST_GGUF_FILE) $(TEST_MEMORY_TEST) $(TEST_KV_PAGE_CACHE_TEST) \
		$(TEST_DRAFT_CONTEXT_PLAN_TEST) $(TEST_RAGGED_SCHEDULER_TEST) \
		$(TEST_ELASTIC_KV_TEST) $(TEST_NATIVE_LOOP_TEST) $(TEST_METRICS_TEST) \
		$(TEST_MEMORY_AUDIT_TEST) $(TEST_MEMORY_GOVERNOR_TEST) $(TEST_STATUS_TEST) \
		$(TEST_Q8_CPU_TEST) $(TEST_SLOT_FILE) $(TEST_CACHE_DIRECTORY) \
		$(TEST_PERSISTENT_CACHE) $(TEST_ENGINE_CACHE_TEST) $(TEST_PROTOCOL_TEST) \
		$(TEST_ENGINE_TEST) $(TEST_FD_TRANSPORT_TEST) $(TEST_ANE_FFN_CALIBRATION): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

# C++ tests linked with the frameworks.
$(TEST_SAFETENSORS_CHECKPOINT): dev/tests/engine/safetensors_checkpoint_test.cpp $(ENGINE_LIBRARY)
$(TEST_MODEL_CONFIGURATION): dev/tests/engine/model_configuration_test.cpp $(ENGINE_LIBRARY)
$(TEST_VISION_PREPARATION): dev/tests/engine/vision_preparation_test.cpp $(ENGINE_LIBRARY)
$(TEST_NATIVE_ARGUMENTS_TEST): dev/tests/engine/native_arguments_test.cpp \
		$(ENGINE_LIBRARY)
$(TEST_ANE_FFN_STARTUP): dev/tests/engine/ane_ffn_startup_test.cpp $(ENGINE_LIBRARY)
$(TEST_MODEL_LOADING_TEST): dev/tests/engine/model_loading_test.cpp \
		$(ENGINE_LIBRARY)
$(TEST_OPERATOR_WORKSPACE): dev/tests/engine/operator_workspace_test.cc \
		$(ENGINE_LIBRARY)
$(TEST_EXECUTION_PLANS): dev/tests/engine/execution_plans_test.cc \
		$(ENGINE_LIBRARY)
$(TEST_MODEL_EXECUTION_PLANS): dev/tests/engine/model_execution_plan_test.cpp \
		$(ENGINE_LIBRARY)
$(TEST_VISION_METAL_TEST): dev/tests/engine/vision_metal_test.cpp \
		$(ENGINE_LIBRARY) $(LIB)

$(TEST_SAFETENSORS_CHECKPOINT) $(TEST_MODEL_CONFIGURATION) $(TEST_VISION_PREPARATION) \
		$(TEST_NATIVE_ARGUMENTS_TEST) $(TEST_ANE_FFN_STARTUP) $(TEST_MODEL_LOADING_TEST) $(TEST_OPERATOR_WORKSPACE) \
		$(TEST_EXECUTION_PLANS) $(TEST_MODEL_EXECUTION_PLANS) $(TEST_VISION_METAL_TEST): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

# Objective-C++ tests, oracles and benchmarks linked with the frameworks.
$(TEST_DRAFT_PREPARATION): dev/tests/engine/draft_preparation_test.mm $(ENGINE_LIBRARY)
$(TEST_GGUF_PROJECTION): dev/tests/engine/gguf_projection_test.mm $(ENGINE_LIBRARY) $(LIB)
$(TEST_ANE_FFN): dev/tests/engine/ane_ffn_test.mm $(ENGINE_INSTRUMENTED_ANE_OBJECT) $(ENGINE_LIBRARY) $(LIB)
$(TEST_ANE_PROGRAM_FAULTS): dev/tests/engine/ane_program_faults_test.mm \
		$(ENGINE_INSTRUMENTED_ANE_OBJECT) $(ENGINE_LIBRARY) $(LIB)
$(TEST_GGUF_DEQUANT): dev/tests/engine/gguf_dequant_test.mm $(ENGINE_LIBRARY)
$(TEST_GGUF_ROTATION): dev/tests/engine/gguf_rotation_test.mm $(ENGINE_LIBRARY)
$(TEST_GGUF_REFERENCE): dev/tests/engine/gguf_reference_test.mm $(ENGINE_LIBRARY)
$(TEST_GGUF_PLANNER): dev/tests/engine/gguf_planner_test.mm $(ENGINE_LIBRARY)
$(TEST_GGUF_PREPARATION): dev/tests/engine/gguf_preparation_test.mm $(ENGINE_LIBRARY)
$(TEST_BOOTSTRAP_TEST): dev/tests/engine/runtime_bootstrap_test.mm \
		dev/tests/engine/ProtocolPeer.cpp $(ENGINE_LIBRARY)
$(TEST_RESOURCES_TEST): dev/tests/engine/runtime_resources_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_KV_PAGE_TIER_TEST): dev/tests/engine/kv_page_tier_test.mm \
		$(ENGINE_LIBRARY)
$(TEST_QWEN_STATE_TEST): dev/tests/engine/qwen_state_storage_test.mm \
		$(ENGINE_LIBRARY)
$(TEST_Q8_ATTENTION_TEST): dev/tests/engine/q8_flash_attention_metal_test.mm \
		$(ENGINE_LIBRARY) $(TEST_Q8_ATTENTION_LIB)
$(TEST_Q8_PREFILL_TEST): dev/tests/engine/q8_chunked_prefill_metal_test.mm \
		$(ENGINE_LIBRARY) $(TEST_Q8_ATTENTION_LIB)
$(TEST_GGUF_MOE): dev/tests/engine/gguf_moe_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GDN_METAL_TEST): dev/tests/engine/gdn_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_ATTENTION_PLAN): dev/tests/engine/paged_attention_plan_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_LINEAR_PLAN): dev/tests/engine/linear_plan_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_DFLASH_BATCH_CONTROL_TEST): dev/tests/engine/dflash_batch_control_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_DRAFT_ATTENTION_TEST): dev/tests/engine/draft_attention_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GDN_DECODE_TEST): dev/tests/engine/gdn_decode_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_DRAFT_SELECTOR_TEST): dev/tests/engine/draft_selector_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_LIVE_VERIFY_ROWS): dev/tests/engine/live_verify_rows_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_TARGET_SAMPLING_TEST): dev/tests/engine/target_sampling_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_Q8_METAL_TEST): dev/tests/engine/q8_paged_kv_metal_test.mm
$(TEST_Q8_STORAGE_TEST): dev/tests/engine/q8_page_storage_test.mm \
		$(ENGINE_LIBRARY)
$(TEST_METAL_BACKEND_TEST): dev/tests/engine/metal_backend_test.mm \
		$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_LIBRARY)
$(TEST_HANDOFF_TEST): dev/tests/engine/handoff_test.mm $(ENGINE_LIBRARY)
$(TEST_VISION_ENCODER_TEST): dev/tests/engine/vision_encoder_test.mm \
		$(ENGINE_LIBRARY)
$(TEST_MODEL_RUNTIME_ORACLE): dev/tests/engine/model_runtime_oracle_test.mm \
		$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_INSTRUMENTED_ANE_OBJECT) $(ENGINE_LIBRARY) $(LIB) \
		$(BUILD_ID_HEADER)
# The oracle remembers the split's calibration under the build, as the engine does.
$(TEST_MODEL_RUNTIME_ORACLE): ENGINE_TEST_CXXFLAGS += -include $(BUILD_ID_HEADER)
$(TEST_DECODE_PROFILE): dev/benchmarks/decode_profile.mm \
		$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_LIBRARY) $(LIB)
$(TEST_ATTENTION_SWEEP): dev/benchmarks/attention_sweep.mm \
		dev/benchmarks/DispatchReplay.hpp \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GGUF_PROJECTION_BENCHMARK): dev/benchmarks/gguf_projection_benchmark.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GGUF_MOE_BENCHMARK): dev/benchmarks/gguf_moe_benchmark.mm \
		dev/benchmarks/DispatchReplay.hpp \
		$(ENGINE_LIBRARY) $(LIB)

$(TEST_DRAFT_PREPARATION) $(TEST_GGUF_PROJECTION) $(TEST_ANE_FFN) \
		$(TEST_ANE_PROGRAM_FAULTS) $(TEST_GGUF_DEQUANT) $(TEST_GGUF_ROTATION) \
		$(TEST_GGUF_REFERENCE) $(TEST_GGUF_PLANNER) $(TEST_GGUF_PREPARATION) \
		$(TEST_BOOTSTRAP_TEST) $(TEST_RESOURCES_TEST) \
		$(TEST_KV_PAGE_TIER_TEST) $(TEST_QWEN_STATE_TEST) $(TEST_Q8_ATTENTION_TEST) \
		$(TEST_Q8_PREFILL_TEST) $(TEST_GGUF_MOE) \
		$(TEST_GDN_METAL_TEST) $(TEST_ATTENTION_PLAN) $(TEST_LINEAR_PLAN) \
		$(TEST_DFLASH_BATCH_CONTROL_TEST) $(TEST_DRAFT_ATTENTION_TEST) \
		$(TEST_GDN_DECODE_TEST) $(TEST_DRAFT_SELECTOR_TEST) $(TEST_LIVE_VERIFY_ROWS) \
		$(TEST_TARGET_SAMPLING_TEST) $(TEST_Q8_METAL_TEST) $(TEST_Q8_STORAGE_TEST) \
		$(TEST_METAL_BACKEND_TEST) $(TEST_HANDOFF_TEST) $(TEST_VISION_ENCODER_TEST) \
		$(TEST_MODEL_RUNTIME_ORACLE) \
		$(TEST_DECODE_PROFILE) $(TEST_ATTENTION_SWEEP) \
		$(TEST_GGUF_PROJECTION_BENCHMARK) $(TEST_GGUF_MOE_BENCHMARK): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

# Tools built with the production flags.
$(WEIGHT_DIGESTS): dev/tools/weight_digests.mm $(ENGINE_LIBRARY)

$(WEIGHT_DIGESTS): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_BACKEND_BENCHMARK): dev/benchmarks/backend_benchmark.mm \
		dev/benchmarks/PrefillWork.hpp \
		$(ENGINE_LIBRARY) $(LIB) $(BUILD_ID_HEADER) \
		| $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -include $(BUILD_ID_HEADER) $< \
		$(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@


# Test kernels, each compiled from its Metal source with the test flags.
$(TEST_Q8_AIR): dev/tests/engine/q8_page_format_oracle.metal runtime/metal/abi/ExecutionGeometry.h
$(TEST_METAL_BACKEND_AIR): dev/tests/engine/metal_backend_test.metal

$(TEST_Q8_AIR) $(TEST_METAL_BACKEND_AIR): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(METAL) $(TEST_METALFLAGS) -c $(filter %.metal,$^) -o $@

$(ENGINE_TEST_BUILD)/kernels/%.air: runtime/metal/kernels/%.metal \
		$(KERNEL_HEADERS) \
		| $(ENGINE_TEST_BUILD)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(METAL) $(TEST_METALFLAGS) -c $< -o $@

# Production flags, not TEST_METALFLAGS: gguf-dequant checks the shipped
# dequantizer bitwise, so it compiles as production does (its AIR does not
# depend on -O3 with metal 32023, but that of some kernels does).
$(TEST_GGUF_DEQUANT_AIR): dev/tests/engine/gguf_dequant_test.metal \
		$(KERNEL_HEADERS) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(METAL) $(PROD_METALFLAGS) -c $< -o $@

# Test libraries, each linked from the AIRs it lists.
$(TEST_GGUF_DEQUANT_LIB): $(TEST_GGUF_DEQUANT_AIR) $(TEST_RESIDENCY_AIR)
$(TEST_Q8_LIB): $(TEST_Q8_AIR) $(TEST_RESIDENCY_AIR)
$(TEST_Q8_ATTENTION_LIB): $(TEST_Q8_KERNEL_AIRS)
$(TEST_METAL_BACKEND_LIB): $(TEST_METAL_BACKEND_AIR) $(TEST_RESIDENCY_AIR)
# The production kernels and the test kernels of metal_backend_test.metal in
# one library, for the kernel tests that order a test copy between production
# dispatches. It relinks whenever the production library does.
$(TEST_PRODUCTION_LIB): $(PRODUCTION_AIRS) $(TEST_METAL_BACKEND_AIR) $(LIB)

$(TEST_GGUF_DEQUANT_LIB) $(TEST_Q8_LIB) $(TEST_Q8_ATTENTION_LIB) $(TEST_METAL_BACKEND_LIB) \
		$(TEST_PRODUCTION_LIB):
	$(RUN_CONFIGURED) $(METALLIB) $(filter %.air,$^) -o $@

.PHONY: verify-build-identity
verify-build-identity: $(TARGET) $(BUILD_ID_HEADER) $(BUILD_ID_STAMP)
	$(BUILD_ID_PYTHON) $(BUILD_ID_SCRIPT) check --root . \
		--header $(BUILD_ID_HEADER) --stamp $(BUILD_ID_STAMP) \
		$(BUILD_ID_CONSTANT_ARGS) --binary $(TARGET)

# Metal kernel tests run with GPU-side bounds checking so a kernel that writes
# past host-sized scratch fails here instead of corrupting memory under the
# real model. (The API debug layer is not enabled: it misestimates the 32-row
# prefill kernels' threadgroup memory, which Metal itself reports as 32768.)
METAL_TEST_ENV := MTL_SHADER_VALIDATION=1
test-engine: test-engine-cpu test-engine-metal

test-engine-cpu: $(TEST_CPU_TARGETS) $(TEST_ATTENTION_SWEEP)
	$(TEST_SLOT_FILE)
	$(TEST_CACHE_DIRECTORY)
	$(BUILD_ID_PYTHON) dev/tests/engine/run_vision_preparation.py $(TEST_VISION_PREPARATION) $(WEIGHT_GOLDENS)
	$(TEST_SAFETENSORS_CHECKPOINT)
	$(TEST_MODEL_CONFIGURATION) $(MODEL_CONFIGS)
	$(TEST_WEIGHT_SOURCE)
	$(TEST_GGUF_FILE)
	$(TEST_GGUF_REFERENCE) $(WEIGHT_GOLDENS) $(MLX_QUANTIZATION_FIXTURE)
	$(TEST_GGUF_PLANNER)
	$(TEST_LINEAR_PLAN) --cpu
	/bin/sh dev/tests/attention_sweep_cli.sh $(TEST_ATTENTION_SWEEP)
	$(TEST_OPERATOR_WORKSPACE)
	$(TEST_ANE_FFN_CALIBRATION)
	$(TEST_ANE_FFN_STARTUP)
	$(TEST_EXECUTION_PLANS)
	$(TEST_MODEL_EXECUTION_PLANS)
	$(TEST_MEMORY_TEST)
	$(TEST_KV_PAGE_CACHE_TEST)
	$(TEST_ENGINE_CACHE_TEST)
	$(TEST_PERSISTENT_CACHE)
	$(TEST_DRAFT_CONTEXT_PLAN_TEST)
	$(TEST_RAGGED_SCHEDULER_TEST)
	$(TEST_ENGINE_TEST)
	$(TEST_ELASTIC_KV_TEST)
	$(TEST_PROTOCOL_TEST) dev/tests/engine/protocol_golden.txt
	$(TEST_NATIVE_LOOP_TEST)
	$(TEST_FD_TRANSPORT_TEST)
	$(TEST_BOOTSTRAP_TEST)
	$(TEST_NATIVE_ARGUMENTS_TEST) dev/tests/engine/native_command_golden.txt $(MODEL_CONFIGS)
	$(TEST_METRICS_TEST)
	$(TEST_MEMORY_AUDIT_TEST)
	$(TEST_MEMORY_GOVERNOR_TEST)
	$(TEST_STATUS_TEST) dev/tests/engine/status_golden.json
	$(TEST_Q8_CPU_TEST)

test-engine-metal: $(TEST_METAL_TARGETS)
	$(METAL_TEST_ENV) $(BUILD_ID_PYTHON) dev/tests/engine/run_draft_preparation.py $(TEST_DRAFT_PREPARATION) $(LIB) \
		$(WEIGHT_GOLDENS)
	$(METAL_TEST_ENV) $(TEST_GGUF_PREPARATION) $(LIB) $(WEIGHT_GOLDENS) $(MLX_QUANTIZATION_FIXTURE)
	$(METAL_TEST_ENV) $(TEST_GGUF_DEQUANT) $(TEST_GGUF_DEQUANT_LIB)
	$(METAL_TEST_ENV) $(TEST_GGUF_ROTATION) $(LIB)
	$(METAL_TEST_ENV) $(TEST_GGUF_PROJECTION) $(TEST_PRODUCTION_LIB)
	$(METAL_TEST_ENV) $(TEST_ANE_FFN) $(LIB) kernels
	$(TEST_ANE_FFN) $(LIB) split
	$(TEST_ANE_FFN) $(LIB) faults
	$(TEST_ANE_FFN) $(LIB) program
	$(TEST_ANE_PROGRAM_FAULTS) $(LIB)
	$(METAL_TEST_ENV) $(TEST_GGUF_MOE) $(LIB)
	$(METAL_TEST_ENV) $(TEST_ATTENTION_PLAN) $(LIB)
	$(METAL_TEST_ENV) $(TEST_LINEAR_PLAN) $(LIB)
	$(METAL_TEST_ENV) $(TEST_RESOURCES_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_KV_PAGE_TIER_TEST) $(TEST_METAL_BACKEND_LIB)
	$(METAL_TEST_ENV) $(TEST_MODEL_LOADING_TEST) $(TEST_METAL_BACKEND_LIB) $(LIB)
	$(METAL_TEST_ENV) $(TEST_QWEN_STATE_TEST) $(TEST_Q8_LIB)
	$(METAL_TEST_ENV) $(TEST_Q8_METAL_TEST) $(TEST_Q8_LIB)
	$(METAL_TEST_ENV) $(TEST_Q8_STORAGE_TEST) $(TEST_METAL_BACKEND_LIB)
	$(METAL_TEST_ENV) $(TEST_Q8_ATTENTION_TEST) $(TEST_Q8_ATTENTION_LIB)
	$(METAL_TEST_ENV) $(TEST_Q8_PREFILL_TEST) $(TEST_Q8_ATTENTION_LIB)
	$(METAL_TEST_ENV) $(TEST_GDN_METAL_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_DFLASH_BATCH_CONTROL_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_DRAFT_ATTENTION_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_GDN_DECODE_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_DRAFT_SELECTOR_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_LIVE_VERIFY_ROWS) $(LIB)
	$(METAL_TEST_ENV) $(TEST_TARGET_SAMPLING_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_VISION_METAL_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_METAL_BACKEND_TEST) $(TEST_METAL_BACKEND_LIB)
	$(METAL_TEST_ENV) $(TEST_HANDOFF_TEST) $(TEST_METAL_BACKEND_LIB)

.PHONY: test-real
# The vision fixture is named after the installed model's family, which its
# model.json records. Nothing is printed when the installation serves no
# vision (vision_format none: --language-only, or a GGUF without an mmproj).
VISION_FIXTURE_FAMILY := import json, pathlib, sys; \
	model = json.loads((pathlib.Path(sys.argv[1]) / "model.json").read_text()); \
	print("" if model["vision_format"] == "none" else model["family"].lower())
test-real: preflight $(TARGET) $(TEST_MODEL_RUNTIME_ORACLE) \
		$(TEST_VISION_ENCODER_TEST) $(LIB)
	family=$$($(BUILD_ID_PYTHON) -c '$(VISION_FIXTURE_FAMILY)' "$(MODEL_ROOT)") && \
		if test -n "$$family"; then \
			$(METAL_TEST_ENV) $(TEST_VISION_ENCODER_TEST) $(LIB) "$(MODEL_ROOT)" \
				dev/tests/fixtures/vision-parity/$$family; \
		else echo "vision parity: skipped, the installation serves text only"; fi
	$(TEST_MODEL_RUNTIME_ORACLE) $(LIB) "$(MODEL_ROOT)" --ane-ffn-share 0
	$(TEST_MODEL_RUNTIME_ORACLE) $(LIB) "$(MODEL_ROOT)"

.PHONY: benchmark-backend \
	benchmark-decode-profile benchmark-attention-sweep \
	benchmark-gguf-projection benchmark-gguf-moe
# decode-profile replays the installed model's prefill and decode commands as
# separate dispatches; DECODE_PROFILE_ARGS passes --prompt-tokens/--cycles.
benchmark-decode-profile: preflight $(TARGET) $(TEST_DECODE_PROFILE) $(LIB)
	$(TEST_DECODE_PROFILE) $(LIB) "$(MODEL_ROOT)" $(DECODE_PROFILE_ARGS)

# Attention kernels alone on one layer of synthetic Q8 history across cache
# lengths; ATTENTION_SWEEP_ARGS passes --histories/--shapes/--lanes/--repeat.
benchmark-attention-sweep: $(TEST_ATTENTION_SWEEP) $(LIB)
	$(TEST_ATTENTION_SWEEP) $(LIB) $(ATTENTION_SWEEP_ARGS)

# One GGUF projection on both decode tiles at every lane count and K split
# (the split tiers of runtime/ops/LinearGguf.cpp), or with prefill=R[,R...]
# on the 128-row prefill tile at each chunk of R rows; GGUF_PROJECTION_ARGS
# passes <fmt[+fmt+fmt]> <N[+N+N]> <K> [none|residual|gateup] [rounds]
# [prefill=R[,R...]].
GGUF_PROJECTION_ARGS ?= q4k 5120 8192
benchmark-gguf-projection: $(TEST_GGUF_PROJECTION_BENCHMARK) $(LIB)
	$(TEST_GGUF_PROJECTION_BENCHMARK) $(LIB) $(GGUF_PROJECTION_ARGS)

# One MoE layer at the 35B shape in GGUF formats, on the device's
# plans and the other GGUF tile (ops/MoE.cpp); GGUF_MOE_ARGS passes [rounds]
# [gate/up format] [down format] (q4k q5k by default).
benchmark-gguf-moe: $(TEST_GGUF_MOE_BENCHMARK) $(LIB)
	$(TEST_GGUF_MOE_BENCHMARK) $(LIB) $(GGUF_MOE_ARGS)

benchmark-backend: preflight $(TARGET) $(TEST_BACKEND_BENCHMARK) $(LIB)
	$(TEST_BACKEND_BENCHMARK) $(LIB) "$(MODEL_ROOT)"

# CPU tests that also run under the sanitizers: each is built three times
# from the same sources, once as the CPU tests above are and once under each
# sanitizer.
$(TEST_SLOT_FILE) $(TEST_SLOT_FILE_ASAN) $(TEST_SLOT_FILE_TSAN): \
		runtime/model/SlotFile.cpp dev/tests/engine/slot_file_test.cpp
$(TEST_CACHE_DIRECTORY) $(TEST_CACHE_DIRECTORY_ASAN) $(TEST_CACHE_DIRECTORY_TSAN): \
		runtime/engine/CacheDirectory.cpp dev/tests/engine/cache_directory_test.cpp
$(TEST_PERSISTENT_CACHE) $(TEST_PERSISTENT_CACHE_ASAN) $(TEST_PERSISTENT_CACHE_TSAN): \
		$(CACHE_SOURCES) runtime/model/SlotFile.cpp dev/tests/engine/persistent_cache_test.cpp
$(TEST_ENGINE_CACHE_TEST) $(TEST_ENGINE_CACHE_ASAN) $(TEST_ENGINE_CACHE_TSAN): \
		$(CACHE_SOURCES) runtime/model/SlotFile.cpp dev/tests/engine/engine_cache_test.cpp
$(TEST_PROTOCOL_TEST) $(TEST_PROTOCOL_ASAN) $(TEST_PROTOCOL_TSAN): \
		runtime/engine/Protocol.cpp dev/tests/engine/ProtocolPeer.cpp \
		dev/tests/engine/protocol_test.cpp
$(TEST_ENGINE_TEST) $(TEST_ENGINE_ASAN) $(TEST_ENGINE_TSAN): \
		$(BACKEND_CONTROL_SOURCES) \
		dev/benchmarks/PrefillWork.hpp \
		dev/tests/engine/engine_test.cpp
$(TEST_FD_TRANSPORT_TEST) $(TEST_FD_TRANSPORT_ASAN) $(TEST_FD_TRANSPORT_TSAN): \
		$(NATIVE_RUNTIME_SOURCES) \
		runtime/engine/FdTransport.cpp \
		dev/tests/engine/ProtocolPeer.cpp \
		dev/tests/engine/native_fd_transport_test.cpp
$(TEST_ANE_FFN_CALIBRATION) $(TEST_ANE_FFN_CALIBRATION_ASAN) $(TEST_ANE_FFN_CALIBRATION_TSAN): \
		runtime/ops/AneFfnCalibration.cpp dev/tests/engine/ane_ffn_calibration_test.cpp

$(filter %-asan-ubsan,$(SANITIZER_CONFIG_TARGETS)): SANITIZERS := address,undefined
$(filter %-tsan,$(SANITIZER_CONFIG_TARGETS)): SANITIZERS := thread
$(SANITIZER_CONFIG_TARGETS): | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=$(SANITIZERS) \
		$(TEST_INPUTS) -o $@

.PHONY: test-sanitizers
ASAN_TEST_ENV := ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
TSAN_TEST_ENV := TSAN_OPTIONS=halt_on_error=1
test-sanitizers: $(SANITIZER_CONFIG_TARGETS)
	$(ASAN_TEST_ENV) $(TEST_SLOT_FILE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_SLOT_FILE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_CACHE_DIRECTORY_ASAN)
	$(TSAN_TEST_ENV) $(TEST_CACHE_DIRECTORY_TSAN)
	$(ASAN_TEST_ENV) $(TEST_PERSISTENT_CACHE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_PERSISTENT_CACHE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_ENGINE_CACHE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_ENGINE_CACHE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_PROTOCOL_ASAN) dev/tests/engine/protocol_golden.txt
	$(TSAN_TEST_ENV) $(TEST_PROTOCOL_TSAN) dev/tests/engine/protocol_golden.txt
	$(ASAN_TEST_ENV) $(TEST_ENGINE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_ENGINE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_FD_TRANSPORT_ASAN)
	$(TSAN_TEST_ENV) $(TEST_FD_TRANSPORT_TSAN)
	$(ASAN_TEST_ENV) $(TEST_ANE_FFN_CALIBRATION_ASAN)
	$(TSAN_TEST_ENV) $(TEST_ANE_FFN_CALIBRATION_TSAN)
