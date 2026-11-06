ENGINE_TEST_BUILD := $(BUILD)/engine-tests
ENGINE_TEST_CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror -Iruntime -Idev \
	$(MACOS_TARGET_FLAG)
# Production flags less -O3, so Metal optimizes at its default -O2. With
# metal 32023 the kernels built with them get the same AIR as at -O3, but
# three production kernels do not (shared/gguf_linear, moe_gguf,
# normalization): a test that recompiles those needs PROD_METALFLAGS.
TEST_METALFLAGS := $(filter-out -O3,$(PROD_METALFLAGS))
# Outputs of this file's rules record its digest and the production C++
# configuration; those built with the production C++ flags record no more
# (PRODUCTION_FLAG_TOOLS).
NATIVE_CONFIG_DIGEST := $(shell printf '%s\0' $(CONFIG_DIGEST) \
	"$$(shasum -a 256 dev/native.mk)" | shasum -a 256 | cut -c1-16)
# Test kernels build with METAL and METALLIB, and three with PROD_METALFLAGS
# (above), so the production Metal configuration is part of theirs.
TEST_CONFIG_DIGEST := $(shell printf '%s\0' $(NATIVE_CONFIG_DIGEST) \
	$(METAL_CONFIG_DIGEST) $(call shell-quote,$(ENGINE_TEST_CXXFLAGS)) \
	$(call shell-quote,$(TEST_METALFLAGS)) | shasum -a 256 | cut -c1-16)
ENGINE_SANITIZER_BUILD := $(BUILD)/sanitizers
ENGINE_SANITIZER_CXXFLAGS := -std=c++20 -O1 -g -fno-omit-frame-pointer \
	-Wall -Wextra -Werror -Iruntime -Idev $(MACOS_TARGET_FLAG)
SANITIZER_CONFIG_DIGEST := $(shell printf '%s\0' $(NATIVE_CONFIG_DIGEST) \
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
	runtime/engine/Engine.cpp \
	runtime/engine/EngineMemory.cpp \
	runtime/engine/PassCommand.cpp \
	runtime/engine/WeightResidency.cpp
NATIVE_RUNTIME_SOURCES := $(BACKEND_CONTROL_SOURCES) \
	runtime/engine/Protocol.cpp \
	runtime/engine/NativeRuntime.cpp
TEST_SLOT_FILE_ASAN := $(ENGINE_SANITIZER_BUILD)/slot-file-asan-ubsan
TEST_SLOT_FILE_TSAN := $(ENGINE_SANITIZER_BUILD)/slot-file-tsan
TEST_KV_SLOT_FORMAT_ASAN := $(ENGINE_SANITIZER_BUILD)/kv-slot-format-asan-ubsan
TEST_KV_PAGE_TIER_HOST_ASAN := $(ENGINE_SANITIZER_BUILD)/kv-page-tier-host-asan-ubsan
TEST_KV_PAGE_TIER_HOST_TSAN := $(ENGINE_SANITIZER_BUILD)/kv-page-tier-host-tsan
TEST_DISK_SPACE_ASAN := $(ENGINE_SANITIZER_BUILD)/disk-space-asan-ubsan
TEST_DISK_SPACE_TSAN := $(ENGINE_SANITIZER_BUILD)/disk-space-tsan
TEST_CACHE_DIRECTORY_ASAN := $(ENGINE_SANITIZER_BUILD)/cache-directory-asan-ubsan
TEST_CACHE_DIRECTORY_TSAN := $(ENGINE_SANITIZER_BUILD)/cache-directory-tsan
TEST_PERSISTENT_CACHE_ASAN := $(ENGINE_SANITIZER_BUILD)/persistent-cache-asan-ubsan
TEST_PERSISTENT_CACHE_TSAN := $(ENGINE_SANITIZER_BUILD)/persistent-cache-tsan
TEST_ENGINE_CACHE_ASAN := $(ENGINE_SANITIZER_BUILD)/engine-cache-asan-ubsan
TEST_ENGINE_CACHE_TSAN := $(ENGINE_SANITIZER_BUILD)/engine-cache-tsan
TEST_CACHE_PROPERTY_ASAN := $(ENGINE_SANITIZER_BUILD)/cache-property-asan-ubsan
TEST_CACHE_PROPERTY_TSAN := $(ENGINE_SANITIZER_BUILD)/cache-property-tsan
TEST_PROTOCOL_ASAN := $(ENGINE_SANITIZER_BUILD)/protocol-asan-ubsan
TEST_PROTOCOL_TSAN := $(ENGINE_SANITIZER_BUILD)/protocol-tsan
TEST_ENGINE_ASAN := $(ENGINE_SANITIZER_BUILD)/engine-asan-ubsan
TEST_ENGINE_TSAN := $(ENGINE_SANITIZER_BUILD)/engine-tsan
TEST_RAGGED_SCHEDULER_ASAN := $(ENGINE_SANITIZER_BUILD)/ragged-scheduler-asan-ubsan
TEST_RAGGED_SCHEDULER_TSAN := $(ENGINE_SANITIZER_BUILD)/ragged-scheduler-tsan
TEST_MODEL_BOUNDARY_ASAN := $(ENGINE_SANITIZER_BUILD)/model-boundary-asan-ubsan
TEST_MODEL_BOUNDARY_TSAN := $(ENGINE_SANITIZER_BUILD)/model-boundary-tsan
TEST_FAKE_MEMORY_ASAN := $(ENGINE_SANITIZER_BUILD)/fake-memory-asan-ubsan
TEST_FAKE_MEMORY_TSAN := $(ENGINE_SANITIZER_BUILD)/fake-memory-tsan
TEST_TIMING_PROFILE_ASAN := $(ENGINE_SANITIZER_BUILD)/timing-profile-asan-ubsan
TEST_TIMING_PROFILE_TSAN := $(ENGINE_SANITIZER_BUILD)/timing-profile-tsan
TEST_FD_TRANSPORT_ASAN := $(ENGINE_SANITIZER_BUILD)/native-fd-asan-ubsan
TEST_FD_TRANSPORT_TSAN := $(ENGINE_SANITIZER_BUILD)/native-fd-tsan
TEST_NATIVE_LOOP_ASAN := $(ENGINE_SANITIZER_BUILD)/native-engine-loop-asan-ubsan
TEST_NATIVE_LOOP_TSAN := $(ENGINE_SANITIZER_BUILD)/native-engine-loop-tsan
TEST_ENGINE_SIM_ASAN := $(ENGINE_SANITIZER_BUILD)/engine-sim-asan-ubsan
TEST_ENGINE_SIM_TSAN := $(ENGINE_SANITIZER_BUILD)/engine-sim-tsan
TEST_ANE_FFN_CALIBRATION_ASAN := $(ENGINE_SANITIZER_BUILD)/ane-ffn-calibration-asan-ubsan
TEST_ANE_FFN_CALIBRATION_TSAN := $(ENGINE_SANITIZER_BUILD)/ane-ffn-calibration-tsan
TEST_SPARSE_ROW_STORE_ASAN := $(ENGINE_SANITIZER_BUILD)/sparse-row-store-asan-ubsan
TEST_SPARSE_ROW_STORE_TSAN := $(ENGINE_SANITIZER_BUILD)/sparse-row-store-tsan
TEST_SSD_READERS_ASAN := $(ENGINE_SANITIZER_BUILD)/ssd-readers-asan-ubsan
TEST_SSD_READERS_TSAN := $(ENGINE_SANITIZER_BUILD)/ssd-readers-tsan
TEST_ELASTIC_KV_ASAN := $(ENGINE_SANITIZER_BUILD)/elastic-kv-pool-asan-ubsan
TEST_ELASTIC_KV_TSAN := $(ENGINE_SANITIZER_BUILD)/elastic-kv-pool-tsan
TEST_SLOT_RECYCLER_ASAN := $(ENGINE_SANITIZER_BUILD)/slot-recycler-asan-ubsan
TEST_SLOT_RECYCLER_TSAN := $(ENGINE_SANITIZER_BUILD)/slot-recycler-tsan
# The memory plan, its audit and /status: arithmetic and JSON on one thread.
TEST_MEMORY_PLAN_ASAN := $(ENGINE_SANITIZER_BUILD)/engine-memory-plan-asan-ubsan
TEST_MEMORY_AUDIT_ASAN := $(ENGINE_SANITIZER_BUILD)/memory-audit-asan-ubsan
TEST_STATUS_ASAN := $(ENGINE_SANITIZER_BUILD)/runtime-status-asan-ubsan
TEST_JSON_WRITER_ASAN := $(ENGINE_SANITIZER_BUILD)/json-writer-asan-ubsan
TEST_MEMORY_LEDGER_ASAN := $(ENGINE_SANITIZER_BUILD)/memory-ledger-asan-ubsan
TEST_MEMORY_TEST := $(ENGINE_TEST_BUILD)/engine-memory-plan
TEST_VISION_PREPARATION := $(ENGINE_TEST_BUILD)/vision-preparation
TEST_DRAFT_PREPARATION := $(ENGINE_TEST_BUILD)/draft-preparation
TEST_SAFETENSORS_CHECKPOINT := $(ENGINE_TEST_BUILD)/safetensors-checkpoint
TEST_MODEL_CONFIGURATION := $(ENGINE_TEST_BUILD)/model-configuration
TEST_FAMILY_VALUES := $(ENGINE_TEST_BUILD)/family-values
TEST_WEIGHT_SOURCE := $(ENGINE_TEST_BUILD)/weight-source
TEST_LEGACY_WEIGHT_CACHE := $(ENGINE_TEST_BUILD)/legacy-weight-cache
TEST_GGUF_FILE := $(ENGINE_TEST_BUILD)/gguf-file
TEST_GGUF_PROJECTION := $(ENGINE_TEST_BUILD)/gguf-projection
TEST_ANE_FFN := $(ENGINE_TEST_BUILD)/ane-ffn
TEST_ANE_FFN_CALIBRATION := $(ENGINE_TEST_BUILD)/ane-ffn-calibration
TEST_ANE_FFN_STARTUP := $(ENGINE_TEST_BUILD)/ane-ffn-startup
TEST_ANE_PROGRAM_FAULTS := $(ENGINE_TEST_BUILD)/ane-program-faults
TEST_GGUF_DEQUANT := $(ENGINE_TEST_BUILD)/gguf-dequant
TEST_GGUF_ROTATION := $(ENGINE_TEST_BUILD)/gguf-rotation
TEST_GGUF_MOE := $(ENGINE_TEST_BUILD)/gguf-moe
TEST_GGUF_MOE_E512 := $(ENGINE_TEST_BUILD)/gguf-moe-e512
TEST_GGUF_MOE_SIGMOID := $(ENGINE_TEST_BUILD)/gguf-moe-sigmoid
TEST_EXPERT_STORE := $(ENGINE_TEST_BUILD)/expert-store
TEST_KV_BLOCK_CACHE := $(ENGINE_TEST_BUILD)/kv-block-cache
TEST_KV_HISTORY_STREAM := $(ENGINE_TEST_BUILD)/kv-history-stream
TEST_QWEN4_KERNELS := $(ENGINE_TEST_BUILD)/qwen4-kernels
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
TEST_CACHE_PROPERTY := $(ENGINE_TEST_BUILD)/cache-property
TEST_DRAFT_CONTEXT_PLAN_TEST := $(ENGINE_TEST_BUILD)/draft-context-plan
TEST_RAGGED_SCHEDULER_TEST := $(ENGINE_TEST_BUILD)/ragged-scheduler
TEST_MODEL_BOUNDARY_TEST := $(ENGINE_TEST_BUILD)/model-boundary
TEST_WEIGHT_RESIDENCY := $(ENGINE_TEST_BUILD)/weight-residency
TEST_FAKE_MEMORY := $(ENGINE_TEST_BUILD)/fake-memory
TEST_TIMING_PROFILE := $(ENGINE_TEST_BUILD)/timing-profile
TEST_ENGINE_TEST := $(ENGINE_TEST_BUILD)/engine
TEST_ELASTIC_KV_TEST := $(ENGINE_TEST_BUILD)/elastic-kv-pool
TEST_PROTOCOL_TEST := $(ENGINE_TEST_BUILD)/protocol
TEST_NATIVE_LOOP_TEST := $(ENGINE_TEST_BUILD)/native-engine-loop
TEST_ENGINE_SIM := $(ENGINE_TEST_BUILD)/engine-sim
TEST_PRESSURE_SERVING := $(ENGINE_TEST_BUILD)/pressure-serving
TEST_FD_TRANSPORT_TEST := $(ENGINE_TEST_BUILD)/native-fd-transport
TEST_BOOTSTRAP_TEST := $(ENGINE_TEST_BUILD)/runtime-bootstrap
TEST_NATIVE_ARGUMENTS_TEST := $(ENGINE_TEST_BUILD)/native-arguments
TEST_RESOURCES_TEST := $(ENGINE_TEST_BUILD)/runtime-resources
TEST_KV_PAGE_TIER_TEST := $(ENGINE_TEST_BUILD)/kv-page-tier
TEST_METRICS_TEST := $(ENGINE_TEST_BUILD)/runtime-metrics
TEST_MODEL_LOADING_TEST := $(ENGINE_TEST_BUILD)/model-loading
TEST_MEMORY_AUDIT_TEST := $(ENGINE_TEST_BUILD)/memory-audit
TEST_MEMORY_GOVERNOR_TEST := $(ENGINE_TEST_BUILD)/memory-governor
TEST_MEMORY_LEDGER := $(ENGINE_TEST_BUILD)/memory-ledger
TEST_QWEN_STATE_TEST := $(ENGINE_TEST_BUILD)/qwen-state-storage
TEST_STATUS_TEST := $(ENGINE_TEST_BUILD)/runtime-status
TEST_JSON_WRITER := $(ENGINE_TEST_BUILD)/json-writer
TEST_Q8_CPU_TEST := $(ENGINE_TEST_BUILD)/q8-paged-kv
TEST_Q8_METAL_TEST := $(ENGINE_TEST_BUILD)/q8-paged-kv-metal
TEST_Q8_STORAGE_TEST := $(ENGINE_TEST_BUILD)/q8-page-storage
TEST_Q8_ATTENTION_TEST := $(ENGINE_TEST_BUILD)/q8-flash-attention
TEST_Q8_PREFILL_TEST := $(ENGINE_TEST_BUILD)/q8-chunked-prefill
TEST_GDN_METAL_TEST := $(ENGINE_TEST_BUILD)/gdn-metal
TEST_OPERATOR_WORKSPACE := $(ENGINE_TEST_BUILD)/operator-workspace
TEST_EXECUTION_PLANS := $(ENGINE_TEST_BUILD)/execution-plans
TEST_KERNEL_SELECTION := $(ENGINE_TEST_BUILD)/kernel-selection
TEST_MODEL_EXECUTION_PLANS := $(ENGINE_TEST_BUILD)/model-execution-plans
TEST_COMMAND_LIMITS := $(ENGINE_TEST_BUILD)/command-limits
TEST_ATTENTION_PLAN := $(ENGINE_TEST_BUILD)/paged-attention-plan
TEST_ATTENTION_VARIANTS := $(ENGINE_TEST_BUILD)/attention-variants
TEST_LINEAR_PLAN := $(ENGINE_TEST_BUILD)/linear-plan
TEST_DFLASH_BATCH_CONTROL_TEST := $(ENGINE_TEST_BUILD)/dflash-batch-control
TEST_DRAFT_ATTENTION_TEST := $(ENGINE_TEST_BUILD)/draft-attention
TEST_GDN_DECODE_TEST := $(ENGINE_TEST_BUILD)/gdn-decode
TEST_DRAFT_SELECTOR_TEST := $(ENGINE_TEST_BUILD)/draft-selector
TEST_TARGET_SAMPLING_TEST := $(ENGINE_TEST_BUILD)/target-sampling
TEST_VISION_METAL_TEST := $(ENGINE_TEST_BUILD)/vision-metal
TEST_VISION_ATTENTION_TEST := $(ENGINE_TEST_BUILD)/vision-attention
TEST_BACKEND_BENCHMARK := $(ENGINE_TEST_BUILD)/backend-benchmark
TEST_DECODE_PROFILE := $(ENGINE_TEST_BUILD)/decode-profile
TEST_TEACHER_LOGITS := $(ENGINE_TEST_BUILD)/teacher-logits
TEST_ATTENTION_SWEEP := $(ENGINE_TEST_BUILD)/attention-sweep
TEST_GGUF_PROJECTION_BENCHMARK := $(ENGINE_TEST_BUILD)/gguf-projection-benchmark
TEST_GGUF_MOE_BENCHMARK := $(ENGINE_TEST_BUILD)/gguf-moe-benchmark
TEST_MODEL_RUNTIME_ORACLE := $(ENGINE_TEST_BUILD)/model-runtime-oracle
WEIGHT_DIGESTS := $(ENGINE_TEST_BUILD)/weight-digests
TEST_VISION_ENCODER_TEST := $(ENGINE_TEST_BUILD)/vision-encoder
TEST_Q8_AIR := $(ENGINE_TEST_BUILD)/q8-paged-kv.air
TEST_Q8_LIB := $(ENGINE_TEST_BUILD)/q8-paged-kv.metallib
# Both Q8 tests share the attention and store kernels of both phases, the
# sparse ones included.
TEST_Q8_KERNEL_SOURCES := \
	runtime/metal/kernels/prefill/paged_attention.metal \
	runtime/metal/kernels/decode/paged_attention.metal \
	runtime/metal/kernels/prefill/paged_attention_store.metal \
	runtime/metal/kernels/decode/paged_attention_store.metal \
	runtime/metal/kernels/shared/sparse_attention.metal
TEST_Q8_KERNEL_AIRS := $(patsubst runtime/metal/kernels/%.metal,$(ENGINE_TEST_BUILD)/kernels/%.air,$(TEST_Q8_KERNEL_SOURCES))
# Every library a MetalBackend loads has the kernel that ends its residency;
# the test libraries built without the production kernels link it in.
TEST_RESIDENCY_AIR := $(ENGINE_TEST_BUILD)/kernels/shared/residency.air
# The host-step library beside the test libraries (HOST_STEP_LIB), which
# unlike the engine's must build: the Metal suite checks the spin.
TEST_HOST_STEP_AIR := $(ENGINE_TEST_BUILD)/kernels/host_step.air
TEST_HOST_STEP_LIB := $(ENGINE_TEST_BUILD)/host-step.metallib
TEST_Q8_ATTENTION_LIB := $(ENGINE_TEST_BUILD)/q8-attention.metallib
TEST_METAL_BACKEND_TEST := $(ENGINE_TEST_BUILD)/metal-backend
TEST_HANDOFF_TEST := $(ENGINE_TEST_BUILD)/handoff
TEST_METAL_BACKEND_AIR := $(ENGINE_TEST_BUILD)/metal-backend.air
TEST_METAL_BACKEND_LIB := $(ENGINE_TEST_BUILD)/metal-backend.metallib
TEST_PRODUCTION_LIB := $(ENGINE_TEST_BUILD)/production-and-test.metallib

TEST_SLOT_FILE := $(ENGINE_TEST_BUILD)/slot-file
TEST_KV_SLOT_FORMAT := $(ENGINE_TEST_BUILD)/kv-slot-format
TEST_KV_PAGE_TIER_HOST := $(ENGINE_TEST_BUILD)/kv-page-tier-host
TEST_SPARSE_ROW_STORE := $(ENGINE_TEST_BUILD)/sparse-row-store
TEST_SSD_READERS := $(ENGINE_TEST_BUILD)/ssd-readers
TEST_SLOT_RECYCLER := $(ENGINE_TEST_BUILD)/slot-recycler
TEST_EXPERT_CHUNK_LAYOUT := $(ENGINE_TEST_BUILD)/expert-chunk-layout
TEST_VERIFY_BUDGET := $(ENGINE_TEST_BUILD)/verify-budget
TEST_CACHE_DIRECTORY := $(ENGINE_TEST_BUILD)/cache-directory
TEST_DISK_SPACE := $(ENGINE_TEST_BUILD)/disk-space
TEST_PERSISTENT_CACHE := $(ENGINE_TEST_BUILD)/persistent-cache

TEST_CPU_TARGETS := $(TEST_SLOT_FILE) $(TEST_SPARSE_ROW_STORE) $(TEST_SSD_READERS) $(TEST_SLOT_RECYCLER) \
	$(TEST_EXPERT_CHUNK_LAYOUT) $(TEST_VERIFY_BUDGET) \
	$(TEST_CACHE_DIRECTORY) $(TEST_DISK_SPACE) $(TEST_LEGACY_WEIGHT_CACHE) $(TEST_VISION_PREPARATION) $(TEST_SAFETENSORS_CHECKPOINT) $(TEST_WEIGHT_SOURCE) $(TEST_OPERATOR_WORKSPACE) \
	$(TEST_MODEL_CONFIGURATION) $(TEST_FAMILY_VALUES) \
	$(TEST_KV_SLOT_FORMAT) $(TEST_KV_PAGE_TIER_HOST) \
	$(TEST_GGUF_FILE) \
	$(TEST_GGUF_REFERENCE) $(TEST_GGUF_PLANNER) \
	$(TEST_LINEAR_PLAN) \
	$(TEST_ANE_FFN_CALIBRATION) \
	$(TEST_ANE_FFN_STARTUP) \
	$(TEST_EXECUTION_PLANS) \
	$(TEST_KERNEL_SELECTION) \
	$(TEST_MODEL_EXECUTION_PLANS) \
	$(TEST_COMMAND_LIMITS) \
	$(TEST_MEMORY_TEST) \
	$(TEST_KV_PAGE_CACHE_TEST) \
	$(TEST_ENGINE_CACHE_TEST) \
	$(TEST_CACHE_PROPERTY) \
	$(TEST_PERSISTENT_CACHE) \
	$(TEST_DRAFT_CONTEXT_PLAN_TEST) \
	$(TEST_RAGGED_SCHEDULER_TEST) \
	$(TEST_MODEL_BOUNDARY_TEST) \
	$(TEST_WEIGHT_RESIDENCY) \
	$(TEST_FAKE_MEMORY) \
	$(TEST_TIMING_PROFILE) \
	$(TEST_ENGINE_TEST) \
	$(TEST_ELASTIC_KV_TEST) \
	$(TEST_PROTOCOL_TEST) \
	$(TEST_NATIVE_LOOP_TEST) \
	$(TEST_ENGINE_SIM) \
	$(TEST_FD_TRANSPORT_TEST) \
	$(TEST_BOOTSTRAP_TEST) \
	$(TEST_NATIVE_ARGUMENTS_TEST) \
	$(TEST_METRICS_TEST) \
	$(TEST_MEMORY_AUDIT_TEST) \
	$(TEST_MEMORY_GOVERNOR_TEST) \
	$(TEST_MEMORY_LEDGER) \
	$(TEST_STATUS_TEST) \
	$(TEST_JSON_WRITER) \
	$(TEST_Q8_CPU_TEST)

TEST_METAL_TARGETS := $(TEST_DRAFT_PREPARATION) \
	$(TEST_GGUF_PROJECTION) \
	$(TEST_ANE_FFN) \
	$(TEST_ANE_PROGRAM_FAULTS) \
	$(TEST_GGUF_DEQUANT) \
	$(TEST_GGUF_ROTATION) \
	$(TEST_GGUF_MOE) \
	$(TEST_GGUF_MOE_E512) \
	$(TEST_GGUF_MOE_SIGMOID) \
	$(TEST_EXPERT_STORE) \
	$(TEST_KV_BLOCK_CACHE) \
	$(TEST_KV_HISTORY_STREAM) \
	$(TEST_QWEN4_KERNELS) \
	$(TEST_GGUF_PREPARATION) \
	$(TEST_ATTENTION_PLAN) \
	$(TEST_ATTENTION_VARIANTS) \
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
	$(TEST_TARGET_SAMPLING_TEST) \
	$(TEST_VISION_METAL_TEST) \
	$(TEST_VISION_ATTENTION_TEST) \
	$(TEST_METAL_BACKEND_TEST) \
	$(TEST_HANDOFF_TEST) \
	$(TEST_PRESSURE_SERVING) \
	$(LIB) $(HOST_STEP_LIB) $(TEST_METAL_BACKEND_LIB) $(TEST_PRODUCTION_LIB) $(TEST_HOST_STEP_LIB) \
	$(TEST_Q8_LIB) $(TEST_Q8_ATTENTION_LIB) $(TEST_GGUF_DEQUANT_LIB)

# Keep every output that uses a flag set together, including standalone
# benchmarks, real-model tests and intermediate test AIRs/metallibs.
TEST_CONFIG_TARGETS := $(filter-out $(LIB) $(HOST_STEP_LIB),$(sort $(TEST_CPU_TARGETS) $(TEST_METAL_TARGETS))) \
	$(TEST_MODEL_RUNTIME_ORACLE) $(TEST_VISION_ENCODER_TEST) \
	$(TEST_DECODE_PROFILE) $(TEST_TEACHER_LOGITS) $(TEST_ATTENTION_SWEEP) \
	$(TEST_GGUF_PROJECTION_BENCHMARK) $(TEST_GGUF_MOE_BENCHMARK) \
	$(TEST_Q8_AIR) $(TEST_Q8_KERNEL_AIRS) $(TEST_RESIDENCY_AIR) $(TEST_HOST_STEP_AIR) $(TEST_METAL_BACKEND_AIR) \
	$(TEST_GGUF_DEQUANT_AIR)
# Benchmarks and tools that build with the production flags.
PRODUCTION_FLAG_TOOLS := $(TEST_BACKEND_BENCHMARK) $(WEIGHT_DIGESTS)
SANITIZER_CONFIG_TARGETS := $(TEST_SLOT_FILE_ASAN) $(TEST_SLOT_FILE_TSAN) \
	$(TEST_KV_SLOT_FORMAT_ASAN) \
	$(TEST_KV_PAGE_TIER_HOST_ASAN) $(TEST_KV_PAGE_TIER_HOST_TSAN) \
	$(TEST_DISK_SPACE_ASAN) $(TEST_DISK_SPACE_TSAN) \
	$(TEST_CACHE_DIRECTORY_ASAN) $(TEST_CACHE_DIRECTORY_TSAN) \
	$(TEST_PERSISTENT_CACHE_ASAN) $(TEST_PERSISTENT_CACHE_TSAN) \
	$(TEST_ENGINE_CACHE_ASAN) $(TEST_ENGINE_CACHE_TSAN) \
	$(TEST_CACHE_PROPERTY_ASAN) $(TEST_CACHE_PROPERTY_TSAN) \
	$(TEST_PROTOCOL_ASAN) $(TEST_PROTOCOL_TSAN) \
	$(TEST_ENGINE_ASAN) $(TEST_ENGINE_TSAN) \
	$(TEST_RAGGED_SCHEDULER_ASAN) $(TEST_RAGGED_SCHEDULER_TSAN) \
	$(TEST_MODEL_BOUNDARY_ASAN) $(TEST_MODEL_BOUNDARY_TSAN) \
	$(TEST_FAKE_MEMORY_ASAN) $(TEST_FAKE_MEMORY_TSAN) \
	$(TEST_TIMING_PROFILE_ASAN) $(TEST_TIMING_PROFILE_TSAN) \
	$(TEST_FD_TRANSPORT_ASAN) $(TEST_FD_TRANSPORT_TSAN) \
	$(TEST_NATIVE_LOOP_ASAN) $(TEST_NATIVE_LOOP_TSAN) \
	$(TEST_ENGINE_SIM_ASAN) $(TEST_ENGINE_SIM_TSAN) \
	$(TEST_ANE_FFN_CALIBRATION_ASAN) $(TEST_ANE_FFN_CALIBRATION_TSAN) \
	$(TEST_SPARSE_ROW_STORE_ASAN) $(TEST_SPARSE_ROW_STORE_TSAN) \
	$(TEST_SSD_READERS_ASAN) $(TEST_SSD_READERS_TSAN) \
	$(TEST_ELASTIC_KV_ASAN) $(TEST_ELASTIC_KV_TSAN) \
	$(TEST_SLOT_RECYCLER_ASAN) $(TEST_SLOT_RECYCLER_TSAN) \
	$(TEST_MEMORY_PLAN_ASAN) $(TEST_MEMORY_AUDIT_ASAN) $(TEST_STATUS_ASAN) $(TEST_JSON_WRITER_ASAN) \
	$(TEST_MEMORY_LEDGER_ASAN)

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
$(TEST_VERIFY_BUDGET): dev/tests/engine/verify_budget_test.cpp runtime/model/VerifyBudget.cpp
$(TEST_EXPERT_CHUNK_LAYOUT): dev/tests/engine/expert_chunk_layout_test.cpp
$(TEST_GGUF_FILE): dev/tests/engine/gguf_file_test.cpp runtime/model/GgufFile.cpp \
		runtime/model/WeightSource.cpp
$(TEST_KV_PAGE_CACHE_TEST): runtime/engine/KvPool.cpp \
		runtime/engine/KvCache.cpp \
		dev/tests/engine/kv_page_cache_test.cpp
$(TEST_DRAFT_CONTEXT_PLAN_TEST): runtime/model/DraftContextPlan.cpp \
		dev/benchmarks/PrefillWork.hpp \
		dev/tests/engine/draft_context_plan_test.cpp
$(TEST_METRICS_TEST): $(BACKEND_CONTROL_SOURCES) \
		runtime/engine/MemoryGovernor.cpp runtime/engine/MemoryLedger.cpp \
		dev/tests/engine/runtime_metrics_test.cpp
$(TEST_MEMORY_GOVERNOR_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryGovernor.cpp \
		runtime/engine/MemoryLedger.cpp \
		dev/tests/engine/memory_governor_test.cpp
$(TEST_MEMORY_LEDGER) $(TEST_MEMORY_LEDGER_ASAN): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryGovernor.cpp runtime/engine/MemoryLedger.cpp \
		dev/tests/engine/memory_ledger_test.cpp
$(TEST_Q8_CPU_TEST): dev/tests/engine/q8_paged_kv_test.cc
$(TEST_LEGACY_WEIGHT_CACHE): dev/tests/engine/legacy_weight_cache_test.cpp

$(TEST_WEIGHT_SOURCE) $(TEST_GGUF_FILE) $(TEST_MEMORY_TEST) $(TEST_KV_PAGE_CACHE_TEST) \
		$(TEST_DRAFT_CONTEXT_PLAN_TEST) $(TEST_RAGGED_SCHEDULER_TEST) $(TEST_MODEL_BOUNDARY_TEST) \
		$(TEST_WEIGHT_RESIDENCY) \
		$(TEST_ELASTIC_KV_TEST) $(TEST_NATIVE_LOOP_TEST) $(TEST_ENGINE_SIM) $(TEST_METRICS_TEST) \
		$(TEST_MEMORY_AUDIT_TEST) $(TEST_MEMORY_GOVERNOR_TEST) $(TEST_MEMORY_LEDGER) $(TEST_STATUS_TEST) \
		$(TEST_JSON_WRITER) \
		$(TEST_Q8_CPU_TEST) $(TEST_SLOT_FILE) $(TEST_SPARSE_ROW_STORE) $(TEST_SSD_READERS) $(TEST_SLOT_RECYCLER) \
		$(TEST_EXPERT_CHUNK_LAYOUT) $(TEST_VERIFY_BUDGET) \
		$(TEST_CACHE_DIRECTORY) $(TEST_DISK_SPACE) $(TEST_KV_SLOT_FORMAT) $(TEST_KV_PAGE_TIER_HOST) \
		$(TEST_LEGACY_WEIGHT_CACHE) $(TEST_PERSISTENT_CACHE) $(TEST_ENGINE_CACHE_TEST) $(TEST_CACHE_PROPERTY) \
		$(TEST_PROTOCOL_TEST) \
		$(TEST_FAKE_MEMORY) $(TEST_TIMING_PROFILE) \
		$(TEST_ENGINE_TEST) $(TEST_FD_TRANSPORT_TEST) $(TEST_ANE_FFN_CALIBRATION): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

# C++ tests linked with the frameworks.
$(TEST_SAFETENSORS_CHECKPOINT): dev/tests/engine/safetensors_checkpoint_test.cpp $(ENGINE_LIBRARY)
$(TEST_FAMILY_VALUES): dev/tests/engine/family_values_test.cpp $(ENGINE_LIBRARY)
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
$(TEST_KERNEL_SELECTION): dev/tests/engine/kernel_selection_test.cc \
		$(ENGINE_LIBRARY)
$(TEST_MODEL_EXECUTION_PLANS): dev/tests/engine/model_execution_plan_test.cpp \
		$(ENGINE_LIBRARY)
$(TEST_VISION_METAL_TEST): dev/tests/engine/vision_metal_test.cpp \
		$(ENGINE_LIBRARY) $(LIB)

$(TEST_SAFETENSORS_CHECKPOINT) $(TEST_FAMILY_VALUES) $(TEST_VISION_PREPARATION) \
		$(TEST_NATIVE_ARGUMENTS_TEST) $(TEST_ANE_FFN_STARTUP) $(TEST_MODEL_LOADING_TEST) $(TEST_OPERATOR_WORKSPACE) \
		$(TEST_EXECUTION_PLANS) $(TEST_KERNEL_SELECTION) $(TEST_MODEL_EXECUTION_PLANS) $(TEST_VISION_METAL_TEST): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@
$(TEST_COMMAND_LIMITS): dev/tests/engine/command_limits_test.cpp $(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) $(ENGINE_LINKFLAGS) -o $@

# Objective-C++ tests, oracles and benchmarks linked with the frameworks.
$(TEST_MODEL_CONFIGURATION): dev/tests/engine/model_configuration_test.mm $(ENGINE_LIBRARY)
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
$(TEST_RESOURCES_TEST): dev/tests/engine/runtime_resources_test.mm dev/tools/LoadOptions.hpp \
		dev/tests/engine/ProtocolPeer.cpp $(ENGINE_LIBRARY) $(LIB)
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
# The same test at Qwen3.8-Flash-Next's 512 experts, 10 per token and expert
# width 640.
$(TEST_GGUF_MOE_E512): dev/tests/engine/gguf_moe_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GGUF_MOE_E512): ENGINE_TEST_CXXFLAGS += -DGGUF_MOE_HIDDEN=256 -DGGUF_MOE_INTERMEDIATE=640 \
		-DGGUF_MOE_EXPERTS=512 -DGGUF_MOE_TOP_K=10
# The same test at Flint's routing: 256 experts, 6 per token by a sigmoid
# router, expert width 512, and a shared expert of two ungated slices.
$(TEST_GGUF_MOE_SIGMOID): dev/tests/engine/gguf_moe_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GGUF_MOE_SIGMOID): ENGINE_TEST_CXXFLAGS += -DGGUF_MOE_HIDDEN=256 -DGGUF_MOE_INTERMEDIATE=512 \
		-DGGUF_MOE_EXPERTS=256 -DGGUF_MOE_TOP_K=6 -DGGUF_MOE_ROUTER=SigmoidNormalized \
		-DGGUF_MOE_SHARED_SLICES=2 -DGGUF_MOE_SHARED_GATE=false
$(TEST_EXPERT_STORE): dev/tests/engine/expert_store_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_KV_BLOCK_CACHE): dev/tests/engine/kv_block_cache_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_KV_HISTORY_STREAM): dev/tests/engine/kv_history_stream_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_QWEN4_KERNELS): dev/tests/engine/qwen4_kernels_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GDN_METAL_TEST): dev/tests/engine/gdn_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_ATTENTION_PLAN): dev/tests/engine/paged_attention_plan_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_ATTENTION_VARIANTS): dev/tests/engine/attention_variants_metal_test.mm \
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
$(TEST_TARGET_SAMPLING_TEST): dev/tests/engine/target_sampling_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_VISION_ATTENTION_TEST): dev/tests/engine/vision_attention_metal_test.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_Q8_METAL_TEST): dev/tests/engine/q8_paged_kv_metal_test.mm
$(TEST_Q8_STORAGE_TEST): dev/tests/engine/q8_page_storage_test.mm \
		$(ENGINE_LIBRARY)
$(TEST_METAL_BACKEND_TEST): dev/tests/engine/metal_backend_test.mm \
		$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_LIBRARY)
$(TEST_HANDOFF_TEST): dev/tests/engine/handoff_test.mm $(ENGINE_LIBRARY)
$(TEST_PRESSURE_SERVING): dev/tests/engine/pressure_serving_test.mm \
		dev/tests/engine/ProtocolPeer.cpp $(ENGINE_LIBRARY) $(LIB)
$(TEST_VISION_ENCODER_TEST): dev/tests/engine/vision_encoder_test.mm \
		$(ENGINE_LIBRARY)
$(TEST_MODEL_RUNTIME_ORACLE): dev/tests/engine/model_runtime_oracle_test.mm \
		dev/tools/LoadOptions.hpp \
		$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_INSTRUMENTED_ANE_OBJECT) \
		$(ENGINE_INSTRUMENTED_RUNTIME_OBJECT) $(ENGINE_LIBRARY) $(LIB) $(BUILD_ID_HEADER)
# The oracle remembers the split's calibration under the build, as the engine does.
$(TEST_MODEL_RUNTIME_ORACLE): ENGINE_TEST_CXXFLAGS += -include $(BUILD_ID_HEADER)
$(TEST_DECODE_PROFILE): dev/benchmarks/decode_profile.mm \
		dev/tools/LoadOptions.hpp \
		$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_LIBRARY) $(LIB)
$(TEST_TEACHER_LOGITS): dev/benchmarks/teacher_logits.mm \
		dev/tools/LoadOptions.hpp \
		$(ENGINE_INSTRUMENTED_RUNTIME_OBJECT) $(ENGINE_LIBRARY) $(LIB)
$(TEST_ATTENTION_SWEEP): dev/benchmarks/attention_sweep.mm \
		dev/benchmarks/DispatchReplay.hpp \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GGUF_PROJECTION_BENCHMARK): dev/benchmarks/gguf_projection_benchmark.mm \
		$(ENGINE_LIBRARY) $(LIB)
$(TEST_GGUF_MOE_BENCHMARK): dev/benchmarks/gguf_moe_benchmark.mm \
		dev/benchmarks/DispatchReplay.hpp \
		$(ENGINE_LIBRARY) $(LIB)

$(TEST_MODEL_CONFIGURATION) $(TEST_DRAFT_PREPARATION) $(TEST_GGUF_PROJECTION) $(TEST_ANE_FFN) \
		$(TEST_ANE_PROGRAM_FAULTS) $(TEST_GGUF_DEQUANT) $(TEST_GGUF_ROTATION) \
		$(TEST_GGUF_REFERENCE) $(TEST_GGUF_PLANNER) $(TEST_GGUF_PREPARATION) \
		$(TEST_BOOTSTRAP_TEST) $(TEST_RESOURCES_TEST) \
		$(TEST_KV_PAGE_TIER_TEST) $(TEST_QWEN_STATE_TEST) $(TEST_Q8_ATTENTION_TEST) \
		$(TEST_Q8_PREFILL_TEST) $(TEST_GGUF_MOE) $(TEST_GGUF_MOE_E512) $(TEST_GGUF_MOE_SIGMOID) \
		$(TEST_EXPERT_STORE) $(TEST_KV_BLOCK_CACHE) $(TEST_KV_HISTORY_STREAM) $(TEST_QWEN4_KERNELS) \
		$(TEST_GDN_METAL_TEST) $(TEST_ATTENTION_PLAN) $(TEST_LINEAR_PLAN) \
		$(TEST_DFLASH_BATCH_CONTROL_TEST) $(TEST_DRAFT_ATTENTION_TEST) \
		$(TEST_GDN_DECODE_TEST) $(TEST_DRAFT_SELECTOR_TEST) \
		$(TEST_TARGET_SAMPLING_TEST) $(TEST_VISION_ATTENTION_TEST) \
		$(TEST_Q8_METAL_TEST) $(TEST_Q8_STORAGE_TEST) $(TEST_ATTENTION_VARIANTS) \
		$(TEST_METAL_BACKEND_TEST) $(TEST_HANDOFF_TEST) $(TEST_PRESSURE_SERVING) $(TEST_VISION_ENCODER_TEST) \
		$(TEST_MODEL_RUNTIME_ORACLE) \
		$(TEST_DECODE_PROFILE) $(TEST_TEACHER_LOGITS) $(TEST_ATTENTION_SWEEP) \
		$(TEST_GGUF_PROJECTION_BENCHMARK) $(TEST_GGUF_MOE_BENCHMARK): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

# Tools built with the production flags.
$(WEIGHT_DIGESTS): dev/tools/weight_digests.mm \
		dev/tools/LoadOptions.hpp $(ENGINE_LIBRARY)

$(WEIGHT_DIGESTS): | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -Idev $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_BACKEND_BENCHMARK): dev/benchmarks/backend_benchmark.mm \
		dev/benchmarks/PrefillWork.hpp dev/benchmarks/TranscriptPrompts.hpp \
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
$(TEST_HOST_STEP_LIB): $(TEST_HOST_STEP_AIR)
# The production kernels and the test kernels of metal_backend_test.metal in
# one library, for the kernel tests that order a test copy between production
# dispatches. It relinks whenever the production library does.
$(TEST_PRODUCTION_LIB): $(PRODUCTION_AIRS) $(TEST_METAL_BACKEND_AIR) $(LIB)

$(TEST_GGUF_DEQUANT_LIB) $(TEST_Q8_LIB) $(TEST_Q8_ATTENTION_LIB) $(TEST_METAL_BACKEND_LIB) \
		$(TEST_PRODUCTION_LIB) $(TEST_HOST_STEP_LIB):
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

# test-engine-cpu runs every suite and fails at its end, naming each that
# failed, so that one run shows every failure instead of the first of 50.
# engine-sim, the slowest, runs last.
ENGINE_CPU_FAILED := $(ENGINE_TEST_BUILD)/test-engine-cpu.failed
ENGINE_CPU_STEP = @step() { echo "$$*"; "$$@" || echo "$$*" >> $(ENGINE_CPU_FAILED); }; step
test-engine-cpu: $(TEST_CPU_TARGETS) $(TEST_ATTENTION_SWEEP) $(TEST_BACKEND_BENCHMARK)
	@: > $(ENGINE_CPU_FAILED)
	$(ENGINE_CPU_STEP) $(TEST_SLOT_FILE)
	$(ENGINE_CPU_STEP) $(TEST_KV_SLOT_FORMAT)
	$(ENGINE_CPU_STEP) $(TEST_KV_PAGE_TIER_HOST)
	$(ENGINE_CPU_STEP) $(TEST_SPARSE_ROW_STORE)
	$(ENGINE_CPU_STEP) $(TEST_SSD_READERS)
	$(ENGINE_CPU_STEP) $(TEST_SLOT_RECYCLER)
	$(ENGINE_CPU_STEP) $(TEST_EXPERT_CHUNK_LAYOUT)
	$(ENGINE_CPU_STEP) $(TEST_VERIFY_BUDGET)
	$(ENGINE_CPU_STEP) $(TEST_CACHE_DIRECTORY)
	$(ENGINE_CPU_STEP) $(TEST_DISK_SPACE)
	$(ENGINE_CPU_STEP) $(TEST_LEGACY_WEIGHT_CACHE)
	$(ENGINE_CPU_STEP) $(BUILD_ID_PYTHON) dev/tests/engine/run_vision_preparation.py $(TEST_VISION_PREPARATION) $(WEIGHT_GOLDENS)
	$(ENGINE_CPU_STEP) $(TEST_SAFETENSORS_CHECKPOINT)
	$(ENGINE_CPU_STEP) $(TEST_MODEL_CONFIGURATION) $(MODEL_CONFIGS)
	$(ENGINE_CPU_STEP) $(TEST_FAMILY_VALUES) dev/tests/engine/family_values_golden.txt
	$(ENGINE_CPU_STEP) $(TEST_WEIGHT_SOURCE)
	$(ENGINE_CPU_STEP) $(TEST_GGUF_FILE)
	$(ENGINE_CPU_STEP) $(TEST_GGUF_REFERENCE) $(WEIGHT_GOLDENS) $(MLX_QUANTIZATION_FIXTURE)
	$(ENGINE_CPU_STEP) $(TEST_GGUF_PLANNER)
	$(ENGINE_CPU_STEP) $(TEST_LINEAR_PLAN) --cpu
	$(ENGINE_CPU_STEP) /bin/sh dev/tests/attention_sweep_cli.sh $(TEST_ATTENTION_SWEEP)
	$(ENGINE_CPU_STEP) /bin/sh dev/tests/backend_benchmark_cli.sh $(TEST_BACKEND_BENCHMARK) $(MODEL_CONFIGS)
	$(ENGINE_CPU_STEP) $(TEST_OPERATOR_WORKSPACE)
	$(ENGINE_CPU_STEP) $(TEST_ANE_FFN_CALIBRATION)
	$(ENGINE_CPU_STEP) $(TEST_ANE_FFN_STARTUP)
	$(ENGINE_CPU_STEP) $(TEST_EXECUTION_PLANS)
	$(ENGINE_CPU_STEP) $(TEST_KERNEL_SELECTION) dev/tests/engine/kernel_selection_golden.txt
	$(ENGINE_CPU_STEP) $(TEST_MODEL_EXECUTION_PLANS)
	$(ENGINE_CPU_STEP) $(TEST_COMMAND_LIMITS)
	$(ENGINE_CPU_STEP) $(TEST_MEMORY_TEST)
	$(ENGINE_CPU_STEP) $(TEST_KV_PAGE_CACHE_TEST)
	$(ENGINE_CPU_STEP) $(TEST_ENGINE_CACHE_TEST)
	$(ENGINE_CPU_STEP) $(TEST_CACHE_PROPERTY)
	$(ENGINE_CPU_STEP) $(TEST_PERSISTENT_CACHE)
	$(ENGINE_CPU_STEP) $(TEST_DRAFT_CONTEXT_PLAN_TEST)
	$(ENGINE_CPU_STEP) $(TEST_RAGGED_SCHEDULER_TEST)
	$(ENGINE_CPU_STEP) $(TEST_MODEL_BOUNDARY_TEST)
	$(ENGINE_CPU_STEP) $(TEST_WEIGHT_RESIDENCY)
	$(ENGINE_CPU_STEP) $(TEST_FAKE_MEMORY)
	$(ENGINE_CPU_STEP) $(TEST_TIMING_PROFILE)
	$(ENGINE_CPU_STEP) $(TEST_ENGINE_TEST)
	$(ENGINE_CPU_STEP) $(TEST_ELASTIC_KV_TEST)
	$(ENGINE_CPU_STEP) $(TEST_PROTOCOL_TEST) dev/tests/engine/protocol_golden.txt
	$(ENGINE_CPU_STEP) $(TEST_NATIVE_LOOP_TEST)
	$(ENGINE_CPU_STEP) $(TEST_FD_TRANSPORT_TEST)
	$(ENGINE_CPU_STEP) $(TEST_BOOTSTRAP_TEST)
	$(ENGINE_CPU_STEP) $(TEST_NATIVE_ARGUMENTS_TEST) dev/tests/engine/native_command_golden.txt $(MODEL_CONFIGS)
	$(ENGINE_CPU_STEP) $(TEST_METRICS_TEST)
	$(ENGINE_CPU_STEP) $(TEST_MEMORY_AUDIT_TEST)
	$(ENGINE_CPU_STEP) $(TEST_MEMORY_GOVERNOR_TEST)
	$(ENGINE_CPU_STEP) $(TEST_MEMORY_LEDGER)
	$(ENGINE_CPU_STEP) $(TEST_STATUS_TEST) dev/tests/engine/status_golden.json dev/tests/engine/status_variants_golden.jsonl
	$(ENGINE_CPU_STEP) $(TEST_JSON_WRITER)
	$(ENGINE_CPU_STEP) $(TEST_Q8_CPU_TEST)
	$(ENGINE_CPU_STEP) $(TEST_ENGINE_SIM)
	@if [ -s $(ENGINE_CPU_FAILED) ]; then echo "test-engine-cpu failed:"; sed "s/^/  /" $(ENGINE_CPU_FAILED); exit 1; fi

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
	$(METAL_TEST_ENV) $(TEST_GGUF_MOE_E512) $(LIB)
	$(METAL_TEST_ENV) $(TEST_GGUF_MOE_SIGMOID) $(LIB)
	$(METAL_TEST_ENV) $(TEST_EXPERT_STORE) $(LIB)
	$(METAL_TEST_ENV) $(TEST_KV_BLOCK_CACHE) $(LIB)
	$(METAL_TEST_ENV) $(TEST_KV_HISTORY_STREAM) $(LIB)
	$(METAL_TEST_ENV) $(TEST_QWEN4_KERNELS) $(LIB)
	$(TEST_QWEN4_KERNELS) $(LIB) bitwise
	$(METAL_TEST_ENV) $(TEST_ATTENTION_PLAN) $(LIB)
	$(METAL_TEST_ENV) $(TEST_ATTENTION_VARIANTS) $(LIB)
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
	$(METAL_TEST_ENV) $(TEST_TARGET_SAMPLING_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_VISION_METAL_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_VISION_ATTENTION_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_METAL_BACKEND_TEST) $(TEST_METAL_BACKEND_LIB)
	$(METAL_TEST_ENV) $(TEST_HANDOFF_TEST) $(TEST_METAL_BACKEND_LIB)
	$(METAL_TEST_ENV) $(TEST_PRESSURE_SERVING) $(LIB)

.PHONY: test-real
# The vision fixture is named after the installed model's family, which its
# model.json records. Nothing is printed when the installation serves no
# vision (vision_format none: --language-only, or a GGUF without an mmproj).
# ORACLE_ARGS: options every run of the runtime oracle takes besides its own,
# as the oracle takes them (model_runtime_oracle_test.mm), an option given
# twice taking its later value. A block-sparse target's spill check runs in
# each: by default, with two staging buffers, and at 32K tokens with the
# cache's default window, so a --spill-tokens here replaces the size of all
# three. Each stops before its prompts, naming what fits, where the memory
# plan's limit leaves less than it holds beside the weights: up to 4.3 GB
# for Qwen3.8-Flash-Next's default checks, and 5.9 GB for its 32K check (on
# the 24 GB M6, 11-02: the spill runtime's arenas 2.27 GB, four FP32 lane
# states 1.11 GB, 17 Full extents 1.84 GB, the hot block cache's floor for
# two lanes 0.61 GB, Index extents and staging). On a 24 GB Mac all three
# runs take ORACLE_ARGS="--expert-memory 6656M --ple-memory 237M", or
# "--expert-memory 5632M --ple-memory 237M" where the installation serves
# vision, whose default checks alone fit only up to about 7700M of experts
# (docs/validate.md). Without --ple-memory the oracle loads
# Qwen3.8-Flash-Next's PLE table whole (the weights then need 37555 MiB, M6
# 11-03) where splash serve streams it; 237M is the row cache splash serve
# gives it on the M6.
ORACLE_ARGS ?=
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
	$(TEST_MODEL_RUNTIME_ORACLE) $(LIB) "$(MODEL_ROOT)" --ane-ffn-share 0 $(ORACLE_ARGS)
	$(TEST_MODEL_RUNTIME_ORACLE) $(LIB) "$(MODEL_ROOT)" --ane-ffn-share 0 --state-format bf16 \
		--spill-buffers 2 $(ORACLE_ARGS)
	$(TEST_MODEL_RUNTIME_ORACLE) $(LIB) "$(MODEL_ROOT)" --spill-tokens 32768 --spill-window 64 $(ORACLE_ARGS)

.PHONY: benchmark-backend \
	benchmark-decode-profile benchmark-attention-sweep \
	benchmark-gguf-projection benchmark-gguf-moe teacher-logits
# decode-profile replays the installed model's prefill and decode commands as
# separate dispatches; DECODE_PROFILE_ARGS passes --prompt-tokens/--cycles,
# or --images HxW[,HxW...] for the vision encoder on synthetic images.
benchmark-decode-profile: preflight $(TARGET) $(TEST_DECODE_PROFILE) $(LIB)
	$(TEST_DECODE_PROFILE) $(LIB) "$(MODEL_ROOT)" $(DECODE_PROFILE_ARGS)

# The installed model's teacher-forced logits over windows of a token-id file,
# for parity checks against another engine; TEACHER_LOGITS_ARGS passes
# TOKENS OUTPUT --window END:ROWS [...] and the runtime options
# (dev/benchmarks/teacher_logits.mm).
teacher-logits: preflight $(TARGET) $(TEST_TEACHER_LOGITS) $(LIB)
	$(TEST_TEACHER_LOGITS) $(LIB) "$(MODEL_ROOT)" $(TEACHER_LOGITS_ARGS)

# Attention kernels alone on one layer of synthetic Q8 history across cache
# lengths; ATTENTION_SWEEP_ARGS passes --histories/--shapes/--lanes/--repeat.
benchmark-attention-sweep: $(TEST_ATTENTION_SWEEP) $(LIB)
	$(TEST_ATTENTION_SWEEP) $(LIB) $(ATTENTION_SWEEP_ARGS)

# One GGUF projection on both decode tiles at every lane count and K split
# (the split tiers of runtime/ops/LinearGguf.cpp), or with prefill=R[,R...]
# on the 128-row prefill tile at each chunk of R rows; the float formats
# float-bf16 and float-f32 time the float projection on both float tiles.
# GGUF_PROJECTION_ARGS passes <fmt[+fmt+fmt]> <N[+N+N]> <K>
# [none|residual|gateup] [rounds] [prefill=R[,R...]].
GGUF_PROJECTION_ARGS ?= q4k 5120 8192
benchmark-gguf-projection: $(TEST_GGUF_PROJECTION_BENCHMARK) $(LIB)
	$(TEST_GGUF_PROJECTION_BENCHMARK) $(LIB) $(GGUF_PROJECTION_ARGS)

# One MoE layer in GGUF formats, on the device's plans and the other GGUF
# tile (ops/MoE.cpp); GGUF_MOE_ARGS passes [35b|flash-next|flint] [rounds]
# [gate/up format] [down format] [shared format] (35b, q4k q5k q80 by
# default).
benchmark-gguf-moe: $(TEST_GGUF_MOE_BENCHMARK) $(LIB)
	$(TEST_GGUF_MOE_BENCHMARK) $(LIB) $(GGUF_MOE_ARGS)

benchmark-backend: preflight $(TARGET) $(TEST_BACKEND_BENCHMARK) $(LIB)
	$(TEST_BACKEND_BENCHMARK) $(LIB) "$(MODEL_ROOT)"

# CPU tests that also run under the sanitizers: each is built three times
# from the same sources, once as the CPU tests above are and once under each
# sanitizer, or twice for a test that starts no thread, which runs under ASan
# alone.
$(TEST_SLOT_FILE) $(TEST_SLOT_FILE_ASAN) $(TEST_SLOT_FILE_TSAN): \
		runtime/model/SlotFile.cpp runtime/model/SsdReaders.cpp dev/tests/engine/slot_file_test.cpp
$(TEST_KV_SLOT_FORMAT) $(TEST_KV_SLOT_FORMAT_ASAN): \
		runtime/ops/KvSlotFormat.cpp dev/tests/engine/kv_slot_format_test.cpp
$(TEST_KV_PAGE_TIER_HOST) $(TEST_KV_PAGE_TIER_HOST_ASAN) $(TEST_KV_PAGE_TIER_HOST_TSAN): \
		runtime/ops/KvSlotFormat.cpp runtime/model/SlotFile.cpp runtime/model/SsdReaders.cpp \
		runtime/model/KvCopyRead.cpp dev/tests/engine/kv_page_tier_host_test.cpp
$(TEST_DISK_SPACE) $(TEST_DISK_SPACE_ASAN) $(TEST_DISK_SPACE_TSAN): \
		runtime/engine/DiskSpace.cpp runtime/engine/CacheDirectory.cpp \
		runtime/model/SlotFile.cpp dev/tests/engine/disk_space_test.cpp
$(TEST_CACHE_DIRECTORY) $(TEST_CACHE_DIRECTORY_ASAN) $(TEST_CACHE_DIRECTORY_TSAN): \
		runtime/engine/CacheDirectory.cpp dev/tests/engine/cache_directory_test.cpp
$(TEST_SPARSE_ROW_STORE) $(TEST_SPARSE_ROW_STORE_ASAN) $(TEST_SPARSE_ROW_STORE_TSAN): \
		runtime/model/SparseRowStore.cpp runtime/model/SsdReaders.cpp runtime/model/WeightSource.cpp \
		dev/tests/engine/sparse_row_store_test.cpp
$(TEST_SSD_READERS) $(TEST_SSD_READERS_ASAN) $(TEST_SSD_READERS_TSAN): \
		runtime/model/SsdReaders.cpp dev/tests/engine/ssd_readers_test.cpp
$(TEST_ELASTIC_KV_TEST) $(TEST_ELASTIC_KV_ASAN) $(TEST_ELASTIC_KV_TSAN): \
		runtime/engine/KvPool.cpp dev/tests/engine/elastic_kv_pool_test.cpp
$(TEST_SLOT_RECYCLER) $(TEST_SLOT_RECYCLER_ASAN) $(TEST_SLOT_RECYCLER_TSAN): \
		runtime/model/SlotRecycler.cpp dev/tests/engine/slot_recycler_test.cpp
$(TEST_PERSISTENT_CACHE) $(TEST_PERSISTENT_CACHE_ASAN) $(TEST_PERSISTENT_CACHE_TSAN): \
		$(CACHE_SOURCES) runtime/engine/MemoryLedger.cpp runtime/model/SlotFile.cpp \
		dev/tests/engine/persistent_cache_test.cpp
$(TEST_ENGINE_CACHE_TEST) $(TEST_ENGINE_CACHE_ASAN) $(TEST_ENGINE_CACHE_TSAN): \
		$(CACHE_SOURCES) runtime/engine/MemoryLedger.cpp runtime/model/SlotFile.cpp \
		dev/tests/engine/engine_cache_test.cpp
$(TEST_CACHE_PROPERTY) $(TEST_CACHE_PROPERTY_ASAN) $(TEST_CACHE_PROPERTY_TSAN): \
		$(CACHE_SOURCES) runtime/engine/MemoryLedger.cpp runtime/model/SlotFile.cpp \
		dev/tests/engine/cache_property_test.cpp
$(TEST_PROTOCOL_TEST) $(TEST_PROTOCOL_ASAN) $(TEST_PROTOCOL_TSAN): \
		runtime/engine/Protocol.cpp dev/tests/engine/ProtocolPeer.cpp \
		dev/tests/engine/protocol_test.cpp
$(TEST_ENGINE_TEST) $(TEST_ENGINE_ASAN) $(TEST_ENGINE_TSAN): \
		$(BACKEND_CONTROL_SOURCES) \
		runtime/engine/MemoryGovernor.cpp \
		runtime/engine/MemoryLedger.cpp \
		dev/benchmarks/PrefillWork.hpp \
		dev/tests/engine/engine_requests_test.cpp dev/tests/engine/engine_checkpoints_test.cpp \
		dev/tests/engine/engine_commands_test.cpp dev/tests/engine/engine_memory_test.cpp \
		dev/tests/engine/engine_spill_test.cpp dev/tests/engine/engine_disk_test.cpp \
		dev/tests/engine/engine_memory_rules_test.cpp \
		dev/tests/engine/TimingProfile.cpp dev/tests/engine/engine_test_main.cpp
$(TEST_RAGGED_SCHEDULER_TEST) $(TEST_RAGGED_SCHEDULER_ASAN) $(TEST_RAGGED_SCHEDULER_TSAN): \
		runtime/engine/Scheduler.cpp \
		dev/tests/engine/TimingProfile.cpp dev/tests/engine/ragged_scheduler_test.cpp
$(TEST_MODEL_BOUNDARY_TEST) $(TEST_MODEL_BOUNDARY_ASAN) $(TEST_MODEL_BOUNDARY_TSAN): \
		dev/tests/engine/model_boundary_test.cpp
$(TEST_WEIGHT_RESIDENCY): runtime/engine/WeightResidency.cpp dev/tests/engine/weight_residency_test.cpp
$(TEST_FAKE_MEMORY) $(TEST_FAKE_MEMORY_ASAN) $(TEST_FAKE_MEMORY_TSAN): \
		$(NATIVE_RUNTIME_SOURCES) \
		runtime/engine/MemoryGovernor.cpp runtime/engine/MemoryControl.cpp runtime/engine/MemoryLedger.cpp \
		dev/tests/engine/ProtocolPeer.cpp dev/tests/engine/fake_memory_test.cpp
$(TEST_TIMING_PROFILE) $(TEST_TIMING_PROFILE_ASAN) $(TEST_TIMING_PROFILE_TSAN): \
		$(BACKEND_CONTROL_SOURCES) \
		runtime/engine/MemoryGovernor.cpp runtime/engine/MemoryLedger.cpp \
		dev/tests/engine/TimingProfile.cpp dev/tests/engine/timing_profile_test.cpp
$(TEST_FD_TRANSPORT_TEST) $(TEST_FD_TRANSPORT_ASAN) $(TEST_FD_TRANSPORT_TSAN): \
		$(NATIVE_RUNTIME_SOURCES) \
		runtime/engine/MemoryGovernor.cpp runtime/engine/MemoryLedger.cpp \
		runtime/engine/FdTransport.cpp dev/tests/engine/metal_stand_in.cpp \
		dev/tests/engine/ProtocolPeer.cpp \
		dev/tests/engine/native_fd_transport_test.cpp
$(TEST_NATIVE_LOOP_TEST) $(TEST_NATIVE_LOOP_ASAN) $(TEST_NATIVE_LOOP_TSAN): \
		$(NATIVE_RUNTIME_SOURCES) \
		runtime/engine/MemoryGovernor.cpp runtime/engine/MemoryControl.cpp \
		runtime/engine/MemoryLedger.cpp \
		dev/tests/engine/ProtocolPeer.cpp \
		dev/tests/engine/native_loop_wire_test.cpp dev/tests/engine/native_loop_weights_test.cpp \
		dev/tests/engine/native_loop_memory_test.cpp dev/tests/engine/native_loop_test_main.cpp
$(TEST_ENGINE_SIM) $(TEST_ENGINE_SIM_ASAN) $(TEST_ENGINE_SIM_TSAN): \
		$(NATIVE_RUNTIME_SOURCES) \
		runtime/engine/MemoryGovernor.cpp runtime/engine/MemoryControl.cpp \
		runtime/engine/MemoryLedger.cpp \
		dev/tests/engine/ProtocolPeer.cpp dev/tests/engine/TimingProfile.cpp \
		dev/tests/engine/engine_sim_test.cpp
$(TEST_ANE_FFN_CALIBRATION) $(TEST_ANE_FFN_CALIBRATION_ASAN) $(TEST_ANE_FFN_CALIBRATION_TSAN): \
		runtime/ops/AneFfnCalibration.cpp dev/tests/engine/ane_ffn_calibration_test.cpp
$(TEST_MEMORY_TEST) $(TEST_MEMORY_PLAN_ASAN): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/StreamedMemoryPlan.cpp \
		dev/tests/engine/engine_memory_plan_test.cpp
$(TEST_MEMORY_AUDIT_TEST) $(TEST_MEMORY_AUDIT_ASAN): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryAudit.cpp \
		dev/tests/engine/memory_audit_test.cpp
$(TEST_STATUS_TEST) $(TEST_STATUS_ASAN): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryAudit.cpp \
		runtime/engine/Status.cpp \
		dev/tests/engine/runtime_status_test.cpp
$(TEST_JSON_WRITER) $(TEST_JSON_WRITER_ASAN): dev/tests/engine/json_writer_test.cpp

$(filter %-asan-ubsan,$(SANITIZER_CONFIG_TARGETS)): SANITIZERS := address,undefined
$(filter %-tsan,$(SANITIZER_CONFIG_TARGETS)): SANITIZERS := thread
$(SANITIZER_CONFIG_TARGETS): | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=$(SANITIZERS) \
		$(TEST_INPUTS) -o $@

.PHONY: test-sanitizers
ASAN_TEST_ENV := ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
TSAN_TEST_ENV := TSAN_OPTIONS=halt_on_error=1
# engine-sim under the sanitizers runs every scenario but those too long for
# them: testRegrowthAfterABurstTakesOnlyCacheThatFreesNow runs past 5 minutes
# under ThreadSanitizer, where the others take at most 1.5. test-engine-cpu
# runs every scenario.
ENGINE_SIM_SANITIZER_SKIPPED := sim/testRegrowthAfterABurstTakesOnlyCacheThatFreesNow
# The names of sim binary $(1)'s scenarios but the skipped ones, for --filter.
engine_sim_sanitizer_scenarios = "$$($(1) --list | grep -v -x -F $(foreach name,$(ENGINE_SIM_SANITIZER_SKIPPED),-e $(name)) | paste -s -d , -)"
test-sanitizers: $(SANITIZER_CONFIG_TARGETS)
	$(ASAN_TEST_ENV) $(TEST_SLOT_FILE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_SLOT_FILE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_KV_SLOT_FORMAT_ASAN)
	$(ASAN_TEST_ENV) $(TEST_KV_PAGE_TIER_HOST_ASAN)
	$(TSAN_TEST_ENV) $(TEST_KV_PAGE_TIER_HOST_TSAN)
	$(ASAN_TEST_ENV) $(TEST_DISK_SPACE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_DISK_SPACE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_CACHE_DIRECTORY_ASAN)
	$(TSAN_TEST_ENV) $(TEST_CACHE_DIRECTORY_TSAN)
	$(ASAN_TEST_ENV) $(TEST_PERSISTENT_CACHE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_PERSISTENT_CACHE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_ENGINE_CACHE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_ENGINE_CACHE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_CACHE_PROPERTY_ASAN)
	$(TSAN_TEST_ENV) $(TEST_CACHE_PROPERTY_TSAN)
	$(ASAN_TEST_ENV) $(TEST_PROTOCOL_ASAN) dev/tests/engine/protocol_golden.txt
	$(TSAN_TEST_ENV) $(TEST_PROTOCOL_TSAN) dev/tests/engine/protocol_golden.txt
	$(ASAN_TEST_ENV) $(TEST_ENGINE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_ENGINE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_RAGGED_SCHEDULER_ASAN)
	$(TSAN_TEST_ENV) $(TEST_RAGGED_SCHEDULER_TSAN)
	$(ASAN_TEST_ENV) $(TEST_MODEL_BOUNDARY_ASAN)
	$(TSAN_TEST_ENV) $(TEST_MODEL_BOUNDARY_TSAN)
	$(ASAN_TEST_ENV) $(TEST_FAKE_MEMORY_ASAN)
	$(TSAN_TEST_ENV) $(TEST_FAKE_MEMORY_TSAN)
	$(ASAN_TEST_ENV) $(TEST_TIMING_PROFILE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_TIMING_PROFILE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_FD_TRANSPORT_ASAN)
	$(TSAN_TEST_ENV) $(TEST_FD_TRANSPORT_TSAN)
	$(ASAN_TEST_ENV) $(TEST_NATIVE_LOOP_ASAN)
	$(TSAN_TEST_ENV) $(TEST_NATIVE_LOOP_TSAN)
	$(ASAN_TEST_ENV) $(TEST_ENGINE_SIM_ASAN) --filter $(call engine_sim_sanitizer_scenarios,$(TEST_ENGINE_SIM_ASAN))
	$(TSAN_TEST_ENV) $(TEST_ENGINE_SIM_TSAN) --filter $(call engine_sim_sanitizer_scenarios,$(TEST_ENGINE_SIM_TSAN))
	$(ASAN_TEST_ENV) $(TEST_ANE_FFN_CALIBRATION_ASAN)
	$(TSAN_TEST_ENV) $(TEST_ANE_FFN_CALIBRATION_TSAN)
	$(ASAN_TEST_ENV) $(TEST_SPARSE_ROW_STORE_ASAN)
	$(TSAN_TEST_ENV) $(TEST_SPARSE_ROW_STORE_TSAN)
	$(ASAN_TEST_ENV) $(TEST_SSD_READERS_ASAN)
	$(TSAN_TEST_ENV) $(TEST_SSD_READERS_TSAN)
	$(ASAN_TEST_ENV) $(TEST_ELASTIC_KV_ASAN)
	$(TSAN_TEST_ENV) $(TEST_ELASTIC_KV_TSAN)
	$(ASAN_TEST_ENV) $(TEST_SLOT_RECYCLER_ASAN)
	$(TSAN_TEST_ENV) $(TEST_SLOT_RECYCLER_TSAN)
	$(ASAN_TEST_ENV) $(TEST_MEMORY_PLAN_ASAN)
	$(ASAN_TEST_ENV) $(TEST_MEMORY_AUDIT_ASAN)
	$(ASAN_TEST_ENV) $(TEST_STATUS_ASAN) dev/tests/engine/status_golden.json dev/tests/engine/status_variants_golden.jsonl
	$(ASAN_TEST_ENV) $(TEST_JSON_WRITER_ASAN)
	$(ASAN_TEST_ENV) $(TEST_MEMORY_LEDGER_ASAN)

# The Metal tests of the backend, whose host-step runner shares each
# command's steps with the thread that encodes it (its behaviour tests: its
# latency tests bound the runner's times on the wall clock, which
# ThreadSanitizer lengthens), and of the components whose host steps share
# their state with the engine thread and the SSD readers, the hot block
# cache, the history stream and the expert store, under ThreadSanitizer on a
# Mac with a GPU: the engine library and the tests built with it, and their
# metallibs, in a build of their own (the builds above link no framework),
# and what it cannot see inside Metal suppressed (metal_tsan_suppressions.txt).
# check-native-metal runs it.
METAL_TSAN_BUILD := $(BUILD)/metal-tsan
METAL_TSAN_CXXFLAGS := $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=thread
METAL_TSAN_TEST_ENV := $(TSAN_TEST_ENV):suppressions=dev/tests/engine/metal_tsan_suppressions.txt
.PHONY: test-sanitizers-metal
test-sanitizers-metal:
	$(MAKE) --no-print-directory -f "$(SPLASH_MAKEFILE)" BUILD=$(METAL_TSAN_BUILD) \
		ENGINE_CXXFLAGS='$(METAL_TSAN_CXXFLAGS)' ENGINE_TEST_CXXFLAGS='$(METAL_TSAN_CXXFLAGS)' \
		$(METAL_TSAN_BUILD)/splash.metallib $(METAL_TSAN_BUILD)/host-step.metallib \
		$(METAL_TSAN_BUILD)/engine-tests/kv-block-cache $(METAL_TSAN_BUILD)/engine-tests/kv-history-stream \
		$(METAL_TSAN_BUILD)/engine-tests/expert-store $(METAL_TSAN_BUILD)/engine-tests/metal-backend \
		$(METAL_TSAN_BUILD)/engine-tests/metal-backend.metallib $(METAL_TSAN_BUILD)/engine-tests/host-step.metallib
	$(METAL_TSAN_TEST_ENV) $(METAL_TSAN_BUILD)/engine-tests/kv-block-cache $(METAL_TSAN_BUILD)/splash.metallib
	$(METAL_TSAN_TEST_ENV) $(METAL_TSAN_BUILD)/engine-tests/kv-history-stream $(METAL_TSAN_BUILD)/splash.metallib
	$(METAL_TSAN_TEST_ENV) $(METAL_TSAN_BUILD)/engine-tests/expert-store $(METAL_TSAN_BUILD)/splash.metallib
	$(METAL_TSAN_TEST_ENV) $(METAL_TSAN_BUILD)/engine-tests/metal-backend \
		$(METAL_TSAN_BUILD)/engine-tests/metal-backend.metallib --filter behaviour/
