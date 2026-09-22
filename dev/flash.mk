# Isolated native Flash-Next executable; the existing model runtime stays intact.
FLASH_BUILD := $(BUILD)/flash
FLASH_WORKER := $(BUILD)/splash-flash
FLASH_FORWARD_ORACLE := $(BUILD)/flash-forward-oracle
FLASH_MTP_PROBE := $(BUILD)/flash-mtp-probe
FLASH_BATCH_VERIFY_ORACLE := $(BUILD)/flash-batch-verify-oracle
FLASH_BATCH_MTP_ORACLE := $(BUILD)/flash-batch-mtp-oracle
FLASH_BATCH_PREFILL_ORACLE := $(BUILD)/flash-batch-prefill-oracle
FLASH_OPERAND_EXPORT := $(BUILD)/flash-operand-export
FLASH_CPP_SOURCES := runtime/flash/FlashAffine.cpp runtime/flash/FlashHC.cpp \
	runtime/flash/FlashGDN.cpp runtime/flash/FlashQSA.cpp \
	runtime/flash/FlashPLE.cpp runtime/flash/FlashMoE.cpp \
	runtime/flash/FlashForward.cpp runtime/flash/FlashMTP.cpp \
	runtime/flash/FlashHCFused.cpp runtime/flash/FlashGDNFused.cpp \
	runtime/flash/FlashBatchForward.cpp runtime/flash/FlashAffineMPP.cpp \
	runtime/flash/FlashDenseCache.cpp runtime/flash/FlashMoEBuckets.cpp \
	runtime/flash/FlashMoEBlocked.cpp runtime/flash/FlashQSAFast.cpp \
	runtime/flash/FlashGDNSeparate.cpp runtime/flash/FlashMTPDepth.cpp \
	runtime/flash/FlashQSAMPP.cpp runtime/flash/FlashDenseSmallRows.cpp \
	runtime/flash/FlashQSABulk.cpp runtime/flash/FlashQSABulkPrepare.cpp \
	runtime/flash/FlashBatchVerify.cpp runtime/flash/FlashBatchVerifyGDN.cpp \
	runtime/flash/FlashBatchMTPForward.cpp runtime/flash/FlashGDNStaged.cpp \
	runtime/flash/FlashExpertDenseCache.cpp runtime/flash/FlashFloatDenseCache.cpp \
	runtime/flash/FlashBatchPrefill.cpp runtime/flash/FlashPLEFused.cpp \
	runtime/flash/FlashPLEPostFused.cpp runtime/flash/FlashGreedyGPU.cpp \
	runtime/flash/FlashInt8Head.cpp runtime/flash/FlashSharedExpertFused.cpp \
	runtime/flash/FlashGDNBatchILP.cpp runtime/flash/FlashGDNLazyRollback.cpp \
	runtime/flash/FlashBF16Q8Head.cpp runtime/flash/FlashIdleResidencyMaintenance.cpp \
	runtime/flash/FlashPLESSD.cpp runtime/flash/FlashPLESSDStore.cpp
FLASH_MM_SOURCES := runtime/flash/FlashDescriptor.mm runtime/flash/FlashWeights.mm \
	runtime/flash/FlashWorker.mm runtime/flash/FlashExpertCachePlan.mm \
	runtime/flash/FlashOperandStore.mm runtime/flash/FlashInt8ExpertStore.mm \
	runtime/flash/FlashInt8ExpertStoreMetadata.mm
FLASH_CPP_OBJECTS := $(patsubst runtime/flash/%.cpp,$(FLASH_BUILD)/%.o,$(FLASH_CPP_SOURCES))
FLASH_MM_OBJECTS := $(patsubst runtime/flash/%.mm,$(FLASH_BUILD)/%.o,$(FLASH_MM_SOURCES))
FLASH_OBJECTS := $(FLASH_CPP_OBJECTS) $(FLASH_MM_OBJECTS)
FLASH_CORE_OBJECTS := $(ENGINE_BUILD)/metal/MetalBackend.o \
	$(ENGINE_BUILD)/metal/DeviceCapabilities.o $(ENGINE_BUILD)/engine/Protocol.o \
	$(ENGINE_BUILD)/engine/MemoryGovernor.o
PRODUCTION_CONFIG_TARGETS += $(FLASH_OBJECTS) $(FLASH_WORKER) $(FLASH_FORWARD_ORACLE) $(FLASH_MTP_PROBE)
PRODUCTION_CONFIG_TARGETS += $(FLASH_BATCH_VERIFY_ORACLE) $(FLASH_BATCH_MTP_ORACLE)
PRODUCTION_CONFIG_TARGETS += $(FLASH_BATCH_PREFILL_ORACLE)
PRODUCTION_CONFIG_TARGETS += $(FLASH_OPERAND_EXPORT)

$(FLASH_BUILD):
	mkdir -p $@

$(FLASH_BUILD)/%.o: runtime/flash/%.cpp | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) -MMD -MP -c $< -o $@

$(FLASH_BUILD)/%.o: runtime/flash/%.mm | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -MMD -MP -c $< -o $@

$(FLASH_WORKER): $(FLASH_OBJECTS) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(FLASH_OBJECTS) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

$(FLASH_FORWARD_ORACLE): dev/benchmarks/flash_forward_oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

$(FLASH_MTP_PROBE): dev/benchmarks/flash_mtp_probe.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-next
$(FLASH_BATCH_VERIFY_ORACLE): dev/benchmarks/flash_batch_verify_oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

$(FLASH_BATCH_MTP_ORACLE): dev/benchmarks/flash_batch_mtp_oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

flash-next: $(FLASH_WORKER) $(LIB)

$(FLASH_OPERAND_EXPORT): dev/tools/flash_operand_export.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

$(FLASH_BATCH_PREFILL_ORACLE): dev/benchmarks/flash_batch_prefill_oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

-include $(FLASH_OBJECTS:.o=.d)
