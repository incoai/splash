# Holds the hash of the development requirements last installed.
DEV_STAMP = $(VENV)/.dev-requirements-installed
PYTHON_TEST_PATTERN ?= test_*.py

# Define native targets before aggregate prerequisites expand.
-include dev/native.mk

.PHONY: architecture-check check check-source \
	check-native-engine check-native-cpu check-native-metal check-native-build \
	check-python install-development _install-development test test-python \
	test-engine test-engine-cpu test-engine-metal

# Decided by the stamp's content, as the environment is: a rebuilt
# environment has no stamp, and under -j make may read the stamp's time
# before install-environment rebuilds it.
install-development: install-environment
	@$(ANNOUNCE_INSTALL_WAIT)
	@/usr/bin/lockf -k "$(INSTALL_LOCK)" $(MAKE) --no-print-directory \
		-f "$(SPLASH_MAKEFILE)" _install-development

_install-development:
	@set -eu; \
	hash=$$(shasum -a 256 dev/requirements.txt | awk '{print $$1}'); \
	if test "$$(cat "$(DEV_STAMP)" 2>/dev/null)" = "$$hash"; then \
		exit 0; \
	fi; \
	"$(PYTHON)" -m pip install -r dev/requirements.txt; \
	echo "$$hash" > "$(DEV_STAMP)"

# Build production and every native test and tool, and run the native tests
# that do not require model weights.
check-native-engine: check-native-cpu check-native-metal check-native-build

check-native-cpu: platform-check $(TARGET) test-engine-cpu

check-native-metal: platform-check $(TARGET) test-engine-metal

# Build every native test and tool TEST_CONFIG_TARGETS names and run none:
# hosted CI has no GPU the Metal tests support and no model for the
# real-weight tools, and builds them so that none breaks unnoticed.
check-native-build: platform-check $(TARGET) $(TEST_CONFIG_TARGETS)

# Every Python test, each in the directory of what it tests (dev/tests/server,
# install, gui or tools). Discovering dev/tests keeps this gate complete as tests
# are added; discovered from the checkout root (-t .), a module has the package
# name it is imported by, so none loads twice. The installer tests check
# configurations with the engine.
test-python: install-development $(TARGET)
	$(PYTHON) -m unittest discover -s dev/tests -t . -p '$(PYTHON_TEST_PATTERN)'

check-python: install-development test-python
	$(PYTHON) -m compileall -q server install/*.py gui \
		dev/tools dev/tests dev/benchmarks
	$(PYTHON) -m ruff check --config dev/ruff.toml server gui \
		install/*.py dev/tools dev/tests dev/benchmarks
	$(PYTHON) -m ruff format --check --config dev/ruff.toml server gui \
		install/*.py dev/tools dev/tests dev/benchmarks

# It parses the server's sources, which need the environment's Python
# (3.12-3.14); BUILD_ID_PYTHON, the build tools' python3, may be Xcode's 3.9.
architecture-check: install-environment
	$(PYTHON) dev/tools/check_architecture.py

check: check-source check-native-engine check-python

check-source: architecture-check
	@git diff --check
	@empty_tree=$$(git hash-object -t tree /dev/null); \
		git diff --check $$empty_tree HEAD

test: test-engine test-python
