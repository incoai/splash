.PHONY: build test

PYTHON ?= python3

build:
	xcodebuild -project Splash.xcodeproj -scheme Splash -destination 'platform=macOS' -derivedDataPath .build/DerivedData build CODE_SIGNING_ALLOWED=NO
	codesign --force --sign - --identifier org.splashml.desktop .build/DerivedData/Build/Products/Debug/Splash.app

test:
	xcodebuild -project Splash.xcodeproj -scheme Splash -destination 'platform=macOS' -derivedDataPath .build/DerivedData test CODE_SIGNING_ALLOWED=NO
	PYTHONPATH=.. $(PYTHON) -m unittest discover -s tests -p 'test_*.py'
