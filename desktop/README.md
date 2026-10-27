# Splash desktop

A native macOS chat window for the Splash server. No sidebar, accounts, web
wrapper, or JavaScript dependencies. Built with SwiftUI and Foundation.

## Run

With Xcode or the Swift 6 command line tools installed, build and open the app
from the repository root:

```sh
make desktop-open
```

This creates `build/Splash.app`. You can open it in Finder or copy it to
Applications. The build signs the app locally; distribution signing and
notarization are not configured.

Start Splash separately with your model, for example:

```sh
./splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

The app connects to `http://127.0.0.1:8000` and discovers the served model. It
checks the connection every ten seconds while idle. Use Connection settings
(`⌘,`) for another address or an optional API key. The desktop client requires
macOS 14 or later; the inference engine retains its own hardware and OS
requirements. The app does not download models or start or stop the server.

## Chat

- Return sends; Shift Return inserts a new line.
- Stop or Escape cancels the response and keeps any partial text.
- Thinking uses the model's default reasoning; turn it off for a direct answer.
- Replies support inline Markdown, fenced code, and copying text.
- Clear chat (`⌘N`) removes the current conversation after confirmation.

The current conversation is restored on the next launch. It is saved as
plain text JSON at `~/Library/Application Support/Splash Desktop/conversation.json`
with access limited to your macOS user. Server address and thinking preference
use macOS defaults. API keys remain in memory and must be reentered after
quitting. Messages go to the server configured in Connection settings.

## Develop and test

```sh
swift run --package-path desktop SplashDesktop
make test-desktop
```

Tests exercise the actual HTTP client with a controlled URL protocol and cover
model discovery, authentication, request bodies, split UTF-8 and SSE frames,
thinking, cancellation, usage metrics, server errors, and incomplete responses.
No model weights or running inference engine are needed.
