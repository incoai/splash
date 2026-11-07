import AppKit
import Foundation
import Testing

@testable import Splash

/// What the tests show: the source checkout's runtime, and a conversation that holds every kind
/// of reply and message.
enum Showcase {
    /// The source checkout these tests run from, whose runtime files the page's scripts come from.
    static let runtime = Runtime(root: URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../.."))

    /// Fails a test that draws Markdown without the checkout's Markdown scripts
    /// (`Runtime.markdownScripts`), where the page would show the replies as plain text.
    static func requireScripts() throws {
        for script in runtime.markdownScripts {
            try #require(FileManager.default.fileExists(atPath: script.path(percentEncoded: false)), "\(script.path(percentEncoded: false)) is missing")
        }
    }

    static let conversation = Conversation(title: "Getting started", messages: [
        ChatMessage(role: .user, text: "How do I get started with Splash? Show me everything a reply can hold."),
        ChatMessage(role: .assistant, text: reply,
                    reasoning: "The user wants an overview.\nI should cover installing, picking a model and chatting.\nA table compares the sizes.",
                    model: "mlx-community/Qwen3-30B-A3B-4bit", tokensPerSecond: 38.4, thinkingTime: 3.2, reasoningTokens: 120),
        ChatMessage(role: .user, text: "What’s in this picture?", images: [picture]),
        ChatMessage(role: .assistant, text: "A gradient from teal to blue, square, with nothing else in it.",
                    model: "mlx-community/Qwen3-30B-A3B-4bit", tokensPerSecond: 41,
                    error: "The reply reached the end of the model’s context."),
        ChatMessage(role: .user, text: "Thanks!\nOne more line, to see a message of two."),
        ChatMessage(role: .assistant, text: "You’re welcome. Ask again whenever you like.",
                    model: "mlx-community/Qwen3-30B-A3B-4bit", tokensPerSecond: 39.6, stopped: true),
    ])

    /// A reply with every kind of Markdown a reply holds.
    static let reply = """
        # Getting started with Splash

        Splash runs models **privately** on this Mac. Replies can use *emphasis*, `inline code`, \
        ~~strikethrough~~, and [links](https://example.com); an image shows as a link: ![diagram](https://example.com/d.png).

        ## Install

        1. Download the app and open it.
        2. Pick a model:
           - **Small** models start fast
           - **Large** models answer better
             - but need more memory
        3. Start chatting.

        ```bash
        curl -fsSL https://example.com/install.sh | sh
        splash serve --model mlx-community/Qwen3-30B-A3B-4bit --port 8000 --max-context 32K --kv-format bf16
        ```

        ### Comparison

        | Model | Memory | Speed | Notes |
        |:------|-------:|------:|-------|
        | Small | 16 GB | 95 tok/s | Good for quick questions |
        | Large | 64 GB | 38 tok/s | Better at code and long documents<br>Slower to load |
        | Largest | 128 GB | 21 tok/s | The best answers of the three, for the longest and hardest work, if the Mac has the memory for it |

        > Models never leave this Mac. Nothing you type is sent anywhere.
        >
        > — The Splash team

        ---

        ```
        plain code without a language
        ```

        - Done
        """

    /// A picture the user attached: a square gradient, as PNG.
    static let picture: Data = {
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 240, pixelsHigh: 240, bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        NSGradient(starting: .systemTeal, ending: .systemBlue)!.draw(in: NSRect(x: 0, y: 0, width: 240, height: 240), angle: 45)
        NSGraphicsContext.restoreGraphicsState()
        return bitmap.representation(using: .png, properties: [:])!
    }()
}
