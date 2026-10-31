import AppKit
import SwiftUI
import WebKit

/// The open conversation, shown as a web page in the system's WebKit: one page, its text selectable
/// across messages and scrolled by WebKit off the main thread. The page (Transcript/transcript.html,
/// with the runtime's markdown-it and highlight.js) loads once; after that only data goes to it, as
/// JSON: the whole conversation when it changes, and while a reply streams, that reply alone. The
/// page outlives the Chat it shows in, so that showing the Chat again, or another conversation,
/// never loads it again.
@MainActor
final class Transcript: NSObject {
    /// What Ask Again asks of the app.
    var askAgain: () -> Void = {}

    /// The page and its scripts.
    private let page: URL?
    private let scripts: [URL]
    private(set) var webView: WKWebView?
    /// Whether the page has loaded, and takes data.
    private(set) var loaded = false
    /// When the page's process ended lately, which bounds how often it loads again.
    private var terminations: [ContinuousClock.Instant] = []
    /// What the page shows, or will once it has loaded.
    private var shown: TranscriptPage?

    init(runtime: Runtime) {
        page = Bundle.resources.url(forResource: "transcript", withExtension: "html", subdirectory: "Transcript")
        scripts = runtime.markdownScripts
    }

    /// Loads the page, once: the first time the Chat shows, or before, so that it shows at once.
    func load() {
        guard webView == nil, let page else { return }
        let configuration = WKWebViewConfiguration()
        // Nothing the page does is kept between launches.
        configuration.websiteDataStore = .nonPersistent()
        for script in scripts {
            guard let source = try? String(contentsOf: script, encoding: .utf8) else { continue }
            configuration.userContentController.addUserScript(
                WKUserScript(source: source, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }
        configuration.userContentController.add(Messages(self), name: "transcript")
        let view = TranscriptWebView(frame: NSRect(x: 0, y: 0, width: 800, height: 600), configuration: configuration)
        // The window's background shows through, from the first frame and in either appearance.
        view.setValue(false, forKey: "drawsBackground")
        view.underPageBackgroundColor = .clear
        // A drop would load what was dropped in place of the page.
        view.unregisterDraggedTypes()
        view.navigationDelegate = self
        view.loadFileURL(page, allowingReadAccessTo: page.deletingLastPathComponent())
        webView = view
    }

    /// Shows `page`: the reply that streams alone when only it changed, the whole conversation
    /// otherwise; nothing when the page shows it already.
    func show(_ page: TranscriptPage) {
        let change = Self.change(from: shown, to: page)
        shown = page
        guard loaded else { return }
        switch change {
        case .none: break
        case .message(let message): call("transcript.update(JSON.parse(json))", with: message)
        case .page(let page): call("transcript.show(JSON.parse(json))", with: page)
        }
    }

    enum Change: Equatable {
        case none
        /// The last message alone, as while it streams.
        case message(TranscriptMessage)
        case page(TranscriptPage)
    }

    /// What the page needs to be sent to go from `shown` to `page`.
    static func change(from shown: TranscriptPage?, to page: TranscriptPage) -> Change {
        guard let shown, shown.id == page.id, shown.askAgain == page.askAgain,
              shown.messages.count == page.messages.count, let last = page.messages.last,
              shown.messages.last?.id == last.id, shown.messages.dropLast() == page.messages.dropLast()
        else { return .page(page) }
        return shown.messages.last == last ? .none : .message(last)
    }

    private func call(_ body: String, with value: some Encodable) {
        guard let webView, let json = try? Self.encoder.encode(value) else { return }
        webView.callAsyncJavaScript(body, arguments: ["json": String(decoding: json, as: UTF8.self)],
                                    in: nil, in: .page, completionHandler: nil)
    }

    /// How data goes to the page: dates as JavaScript has them, in milliseconds.
    static let encoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .millisecondsSince1970
        return encoder
    }()

    /// A message from the page: a copy, or Ask Again.
    fileprivate func receive(_ body: Any) {
        guard let message = body as? [String: Any] else { return }
        if let text = message["copy"] as? String {
            NSPasteboard.put(text)
        } else if message["askAgain"] != nil {
            askAgain()
        }
    }

    /// The symbols the page draws, by name.
    private static let glyphs = [
        "chevron": Glyph("chevron.right", size: 10, weight: .semibold),
        "copy": Glyph("doc.on.doc", size: 12),
        "check": Glyph("checkmark", size: 12),
        "again": Glyph("arrow.clockwise", size: 12),
        "warning": Glyph("exclamationmark.triangle.fill", size: 12, multicolor: true),
        "down": Glyph("arrow.down", size: 13, weight: .semibold),
    ]
}

extension Transcript: WKNavigationDelegate {
    /// The page loads; a link opens in the default browser, web and mail links only.
    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
        guard let url = action.request.url, url != page else { return .allow }
        if action.navigationType == .linkActivated, ["http", "https", "mailto"].contains(url.scheme?.lowercased() ?? "") {
            NSWorkspace.shared.open(url)
        }
        return .cancel
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        loaded = true
        call("transcript.setup(JSON.parse(json))", with: ["glyphs": Self.glyphs])
        if let shown { call("transcript.show(JSON.parse(json))", with: shown) }
    }

    /// The page's process ended, as when the system took its memory back: the page loads again
    /// and shows what it showed. Not a fourth time within a minute: what it shows is then what
    /// ends it, and loading it again and again would only spend the Mac's time.
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        loaded = false
        let now = ContinuousClock.now
        terminations = terminations.filter { now - $0 < .seconds(60) } + [now]
        guard terminations.count <= 3 else { return }
        webView.reload()
    }
}

/// Hands the page's messages to the transcript without holding it: the web view's configuration
/// holds its handlers for as long as the web view lives, and the transcript holds the web view.
private final class Messages: NSObject, WKScriptMessageHandler {
    weak var transcript: Transcript?

    init(_ transcript: Transcript) {
        self.transcript = transcript
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        transcript?.receive(message.body)
    }
}

/// The transcript's web view, its context menu without what a browser's has for pages, which the
/// transcript has no use for: Reload, Back and Forward, Inspect Element, and opening or downloading
/// in another window. Copy, Look Up, Search and Share stay.
private final class TranscriptWebView: WKWebView {
    private static let unwanted: Set<String> = [
        "WKMenuItemIdentifierReload", "WKMenuItemIdentifierGoBack", "WKMenuItemIdentifierGoForward",
        "WKMenuItemIdentifierInspectElement", "WKMenuItemIdentifierOpenLink", "WKMenuItemIdentifierOpenLinkInNewWindow",
        "WKMenuItemIdentifierDownloadLinkedFile", "WKMenuItemIdentifierOpenImageInNewWindow",
        "WKMenuItemIdentifierDownloadImage", "WKMenuItemIdentifierOpenFrameInNewWindow",
    ]

    override func willOpenMenu(_ menu: NSMenu, with event: NSEvent) {
        super.willOpenMenu(menu, with: event)
        // No separator at either end or after another, where the items between them went.
        var items: [NSMenuItem] = []
        for item in menu.items where !Self.unwanted.contains(item.identifier?.rawValue ?? "")
            && !(item.isSeparatorItem && items.last?.isSeparatorItem ?? true) {
            items.append(item)
        }
        if items.last?.isSeparatorItem == true { items.removeLast() }
        menu.items = items
    }
}

/// The transcript in the Chat's layout: a view of its own that takes the transcript's one web view
/// in, so that the page outlives the Chat.
struct TranscriptView: NSViewRepresentable {
    let transcript: Transcript
    let page: TranscriptPage

    func makeNSView(context: Context) -> NSView {
        let container = NSView()
        transcript.load()
        if let webView = transcript.webView {
            webView.frame = container.bounds
            webView.autoresizingMask = [.width, .height]
            container.addSubview(webView)
        }
        return container
    }

    func updateNSView(_ container: NSView, context: Context) {
        transcript.show(page)
    }
}

/// A conversation as the transcript's page shows it.
struct TranscriptPage: Encodable, Equatable {
    /// Ask Again, which the last reply has: whether it can ask now, and what it says.
    struct AskAgain: Encodable, Equatable {
        let enabled: Bool
        let help: String
    }

    let id: UUID
    let messages: [TranscriptMessage]
    let askAgain: AskAgain

    /// `conversation`, whose last reply streams while `live` is its; `asleep` while the model
    /// sleeps, to load again before it reads; models named by `name`.
    init(_ conversation: Conversation, live: ChatStore.Live?, asleep: Bool, askAgain: AskAgain, name: (String) -> String) {
        id = conversation.id
        let streaming = live?.conversation == conversation.id ? conversation.messages.last?.id : nil
        messages = conversation.messages.map { message in
            TranscriptMessage(message, live: message.id == streaming ? live : nil, asleep: asleep, name: name)
        }
        self.askAgain = askAgain
    }
}

/// A message as the page shows it.
struct TranscriptMessage: Encodable, Equatable {
    let id: UUID
    let role: ChatMessage.Role
    let text: String
    let reasoning: String
    /// The user's images, sent as data URLs.
    let images: [Data]
    let error: String?
    /// The line over a reply, if it has one.
    let line: ReplyLine?
    /// Whether the reply streams.
    let live: Bool
    /// The model that answered, how fast, and whether it was stopped.
    let details: String

    init(_ message: ChatMessage, live: ChatStore.Live?, asleep: Bool, name: (String) -> String) {
        id = message.id
        role = message.role
        text = message.text
        reasoning = message.reasoning
        images = message.images
        error = message.error
        line = message.role == .assistant ? ReplyLine(message, live: live, asleep: asleep) : nil
        self.live = live != nil
        details = [message.model.map(name), message.tokensPerSecond.map { "\(Format.whole($0)) tok/s" },
                   message.stopped == true ? "Stopped" : nil]
            .compactMap { $0 }.joined(separator: " · ")
    }

    private enum CodingKeys: CodingKey {
        case id, role, text, reasoning, images, error, line, live, details
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(role, forKey: .role)
        try container.encode(text, forKey: .text)
        try container.encode(reasoning, forKey: .reasoning)
        try container.encode(images.map { "data:\(ChatClient.mediaType($0));base64,\($0.base64EncodedString())" }, forKey: .images)
        try container.encode(error, forKey: .error)
        try container.encode(line, forKey: .line)
        try container.encode(live, forKey: .live)
        try container.encode(details, forKey: .details)
    }
}

/// The line over a reply, of what comes before its answer: Loading into memory while a model the
/// idle release put to sleep loads again, Reading while the prompt is read, then Thinking while the
/// model thinks, and once it answers, what the thinking took. While it waits, the page counts the
/// seconds since `since` itself, once a second, from the first.
struct ReplyLine: Encodable, Equatable {
    let verb: String
    /// Its figures, in a fainter tone.
    let facts: String?
    let since: Date?

    // Every key, null for none, as the page reads them.
    private enum CodingKeys: CodingKey {
        case verb, facts, since
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(verb, forKey: .verb)
        try container.encode(facts, forKey: .facts)
        try container.encode(since, forKey: .since)
    }
}

extension ReplyLine {
    /// The line over `message`, streaming while `live` is given; nil for a reply that neither
    /// waits for its answer nor thought.
    init?(_ message: ChatMessage, live: ChatStore.Live?, asleep: Bool) {
        // Once the answer comes, the line says what the thinking took.
        let waiting = message.text.isEmpty ? live : nil
        if let live = waiting, message.reasoning.isEmpty {
            // After the idle release, the model loads again before it reads.
            verb = asleep ? "Loading into memory" : "Reading"
            facts = asleep ? nil : live.read.map { "\(Int($0 * 100))%" }
            since = facts == nil ? live.asked : nil
        } else if let started = waiting?.thinking {
            verb = "Thinking"
            facts = nil
            since = started
        } else if !message.reasoning.isEmpty {
            verb = "Thinking"
            let facts = [message.reasoningTokens.map { "\(Format.count($0)) tokens" }, message.thinkingTime.map(Format.seconds)]
                .compactMap { $0 }.joined(separator: " · ")
            self.facts = facts.isEmpty ? nil : facts
            since = nil
        } else {
            return nil
        }
    }
}

/// An SF Symbol as the page draws it, as a page can't name symbols: a picture at twice its size,
/// which the page uses as a mask and colors as it colors text (the warning sign keeps its own
/// colors), and its size in points.
private struct Glyph: Encodable {
    let url: String
    let width: Double
    let height: Double

    @MainActor
    init(_ name: String, size: CGFloat, weight: NSFont.Weight = .regular, multicolor: Bool = false) {
        let configuration = NSImage.SymbolConfiguration(pointSize: size, weight: weight)
            .applying(multicolor ? .preferringMulticolor() : NSImage.SymbolConfiguration())
        let image = NSImage(systemSymbolName: name, accessibilityDescription: nil)?.withSymbolConfiguration(configuration) ?? NSImage()
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int((image.size.width * 2).rounded(.up)),
                                      pixelsHigh: Int((image.size.height * 2).rounded(.up)), bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0)
        if let bitmap {
            bitmap.size = image.size
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
            NSAppearance(named: .aqua)?.performAsCurrentDrawingAppearance {
                image.draw(in: NSRect(origin: .zero, size: image.size))
            }
            NSGraphicsContext.restoreGraphicsState()
        }
        url = "data:image/png;base64,\(bitmap?.representation(using: .png, properties: [:])?.base64EncodedString() ?? "")"
        width = image.size.width
        height = image.size.height
    }
}

extension Bundle {
    /// The app's own resources, the transcript's page among them: in the app's Resources once
    /// packaged (dev/tools/package_app.py puts them there, as nothing may sit beside Contents),
    /// where the build put them when the app runs from it.
    static let resources = Bundle.main.url(forResource: "Splash_Splash", withExtension: "bundle").flatMap(Bundle.init(url:))
        ?? .module
}
