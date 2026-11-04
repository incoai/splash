import AppKit
import WebKit

/// The web page the Chat's transcript is (`Transcript`), shown in the system's WebKit: its text
/// selectable across what it shows and scrolled by WebKit off the main thread. It loads once, from
/// the app's resources with the runtime's Markdown scripts (`Runtime.markdownScripts`); after that
/// only data goes to it, as JSON. It outlives the views it shows in, so that showing it again never loads it
/// again.
@MainActor
final class WebPage: NSObject {
    /// What the page asks of the app, but copies, which it takes itself.
    var receive: ([String: Any]) -> Void = { _ in }
    /// What the page is to show once it has loaded, sent then.
    var didLoad: () -> Void = {}

    private let page: URL?
    private let scripts: [URL]
    private let glyphs: [String: Glyph]
    /// The schemes the page loads from the app, as the Chat's images, and what serves each.
    private let schemes: [String: any WKURLSchemeHandler]
    private(set) var webView: WKWebView?
    /// Whether the page has loaded, and takes data.
    private(set) var loaded = false
    /// When the page loaded again after its process ended, the last three times, which bound how
    /// often it does.
    private var reloads: [ContinuousClock.Instant] = []

    /// The page `name`.html in `folder` of the app's resources, drawing `glyphs` besides the
    /// symbols of page.css.
    init(_ name: String, in folder: String, runtime: Runtime, glyphs: [String: Glyph],
         schemes: [String: any WKURLSchemeHandler]) {
        page = Bundle.resources.url(forResource: name, withExtension: "html", subdirectory: folder)
        scripts = runtime.markdownScripts
        self.glyphs = Self.glyphs.merging(glyphs) { _, own in own }
        self.schemes = schemes
    }

    /// Loads the page, once: the first time it shows, or before, so that it shows at once.
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
        configuration.userContentController.add(Messages(self), name: "page")
        for (scheme, handler) in schemes { configuration.setURLSchemeHandler(handler, forURLScheme: scheme) }
        let view = PageWebView(frame: NSRect(x: 0, y: 0, width: 800, height: 600), configuration: configuration)
        // The window's background shows through, from the first frame and in either appearance.
        view.setValue(false, forKey: "drawsBackground")
        view.underPageBackgroundColor = .clear
        // A drop would load what was dropped in place of the page.
        view.unregisterDraggedTypes()
        view.navigationDelegate = self
        view.loadFileURL(page, allowingReadAccessTo: page.deletingLastPathComponent())
        webView = view
    }

    /// Calls `body` in the page, with `value` as the JSON string `json`; nothing before it has loaded.
    func call(_ body: String, with value: some Encodable) {
        guard loaded, let webView, let json = try? Self.encoder.encode(value) else { return }
        webView.callAsyncJavaScript(body, arguments: ["json": String(decoding: json, as: UTF8.self)],
                                    in: nil, in: .page, completionHandler: nil)
    }

    /// A view that holds the page's one web view, for a SwiftUI layout to take in: the page moves
    /// into each one made, loading first if it hasn't.
    func container() -> NSView {
        let container = NSView()
        load()
        if let webView {
            webView.frame = container.bounds
            webView.autoresizingMask = [.width, .height]
            container.addSubview(webView)
        }
        return container
    }

    /// How data goes to the page: dates as JavaScript has them, in milliseconds.
    static let encoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .millisecondsSince1970
        return encoder
    }()

    /// A message from the page: a copy, or what its owner takes.
    fileprivate func take(_ body: Any) {
        guard let message = body as? [String: Any] else { return }
        if let text = message["copy"] as? String {
            NSPasteboard.put(text)
        } else {
            receive(message)
        }
    }

    /// The symbols page.css draws, by name.
    private static let glyphs = [
        "chevron": Glyph("chevron.right", size: 10, weight: .semibold),
        "copy": Glyph("doc.on.doc", size: 12),
        "check": Glyph("checkmark", size: 12),
        "warning": Glyph("exclamationmark.triangle.fill", size: 12, multicolor: true),
        "down": Glyph("arrow.down", size: 13, weight: .semibold),
    ]
}

extension WebPage: WKNavigationDelegate {
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
        call("page.setup(JSON.parse(json))", with: ["glyphs": glyphs])
        didLoad()
    }

    /// The page's process ended, as when the system took its memory back: the page loads again
    /// and shows what it showed, three times a minute at most. A fourth time waits until the first
    /// of the three is a minute old: what the page shows may be what ends it, and loading it again
    /// and again would only spend the Mac's time.
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        loaded = false
        let now = ContinuousClock.now
        let at = reloads.count < 3 ? now : max(now, reloads[0] + .seconds(60))
        reloads = Array((reloads + [at]).suffix(3))
        Task { [weak webView] in
            try? await Task.sleep(until: at, clock: .continuous)
            webView?.reload()
        }
    }
}

/// Hands the page's messages to it without holding it: the web view's configuration holds its
/// handlers for as long as the web view lives, and the page holds the web view.
private final class Messages: NSObject, WKScriptMessageHandler {
    weak var page: WebPage?

    init(_ page: WebPage) {
        self.page = page
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        page?.take(message.body)
    }
}

/// The page's web view, its context menu without what a browser's has for pages, which this one
/// has no use for: Reload, Back and Forward, Inspect Element, and opening or downloading in another
/// window. Copy, Look Up, Search and Share stay.
private final class PageWebView: WKWebView {
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

/// An SF Symbol as a page draws it, as a page can't name symbols: a picture at twice its size,
/// which the page uses as a mask and colors as it colors text (the warning sign keeps its own
/// colors), and its size in points.
struct Glyph: Encodable {
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
