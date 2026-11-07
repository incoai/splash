import AppKit
import SwiftUI
import WebKit

/// Renders the app's windows to PNG files in `SPLASH_APP_SNAPSHOTS`, light and dark, title bar and
/// toolbar included, for people to review the app's look without running it. The window is drawn
/// off screen; web views, which draw in a process of their own, are pictured on their own and laid
/// over it.
@MainActor
enum Snapshot {
    nonisolated static let directory = ProcessInfo.processInfo.environment["SPLASH_APP_SNAPSHOTS"].map { URL(filePath: $0) }

    /// Pictures `view` in a window of `size`, after `arrange`, which runs once the window has drawn.
    static func save(_ name: String, size: CGSize = CGSize(width: 1120, height: 780), _ view: some View,
                     arrange: () async -> Void = {}) async {
        guard let directory else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        _ = NSApplication.shared
        NSApp.setActivationPolicy(.accessory)
        for (suffix, appearance) in [("light", NSAppearance.Name.aqua), ("dark", .darkAqua)] {
            let controller = NSHostingController(rootView: view)
            controller.sceneBridgingOptions = .all
            let window = NSWindow(contentViewController: controller)
            window.title = "Splash"
            window.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
            window.appearance = NSAppearance(named: appearance)
            window.setContentSize(size)
            window.setFrameOrigin(NSPoint(x: -30000, y: -30000))
            window.orderFrontRegardless()
            // Web views come in as SwiftUI lays the window out; WebKit reads the switch as their
            // window comes on screen.
            var seen = Set<ObjectIdentifier>()
            for _ in 0..<15 {
                try? await Task.sleep(for: .seconds(0.1))
                let new = webViews(in: window.contentView!).filter { seen.insert(ObjectIdentifier($0)).inserted }
                guard !new.isEmpty else { continue }
                new.forEach(drawsOffScreen)
                window.orderOut(nil)
                window.orderFrontRegardless()
            }
            await arrange()
            guard let frame = window.contentView?.superview,
                  let bitmap = frame.bitmapImageRepForCachingDisplay(in: frame.bounds)
            else { continue }
            // Each web view's own picture, laid over the window drawn without them, which some
            // systems draw into it and some don't.
            var pictures: [(NSImage, NSRect)] = []
            let webViews = webViews(in: frame)
            for webView in webViews {
                guard let picture = try? await webView.takeSnapshot(configuration: nil) else { continue }
                var rect = webView.convert(webView.bounds, to: frame)
                if frame.isFlipped { rect.origin.y = frame.bounds.height - rect.maxY }
                pictures.append((picture, rect))
            }
            for webView in webViews { webView.isHidden = true }
            frame.cacheDisplay(in: frame.bounds, to: bitmap)
            for webView in webViews { webView.isHidden = false }
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
            for (picture, rect) in pictures { picture.draw(in: rect) }
            NSGraphicsContext.restoreGraphicsState()
            try? bitmap.representation(using: .png, properties: [:])?
                .write(to: directory.appending(path: "\(name)-\(suffix).png"))
            window.orderOut(nil)
        }
    }

    /// A web view in a window off screen counts as hidden, and WebKit then holds back its
    /// animations and its drawing; it draws as on screen once it no longer watches whether its
    /// window is covered (WebKit's own switch for this, `_setWindowOcclusionDetectionEnabled:`).
    static func drawsOffScreen(_ webView: WKWebView) {
        let selector = NSSelectorFromString("_setWindowOcclusionDetectionEnabled:")
        guard webView.responds(to: selector) else { return }
        typealias Setter = @convention(c) (WKWebView, Selector, Bool) -> Void
        unsafeBitCast(webView.method(for: selector), to: Setter.self)(webView, selector, false)
    }

    private static func webViews(in view: NSView) -> [WKWebView] {
        view.subviews.flatMap { ($0 as? WKWebView).map { [$0] } ?? webViews(in: $0) }
    }
}
