import AppKit
import Testing

@testable import Splash

/// What keeps the app in the Dock and the menu bar: a window of its own, on screen or minimized,
/// not the panels and the menu bar item's windows that are not its own.
@MainActor
struct MenuBarAppTests {
    /// A window as AppKit reports it once minimized: not visible, and miniaturized.
    private final class MinimizedWindow: NSWindow {
        override var isVisible: Bool { false }
        override var isMiniaturized: Bool { true }
    }

    private static let off = NSRect(x: -30000, y: -30000, width: 400, height: 300)

    @Test func noWindowLeavesItAMenuBarApp() {
        #expect(AppDelegate.activationPolicy(for: []) == .accessory)
    }

    @Test func aWindowOnScreenMakesItARegularApp() {
        _ = NSApplication.shared
        let window = NSWindow(contentRect: Self.off, styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        defer { window.orderOut(nil) }
        #expect(AppDelegate.activationPolicy(for: [window]) == .accessory)
        window.orderFrontRegardless()
        #expect(AppDelegate.activationPolicy(for: [window]) == .regular)
        window.orderOut(nil)
        #expect(AppDelegate.activationPolicy(for: [window]) == .accessory)
    }

    @Test func aMinimizedWindowKeepsItARegularApp() {
        _ = NSApplication.shared
        let window = MinimizedWindow(contentRect: Self.off, styleMask: [.titled, .closable, .miniaturizable],
                                     backing: .buffered, defer: false)
        #expect(!window.isVisible && window.isMiniaturized)
        #expect(AppDelegate.activationPolicy(for: [window]) == .regular)
    }

    @Test func aPanelLeavesItAMenuBarApp() {
        _ = NSApplication.shared
        let popup = NSPanel(contentRect: Self.off, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        let alert = NSAlert()
        alert.messageText = "Quit Splash?"
        alert.layout()
        alert.window.setFrameOrigin(Self.off.origin)
        defer {
            popup.orderOut(nil)
            alert.window.orderOut(nil)
        }
        popup.orderFrontRegardless()
        alert.window.orderFrontRegardless()
        // The quit question's panel can take the keyboard, as a window of its own can.
        #expect(popup.isVisible && alert.window.isVisible && alert.window.canBecomeKey)
        #expect(AppDelegate.activationPolicy(for: [popup, alert.window]) == .accessory)
    }

    @Test func aWindowThatCannotTakeTheKeyboardLeavesItAMenuBarApp() {
        _ = NSApplication.shared
        // The menu bar item's windows are borderless.
        let chrome = NSWindow(contentRect: Self.off, styleMask: [.borderless], backing: .buffered, defer: false)
        defer { chrome.orderOut(nil) }
        chrome.orderFrontRegardless()
        #expect(chrome.isVisible && !chrome.canBecomeKey)
        #expect(AppDelegate.activationPolicy(for: [chrome]) == .accessory)
    }
}
