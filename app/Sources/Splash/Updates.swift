import Combine
import Foundation
import Sparkle

/// Updates the app with Sparkle, from the appcast every release of Splash carries (`SUFeedURL`),
/// taking an update whose disk image the release key (`SUPublicEDKey`) or, to replace a lost key,
/// the app's Developer ID signed; `dev/tools/package_app.py` writes the keys. Sparkle asks on the
/// second launch whether to check by itself, once a day, and installs an update by quitting the
/// app as its Quit command does, which stops what runs first.
/// An app built without the key, as one `swift build` makes, checks nothing and says nothing.
@MainActor @Observable
final class Updates {
    /// Whether "Check for Updates…" can check now: Sparkle has started and is not checking already.
    private(set) var canCheck = false
    @ObservationIgnored private let controller = SPUStandardUpdaterController(
        startingUpdater: false, updaterDelegate: nil, userDriverDelegate: nil)
    @ObservationIgnored private var observation: AnyCancellable?

    init() {
        // Sparkle changes it on the main thread.
        observation = controller.updater.publisher(for: \.canCheckForUpdates)
            .sink { [weak self] canCheck in MainActor.assumeIsolated { self?.canCheck = canCheck } }
    }

    /// Starts Sparkle's schedule of checks, in an app built with the release key.
    func start() {
        guard Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") != nil else { return }
        controller.startUpdater()
    }

    /// Checks now, showing what Sparkle finds.
    func check() {
        controller.checkForUpdates(nil)
    }
}
