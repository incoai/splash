import Foundation

enum IdleRelease: String, CaseIterable, Identifiable {
    case fiveMinutes = "5m"
    case tenMinutes = "10m"
    case off = "off"

    var id: String { rawValue }
    var cliValue: String { rawValue }
    var title: String {
        switch self {
        case .fiveMinutes: return "5 minutes"
        case .tenMinutes: return "10 minutes"
        case .off: return "Off"
        }
    }
}

final class AppSettings: ObservableObject {
    private enum Key {
        static let port = "splash.desktop.port"
        static let idleRelease = "splash.desktop.idleRelease"
        static let selectedModel = "splash.desktop.selectedModel"
        static let modelCacheDirectory = "splash.desktop.modelCacheDirectory"
    }

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    var port: Int {
        get {
            let value = defaults.integer(forKey: Key.port)
            return (1...65535).contains(value) ? value : 8000
        }
        set {
            guard (1...65535).contains(newValue) else { return }
            objectWillChange.send()
            defaults.set(newValue, forKey: Key.port)
        }
    }

    var idleRelease: IdleRelease {
        get { IdleRelease(rawValue: defaults.string(forKey: Key.idleRelease) ?? "") ?? .tenMinutes }
        set {
            objectWillChange.send()
            defaults.set(newValue.rawValue, forKey: Key.idleRelease)
        }
    }

    var selectedModelID: String? {
        get { defaults.string(forKey: Key.selectedModel) }
        set {
            objectWillChange.send()
            defaults.set(newValue, forKey: Key.selectedModel)
        }
    }

    var modelCacheURL: URL? {
        get {
            guard let path = defaults.string(forKey: Key.modelCacheDirectory),
                  (path as NSString).isAbsolutePath else { return nil }
            return URL(fileURLWithPath: path).standardizedFileURL
        }
        set {
            objectWillChange.send()
            defaults.set(newValue?.standardizedFileURL.path, forKey: Key.modelCacheDirectory)
        }
    }
}
