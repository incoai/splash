import Foundation

/// The app's own words, in the language macOS picked for Splash: the resource
/// bundle's Localizable.strings, which the app ships in English and Simplified
/// Chinese, English standing in for a key another language lacks. System
/// Settings > General > Language & Region > Applications picks it, and without
/// a choice there the system's own language does.
enum Localization {
    /// The language the app draws and its pages speak: `en`, or `zh-Hans`.
    static var language: String { language(preferring: Locale.preferredLanguages) }

    /// The language `languages` asks for, most wanted first: Simplified Chinese
    /// for any Chinese, English for English, and English for any other.
    static func language(preferring languages: [String]) -> String {
        for tag in languages {
            let lower = tag.lowercased()
            if lower.hasPrefix("zh") { return "zh-Hans" }
            if lower.hasPrefix("en") { return "en" }
        }
        return "en"
    }

    /// `key` in the app's language, with `arguments` for the format specifiers
    /// it holds; the key itself where the language lacks it.
    static func string(_ key: String, _ arguments: [CVarArg]) -> String {
        let format = bundle.localizedString(forKey: key, value: key, table: nil)
        guard !arguments.isEmpty else { return format }
        return String(format: format, arguments: arguments)
    }

    /// The resource bundle's localization for the app's language, which names
    /// its folder as it stores it, lowercased on a case-sensitive volume; the
    /// resource bundle itself, English, where it ships none.
    private static let bundle: Bundle = {
        let resources = Bundle.resources
        guard let name = resources.localizations.first(where: { $0.caseInsensitiveCompare(language) == .orderedSame }),
              let path = resources.path(forResource: name, ofType: "lproj"),
              let localized = Bundle(path: path) else { return resources }
        return localized
    }()
}

/// `key` in the app's language, with `arguments` for its format specifiers.
func L(_ key: String, _ arguments: CVarArg...) -> String {
    Localization.string(key, arguments)
}
