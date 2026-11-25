import Foundation
import Testing

@testable import Splash

/// The app's own words: English, and the Simplified Chinese it ships beside.
struct LocalizationTests {
    @Test func picksTheFirstLanguageItSpeaks() {
        #expect(Localization.language(preferring: ["zh-CN", "en-US"]) == "zh-Hans")
        #expect(Localization.language(preferring: ["en-GB", "zh-Hans"]) == "en")
        #expect(Localization.language(preferring: ["fr-FR", "zh-Hant"]) == "zh-Hans")
        #expect(Localization.language(preferring: ["fr-FR", "de-DE"]) == "en")
    }

    @Test func shipsSimplifiedChineseBesideEnglish() throws {
        let resources = Bundle.resources
        // The bundle names a localization's folder as it stores it, lowercased here.
        let language = try #require(resources.localizations.first { $0.hasPrefix("zh") })
        let chinese = try #require(Bundle(url: resources.bundleURL.appendingPathComponent("\(language).lproj")))
        #expect(chinese.localizedString(forKey: "New Chat", value: nil, table: nil) == "新建对话")
        // The format specifiers keep their arguments' order as the translation needs it.
        #expect(chinese.localizedString(forKey: "Port %d", value: nil, table: nil) == "端口 %1$d")
        #expect(chinese.localizedString(forKey: "%@ tokens", value: nil, table: nil) == "%1$@ 个 token")
        // Its arguments land where the translation puts them.
        let answer = chinese.localizedString(forKey: "%@ is answering %d requests.", value: nil, table: nil)
        #expect(String(format: answer, arguments: ["Qwen", 2]) == "Qwen 正在回答 2 个请求。")
    }
}
