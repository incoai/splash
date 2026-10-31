import XCTest
@testable import Splash

final class AppSettingsTests: XCTestCase {
    func testDefaultsAndValidation() {
        let name = UUID().uuidString
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let settings = AppSettings(defaults: defaults)
        XCTAssertEqual(settings.port, 8000)
        XCTAssertEqual(settings.idleRelease, .tenMinutes)
        XCTAssertEqual(IdleRelease.tenMinutes.cliValue, "10m")
        settings.port = 0
        XCTAssertEqual(settings.port, 8000)
        settings.port = 65536
        XCTAssertEqual(settings.port, 8000)
        settings.port = 9000
        XCTAssertEqual(AppSettings(defaults: defaults).port, 9000)
    }

    func testModelCacheDirectoryPersistsAndCanReturnToDefault() {
        let name = UUID().uuidString
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let settings = AppSettings(defaults: defaults)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)

        XCTAssertNil(settings.modelCacheURL)
        settings.modelCacheURL = directory
        XCTAssertEqual(AppSettings(defaults: defaults).modelCacheURL, directory.standardizedFileURL)
        settings.modelCacheURL = nil
        XCTAssertNil(AppSettings(defaults: defaults).modelCacheURL)
    }
}
