import Foundation
import SwiftUI
import Testing

@testable import Splash

struct PendingOptionsTests {
    @Test func namesMainAndAdvancedChangesWithTheirRunningValues() {
        let settings = ServeSettings(values: ["--max-context": "32768", "--max-memory": "40G",
                                              "--port": "9000", "--no-webui": "true"])
        let changes = settings.changes(from: ServeSettings())
        #expect(changes.map(\.id) == ["pending:--max-context", "pending:--max-memory", "pending:--port", "pending:--no-webui"])
        // Both groups render in one Form section: a duplicate ID can replace an editable control.
        #expect(Set(changes.map(\.id)).isDisjoint(with: Set(LaunchOption.all.map(\.id))))
        #expect(changes.map(\.detail) == ["Auto → 32K tokens", "Auto → 40 GB", "8000 → 9000", "On → Off"])
        #expect(changes[2].title == "Network and Access: Port")
    }

    @Test func equivalentPortsListsAndUnusedOptionsDoNotAppear() {
        let running = ServeSettings(values: ["--max-cache-disk": "0", "--allowed-host": "a.example, b.example"])
        let pending = ServeSettings(values: ["--port": "8000", "--max-cache-disk": "0", "--persistent-cache": "true",
                                            "--cache-dir": "/tmp/cache", "--max-cache-disk-writes": "100G",
                                            "--allowed-host": "a.example b.example"])
        #expect(!pending.differs(from: running, model: "test-model"))
        #expect(pending.changes(from: running).isEmpty)
    }

    @Test func enablingAnOptionShowsTheDependentSettingsThatWillTakeEffect() {
        let running = ServeSettings(values: ["--max-cache-disk": "0", "--persistent-cache": "true", "--cache-dir": "/tmp/cache"])
        var pending = running
        pending["--max-cache-disk"] = "16G"
        let changes = pending.changes(from: running)
        #expect(changes.map(\.id) == ["pending:--max-cache-disk", "pending:--persistent-cache", "pending:--cache-dir"])
        #expect(changes.map(\.detail) == ["Off → 16 GB", "Not used → On", "Not used → /tmp/cache"])
    }

    @Test func keyChangesAreDescribedWithoutIncludingEitherKey() throws {
        for (before, after, description) in [(nil, "new-private-key", "Not required → Required"),
                                             ("old-private-key", "new-private-key", "Changed"),
                                             ("old-private-key", nil, "Required → Not required")] as [(String?, String?, String)] {
            let changes = ServeSettings(apiKey: after).changes(from: ServeSettings(apiKey: before))
            let change = try #require(changes.first)
            #expect(changes.count == 1 && change.id == "pending:--api-key" && change.detail == description)
            #expect(!String(reflecting: changes).contains("private-key"))
        }
    }

    /// The review and the existing Restart to Apply action must agree, including each launch
    /// option's applicability and the server's special handling of the port and API key.
    @Test func reviewMatchesTheExistingRestartDecisionForEveryOption() {
        let configurations = [ServeSettings(), ServeSettings(values: ["--max-cache-disk": "0"]),
                              ServeSettings(values: ["--persistent-cache": "true", "--cache-dir": "/tmp/cache"])]
        for running in configurations {
            for option in LaunchOption.all {
                let values: [String]
                switch option.kind {
                case .toggle: values = ["true"]
                case .choice(_, let choices, let custom):
                    values = choices.map(\.value) + [custom.flatMap { $0.parse($0.prompt) }].compactMap { $0 }
                case .text(let custom, _): values = [custom.parse(custom.prompt)].compactMap { $0 }
                case .folder: values = ["/tmp/another-cache"]
                case .secret: values = ["fixture-key"]
                }
                for value in values {
                    var pending = running
                    if case .secret = option.kind { pending.apiKey = value } else { pending[option.flag] = value }
                    #expect(!pending.changes(from: running).isEmpty == pending.differs(from: running, model: "test-model"),
                            "\(option.flag): \(value)")
                }
            }
        }
    }
}

@MainActor
@Suite(.serialized)
struct PendingOptionsFlowTests {
    @Test func discardKeepsServingAndRestartAppliesTheReviewedSettings() async throws {
        let fixture = try ContextFixture()
        do {
            let state = fixture.state
            let original = state.options
            let selection = UserDefaults.standard.object(forKey: Preference.model)
            defer {
                state.options = original
                UserDefaults.standard.set(selection, forKey: Preference.model)
            }
            let running = ServeSettings(values: ["--port": String(fixture.port), "--max-memory": "24G"])
            state.options = running
            #expect(state.pendingOptions.isEmpty)
            await state.launch()
            state.start("test-model")
            try await ready(state)
            let since = state.server.since
            let pid = state.runtime.serveLockOwner(port: fixture.port)
            state.options["--max-context"] = "32768"
            state.options["--max-memory"] = "16G"
            #expect(state.optionsChanged && state.pendingOptions.count == 2)
            #expect(state.server.settings == running)
            state.discardOptions()
            #expect(state.options == running && state.pendingOptions.isEmpty && !state.optionsChanged)
            #expect(state.server.phase == .ready && state.server.since == since)
            #expect(state.runtime.serveLockOwner(port: fixture.port) == pid)
            // Discard restores the running configuration, including its nondefault memory limit.
            #expect(state.options["--max-memory"] == "24G")
            state.options["--max-context"] = "32768"
            let pending = state.options
            state.start("test-model")
            #expect(state.pendingOptions.isEmpty)
            try await ready(state)
            #expect(state.server.settings == pending && state.pendingOptions.isEmpty && !state.optionsChanged)
            #expect(state.server.since != since)
            state.stop()
            state.options["--max-context"] = "8192"
            let stopped = state.options
            state.discardOptions()
            #expect(state.options == stopped && state.pendingOptions.isEmpty)
        } catch {
            await fixture.close()
            throw error
        }
        await fixture.close()
    }

    private func ready(_ state: AppState) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while state.server.phase != .ready {
            try #require(ContinuousClock.now < deadline, "The fixture did not become ready")
            try await Task.sleep(for: .milliseconds(20))
        }
    }

}

@MainActor
@Suite(.serialized, .enabled(if: Snapshot.directory != nil))
struct PendingOptionsSnapshots {
    @Test func pendingLaunchSettings() async throws {
        let fixture = try ContextFixture()
        do {
            let state = fixture.state
            let original = state.options
            let selection = UserDefaults.standard.object(forKey: Preference.model)
            defer {
                state.options = original
                UserDefaults.standard.set(selection, forKey: Preference.model)
            }
            let folder = state.runtime.models.appending(path: "fixture/model")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try Data(#"{"model":"test-model","files":{}}"#.utf8).write(to: folder.appending(path: "model.json"))
            state.refresh()
            let running = ServeSettings(values: ["--port": String(fixture.port)])
            await state.launch()
            state.server.start("test-model", settings: running)
            let deadline = ContinuousClock.now + .seconds(10)
            while state.server.phase != .ready {
                try #require(ContinuousClock.now < deadline, "The fixture did not start")
                try await Task.sleep(for: .milliseconds(20))
            }
            state.options = running
            state.options["--max-context"] = "32768"
            state.options["--port"] = "9000"
            await Snapshot.save("launch-options", size: CGSize(width: 680, height: 600),
                                Form { LaunchOptions() }.formStyle(.grouped).environment(state))
            await Snapshot.save("launch-options-compact", size: CGSize(width: 540, height: 600),
                                Form { LaunchOptions() }.formStyle(.grouped).environment(state))
            await Snapshot.save("server-pending", MainView().environment(state))
            await Snapshot.save("launch-options-review", size: CGSize(width: 680, height: 680),
                                Form { LaunchOptions(reviewChanges: true) }.formStyle(.grouped).environment(state))
            state.options["--allowed-host"] = "inference.example.test, another-inference.example.test"
            await Snapshot.save("launch-options-review-compact", size: CGSize(width: 540, height: 720),
                                Form { LaunchOptions(reviewChanges: true) }.formStyle(.grouped).environment(state))
            state.discardOptions()
            await Snapshot.save("launch-options-discarded", size: CGSize(width: 680, height: 600),
                                Form { LaunchOptions() }.formStyle(.grouped).environment(state))
        } catch {
            await fixture.close()
            throw error
        }
        await fixture.close()
    }
}
