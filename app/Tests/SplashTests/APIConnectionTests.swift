import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Splash

struct APIConnectionTests {
    @Test func usesTheAPIsModelName() {
        let model = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let aliases = ServeSettings(values: ["--served-model-name": "my-model, second", "--announce-served-name": "true"])
        #expect(APIConnection(settings: ServeSettings(), model: model).modelID == model)
        #expect(APIConnection(settings: ServeSettings(), model: "/models/My Model.gguf").modelID == "local/My-Model")
        #expect(APIConnection(settings: aliases, model: model).modelID == "my-model")
        #expect(APIConnection(settings: aliases, model: "/models/My Model.gguf").modelID == "my-model")
        var unannounced = aliases
        unannounced["--announce-served-name"] = nil
        #expect(APIConnection(settings: unannounced, model: model).modelID == model)
    }

    /// Run the copied command in a shell with a recording curl, so JSON, shell metacharacters,
    /// and the environment-variable header are checked as the user's terminal reads them.
    @Test func theCopiedRequestQuotesTheModelAndLeavesTheKeyInTheEnvironment() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-api-shell-\(UUID())")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let marker = folder.appending(path: "must-not-exist")
        let model = "a/model's \"name\" $(touch \(marker.path)) `touch \(marker.path)` \\\nnext"
        let recorder = folder.appending(path: "curl")
        try "#!/bin/sh\n/usr/bin/printf '%s\\0' \"$@\"\n".write(to: recorder, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: recorder.path)
        for key in [nil, "key-from-keychain"] as [String?] {
            let connection = APIConnection(settings: ServeSettings(values: ["--port": "8100"], apiKey: key), model: model)
            #expect(connection.baseURL == "http://127.0.0.1:8100/v1")
            let command = try connection.curl()
            #expect(!command.contains("key-from-keychain"))
            let process = Process(), output = Pipe()
            process.executableURL = URL(filePath: "/bin/sh")
            process.arguments = ["-c", command]
            process.environment = ["PATH": folder.path, "SPLASH_API_KEY": "key-from-environment"]
            process.standardOutput = output
            try process.run()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            try #require(process.terminationStatus == 0)
            let arguments = data.split(separator: 0).map { String(decoding: $0, as: UTF8.self) }
            #expect(arguments.first == "http://127.0.0.1:8100/v1/chat/completions")
            #expect(arguments.contains("Content-Type: application/json"))
            #expect(arguments.contains("Authorization: Bearer key-from-environment") == (key != nil))
            let body = try #require(arguments.last)
            let json = try #require(JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
            #expect(json["model"] as? String == model)
            #expect(json["reasoning_effort"] as? String == "none")
            #expect(json["max_tokens"] as? Int == 128)
            #expect(!FileManager.default.fileExists(atPath: marker.path))
        }
    }
}

@MainActor
@Suite(.serialized)
struct APIConnectionFlowTests {
    @Test func connectionFollowsTheRunningServerAndTheExampleMakesARequest() async throws {
        let fixture = try ContextFixture()
        do {
            let state = fixture.state
            let options = state.options
            let selection = UserDefaults.standard.object(forKey: Preference.model)
            defer {
                state.options = options
                UserDefaults.standard.set(selection, forKey: Preference.model)
            }
            #expect(state.apiConnection == nil)
            let settings = ServeSettings(values: ["--port": String(fixture.port)])
            state.server.start("test-model", settings: settings)
            #expect(state.apiConnection == nil)
            let deadline = ContinuousClock.now + .seconds(10)
            while state.server.phase != .ready {
                try #require(ContinuousClock.now < deadline, "The fixture did not start")
                try await Task.sleep(for: .milliseconds(20))
            }
            let connection = try #require(state.apiConnection)
            #expect(connection.settings == settings && connection.modelID == "test-model")
            try fixture.addModel("another-model")
            state.refresh()
            state.selection = "another-model"
            #expect(state.selection == "another-model")
            state.options["--port"] = "9000"
            state.options["--served-model-name"] = "pending-alias"
            state.options["--announce-served-name"] = "true"
            #expect(state.apiConnection == connection)
            let process = Process(), output = Pipe()
            process.executableURL = URL(filePath: "/bin/sh")
            process.arguments = ["-c", try connection.curl()]
            process.standardOutput = output
            try process.run()
            let reply = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            process.waitUntilExit()
            #expect(process.terminationStatus == 0 && reply.contains("Answer"))
            #expect(try fixture.sent().count == 1)
            var aliases = settings
            aliases["--served-model-name"] = "my-model, another-name"
            aliases["--announce-served-name"] = "true"
            state.server.start("/models/My Model.gguf", settings: aliases)
            #expect(state.apiConnection == nil)
            let switched = ContinuousClock.now + .seconds(10)
            while state.server.phase != .ready || state.server.model != "/models/My Model.gguf" {
                try #require(ContinuousClock.now < switched, "The fixture did not switch")
                try await Task.sleep(for: .milliseconds(20))
            }
            #expect(state.apiConnection?.modelID == "my-model")
            state.server.stop()
            #expect(state.apiConnection == nil)
        } catch {
            await fixture.close()
            throw error
        }
        await fixture.close()
    }
}

@MainActor
@Suite(.serialized, .enabled(if: Snapshot.directory != nil))
struct APIConnectionSnapshots {
    @Test func serverConnection() async throws {
        let fixture = try ContextFixture()
        do {
            let state = fixture.state
            let options = state.options
            let selection = UserDefaults.standard.object(forKey: Preference.model)
            defer {
                state.options = options
                UserDefaults.standard.set(selection, forKey: Preference.model)
            }
            let model = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
            let runtime = state.runtime
            try fixture.addModel(model, bytes: 17_000_000_000)
            try FileManager.default.copyItem(at: Showcase.runtime.suggestedModels, to: runtime.suggestedModels)
            state.refresh()
            state.selection = model
            state.options = ServeSettings(values: ["--port": String(fixture.port)])
            await state.launch()
            state.server.start(model, settings: state.options)
            let deadline = ContinuousClock.now + .seconds(10)
            while state.server.phase != .ready {
                try #require(ContinuousClock.now < deadline, "The fixture did not start")
                try await Task.sleep(for: .milliseconds(20))
            }
            await Snapshot.save("server-api", MainView().environment(state))
            await Snapshot.save("server-api-compact", size: CGSize(width: 800, height: 560), MainView().environment(state))
            let connection = try #require(state.apiConnection)
            await Snapshot.save("server-api-example", size: CGSize(width: 680, height: 560),
                                Form { APIConnectionSection(connection: connection, example: true) }.formStyle(.grouped))
            var protected = connection.settings
            protected.apiKey = "fixture-key-never-shown"
            await Snapshot.save("server-api-key", size: CGSize(width: 680, height: 600), Form {
                APIConnectionSection(connection: APIConnection(settings: protected, model: model), example: true)
            }.formStyle(.grouped))
        } catch {
            await fixture.close()
            throw error
        }
        await fixture.close()
    }
}

private extension ContextFixture {
    /// An installation record for the UI fixture; no model weights are needed or downloaded.
    func addModel(_ model: String, bytes: Int64 = 0) throws {
        let folder = state.runtime.models.appending(path: "fixture/\(UUID())")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let record: [String: Any] = ["model": model, "files": ["target": ["path": "fixture.gguf", "bytes": bytes]]]
        try JSONSerialization.data(withJSONObject: record).write(to: folder.appending(path: "model.json"))
    }
}
