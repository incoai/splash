import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Splash

@MainActor
struct FirstUseTests {
    @Test func examplesFillAnEmptyDraftAndRequestFocus() {
        for example in ChatExample.allCases {
            let draft = Draft()
            draft.suggest(example.text)
            #expect(draft.text == example.text && draft.focusRequests == 1)
            #expect(!draft.isEmpty && draft.images.isEmpty)
        }
    }

    @Test func examplesLeaveExistingTextAndAttachmentsAlone() async throws {
        let draft = Draft()
        draft.text = "My unfinished message"
        draft.suggest(ChatExample.explain.text)
        #expect(draft.text == "My unfinished message" && draft.focusRequests == 0)
        draft.text = ""
        draft.attach([try #require(NSImage(data: Showcase.picture))])
        draft.suggest(ChatExample.write.text)
        #expect(draft.text.isEmpty && draft.focusRequests == 0)
        try await waitFor { draft.attaching == 0 }
        let images = draft.images
        #expect(!images.isEmpty)
        draft.suggest(ChatExample.write.text)
        #expect(draft.text.isEmpty && draft.images == images && draft.focusRequests == 0)
    }

    @Test func aFailedStartDoesNotAdvertiseAHealthySetupStep() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "splash-first-use-failure-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = Runtime(root: root)
        let state = AppState.forTests(in: root.appending(path: "App"), runtime: runtime)
        state.server.start("test-model", settings: ServeSettings())
        guard case .failed = state.server.phase else {
            Issue.record("A missing runtime should fail to start")
            return
        }
        #expect(state.firstUseStage == nil)
    }
}

@MainActor
@Suite(.serialized)
struct FirstUseFlowTests {
    @Test func firstUseGuidesDownloadStartAndAnEditableFirstMessage() async throws {
        let fixture = try FirstUseFixture()
        do {
            let state = fixture.state
            await state.launch()
            #expect(state.firstUseStage == .download && state.installed.isEmpty)
            state.download("test-model")
            #expect(state.firstUseStage == .downloading)
            try fixture.finishDownload()
            try await waitFor("installer completion") { !state.installer.running && state.isInstalled("test-model") }
            #expect(state.firstUseStage == .start && state.selection == "test-model")
            #expect(!state.server.running)
            state.start("test-model")
            #expect(state.firstUseStage == .starting)
            try await waitFor("server ready") { state.server.phase == .ready }
            #expect(state.firstUseStage == .chat)
            state.section = .chat
            state.draft.suggest(ChatExample.explain.text)
            #expect(state.chats.current == nil && state.chats.streaming == nil)
            #expect(try fixture.base.sent().isEmpty)
            state.draft.text += " Use short sentences."
            let edited = state.draft.text
            #expect(state.canSendDraft)
            state.sendDraft()
            try await waitFor("reply completion: \(state.chats.current?.messages.last?.error ?? "no error")") { state.chats.current?.messages.last?.text == "Answer" && state.chats.streaming == nil }
            #expect(state.firstUseStage == nil)
            #expect(state.chats.current?.messages.first?.text == edited)
            #expect(try fixture.base.sent().count == 1)
            state.chats.newChat()
            #expect(state.firstUseStage == nil, "A returning user does not repeat the setup guide")
        } catch {
            await fixture.close()
            throw error
        }
        await fixture.close()
    }
}

@MainActor
@Suite(.serialized, .enabled(if: Snapshot.directory != nil))
struct FirstUseSnapshots {
    @Test func choosingAModelAndOpeningChat() async throws {
        let fixture = try FirstUseFixture()
        do {
            let state = fixture.state
            await state.launch()
            await Snapshot.save("choose-model", MainView().environment(state))
            state.section = .chat
            await Snapshot.save("chat-no-model", MainView().environment(state))
            state.section = .server
            state.download("test-model")
            try await waitFor { state.installer.progress != nil }
            await Snapshot.save("model-downloading", MainView().environment(state))
            try fixture.finishDownload()
            try await waitFor("installer completion") { !state.installer.running && state.isInstalled("test-model") }
            await Snapshot.save("model-start", MainView().environment(state))
            state.start("test-model")
            try await waitFor("server ready") { state.server.phase == .ready }
            await Snapshot.save("model-ready", MainView().environment(state))
            state.section = .chat
            await Snapshot.save("chat-ready", MainView().environment(state))
            await Snapshot.save("chat-ready-compact", size: CGSize(width: 800, height: 560), MainView().environment(state))
            state.draft.suggest(ChatExample.write.text)
            await Snapshot.save("chat-example", MainView().environment(state))
        } catch {
            await fixture.close()
            throw error
        }
        await fixture.close()
    }
}

@MainActor
private func waitFor(_ step: String = "next step", _ condition: () -> Bool) async throws {
    let deadline = ContinuousClock.now + .seconds(10)
    while !condition() {
        try #require(ContinuousClock.now < deadline, "The first-use fixture did not reach \(step)")
        try await Task.sleep(for: .milliseconds(20))
    }
}

/// Drives the actual installer-completion path without network access or model weights.
@MainActor
private final class FirstUseFixture {
    let base: ContextFixture
    let state: AppState

    init() throws {
        base = try ContextFixture()
        let runtime = base.state.runtime
        try FileManager.default.copyItem(at: Showcase.runtime.suggestedModels, to: runtime.suggestedModels)
        try Self.installer.write(to: base.root.appending(path: "install/models.py"), atomically: true, encoding: .utf8)
        state = AppState.forTests(in: base.root.appending(path: "FirstUse"), runtime: runtime)
        state.options = ServeSettings(values: ["--port": String(base.port)])
    }

    func finishDownload() throws {
        try Data().write(to: base.root.appending(path: "finish-download"))
    }

    func close() async {
        state.chatContext.refresh(nil)
        state.chats.stop()
        state.installer.cancel()
        state.server.stop()
        let deadline = ContinuousClock.now + .seconds(6)
        while (state.installer.running || state.server.running) && ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(20))
        }
        await base.close()
    }

    private static let installer = #"""
    import json, time
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    print(json.dumps({'event': 'download', 'repo': 'test-model', 'done': 500000000, 'total': 1000000000}), flush=True)
    while not (root / 'finish-download').exists():
        time.sleep(0.02)
    folder = root / 'install/models/fixture/model'
    folder.mkdir(parents=True)
    (folder / 'fixture.gguf').write_bytes(b'fixture')
    record = {'model': 'test-model', 'files': {'target': {'path': str(folder / 'fixture.gguf'), 'bytes': 7}}}
    (folder / 'model.json').write_text(json.dumps(record))
    """#
}
