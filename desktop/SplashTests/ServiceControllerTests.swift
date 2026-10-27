import Darwin
import XCTest
@testable import Splash

final class ServiceControllerTests: XCTestCase {
    func testServeArguments() {
        let command = SplashCommand(executable: URL(fileURLWithPath: "/usr/bin/true"),
                                    prefixArguments: ["-u", "/runtime/install/launcher.py"],
                                    environment: [:])
        XCTAssertEqual(command.arguments(modelID: "owner/repo", port: 8123, idleRelease: .fiveMinutes),
                       ["-u", "/runtime/install/launcher.py", "serve", "--model", "owner/repo",
                        "--port", "8123", "--idle-release", "5m"])
        XCTAssertEqual(command.arguments(modelID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
                                         port: 8123, idleRelease: .fiveMinutes,
                                         languageOnly: true),
                       ["-u", "/runtime/install/launcher.py", "serve", "--model",
                        "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit", "--language-only",
                        "--port", "8123", "--idle-release", "5m"])
    }

    func testStatusSnapshotTracksOwnedPIDAndActiveWork() throws {
        let data = Data(#"{"instance":{"pid":1234},"ready":true,"weights":{"released":true},"http":{"requests":{"active":2},"token_counts":{"active":1}}}"#.utf8)
        let snapshot = try XCTUnwrap(ServiceSnapshot(data: data))
        XCTAssertEqual(snapshot.pid, 1234)
        XCTAssertTrue(snapshot.ready)
        XCTAssertTrue(snapshot.weightsReleased)
        XCTAssertEqual(snapshot.activeRequests, 3)
    }

    func testStatusSnapshotRejectsInvalidPIDAndCounts() {
        let oversizedPID = Data(#"{"instance":{"pid":2147483648},"ready":true}"#.utf8)
        XCTAssertNil(ServiceSnapshot(data: oversizedPID))

        let overflowingRequests = Data(
            #"{"instance":{"pid":1234},"ready":true,"http":{"requests":{"active":9223372036854775807},"token_counts":{"active":1}}}"#.utf8
        )
        XCTAssertNil(ServiceSnapshot(data: overflowingRequests))

        let negativeRequests = Data(
            #"{"instance":{"pid":1234},"ready":true,"http":{"requests":{"active":-1}}}"#.utf8
        )
        XCTAssertNil(ServiceSnapshot(data: negativeRequests))

        let negativeTokens = Data(
            #"{"instance":{"pid":1234},"ready":true,"http":{"token_counts":{"active":-1}}}"#.utf8
        )
        XCTAssertNil(ServiceSnapshot(data: negativeTokens))
    }

    @MainActor
    func testOccupiedPortDoesNotLaunchOrOwnService() throws {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: INADDR_LOOPBACK.bigEndian)
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        XCTAssertEqual(bound, 0)
        XCTAssertEqual(Darwin.listen(descriptor, 1), 0)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                _ = getsockname(descriptor, $0, &length)
            }
        }
        let port = Int(UInt16(bigEndian: address.sin_port))
        let service = ServiceController()
        let command = SplashCommand(executable: URL(fileURLWithPath: "/usr/bin/true"),
                                    prefixArguments: [], environment: [:])

        XCTAssertThrowsError(try service.start(command: command, modelID: "owner/repo", port: port,
                                               idleRelease: .tenMinutes)) { error in
            guard case ServiceError.portOccupied(let occupiedPort) = error else {
                return XCTFail("Expected occupied-port error, got \(error)")
            }
            XCTAssertEqual(occupiedPort, port)
        }
        XCTAssertEqual(service.status, .externalPortOwner)
        XCTAssertNil(service.runningPort)
        XCTAssertNil(service.runningModelID)
    }

    @MainActor
    func testLaunchFailureDoesNotLeaveServiceOwned() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathComponent("missing-splash")
        let port = try XCTUnwrap((21001...22000).first(where: PortAvailability.isFree))
        let service = ServiceController()
        let command = SplashCommand(executable: missing, prefixArguments: [], environment: [:])

        XCTAssertThrowsError(try service.start(command: command, modelID: "owner/repo", port: port,
                                               idleRelease: .tenMinutes))
        guard case .failed(let message) = service.status else {
            return XCTFail("Expected a failed service status")
        }
        XCTAssertFalse(message.isEmpty)
        XCTAssertNil(service.runningPort)
        XCTAssertNil(service.runningModelID)
    }

    @MainActor
    func testStartsAndStopsOwnedService() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let script = directory.appendingPathComponent("fake_service.py")
        let source = #"""
import argparse
import http.server
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int)
args, _ = parser.parse_known_args()
started = time.monotonic()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        active = 1 if time.monotonic() - started < 4 else 0
        body = json.dumps({"instance": {"pid": os.getpid()}, "ready": True,
                           "weights": {"released": False},
                           "http": {"requests": {"active": active}, "token_counts": {"active": 0}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass

print("service ready; cache=" + os.environ.get("HF_HUB_CACHE", ""), flush=True)
http.server.HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
"""#
        try Data(source.utf8).write(to: script)
        let port = try XCTUnwrap((18000...19000).first(where: PortAvailability.isFree))
        let command = SplashCommand(executable: URL(fileURLWithPath: "/usr/bin/python3"),
                                    prefixArguments: ["-u", script.path], environment: [:])
        let service = ServiceController()
        let cache = directory.appendingPathComponent("cache")
        try service.start(command: command, modelID: "owner/repo", port: port,
                          idleRelease: .tenMinutes, modelCacheURL: cache)
        defer { Task { await service.stop(force: true) } }
        for _ in 0..<60 {
            if case .ready(let model, _) = service.status {
                XCTAssertEqual(model, "owner/repo")
                break
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(service.status, .ready(modelID: "owner/repo", weightsReleased: false))
        let statusURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/status"))
        let (data, _) = try await URLSession.shared.data(from: statusURL)
        XCTAssertGreaterThan(try XCTUnwrap(ServiceSnapshot(data: data)).activeRequests, 0)
        let stoppedAt = Date()
        await service.stop()
        XCTAssertGreaterThan(Date().timeIntervalSince(stoppedAt), 0.5)
        XCTAssertEqual(service.status, .stopped)
        XCTAssertNil(service.runningPort)
        XCTAssertTrue(service.log.contains("service ready; cache=\(cache.path)"))
    }

    @MainActor
    func testLateStatusResponseDoesNotUndoStoppingState() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let pidFile = directory.appendingPathComponent("pid")
        let script = directory.appendingPathComponent("service.py")
        try Data("import os, pathlib, time\npathlib.Path('\(pidFile.path)').write_text(str(os.getpid()))\ntime.sleep(10)\n".utf8)
            .write(to: script)
        let probe = DelayedStatusProbe()
        let service = ServiceController(statusFetcher: { _ in await probe.fetch() })
        let port = try XCTUnwrap((20001...21000).first(where: PortAvailability.isFree))
        try service.start(command: SplashCommand(executable: URL(fileURLWithPath: "/usr/bin/python3"),
                                                 prefixArguments: ["-u", script.path], environment: [:]),
                          modelID: "owner/repo", port: port, idleRelease: .tenMinutes)
        var stopping: Task<Void, Never>?
        do {
            for _ in 0..<100 where probe.firstRequest == nil || !FileManager.default.fileExists(atPath: pidFile.path) {
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            _ = try XCTUnwrap(probe.firstRequest, "Poll did not request status")
            let pid = try XCTUnwrap(Int32(try String(contentsOf: pidFile, encoding: .utf8)))
            probe.pid = pid
            stopping = Task { await service.stop() }
            for _ in 0..<100 where service.status != .stopping || probe.requestCount < 2 {
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            XCTAssertEqual(service.status, .stopping)
            XCTAssertGreaterThanOrEqual(probe.requestCount, 2)
            probe.releaseFirstRequest()
            await probe.waitForFirstResponse()
            await Task.yield()
            XCTAssertEqual(service.status, .stopping)
        } catch {
            probe.releaseFirstRequest()
            await service.stop(force: true)
            await stopping?.value
            throw error
        }
        await service.stop(force: true)
        await stopping?.value
    }

    @MainActor
    func testStopDoesNotWaitForeverAfterStatusBecomesUnavailable() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let pidFile = directory.appendingPathComponent("pid")
        let script = directory.appendingPathComponent("service.py")
        let source = """
        import os, pathlib, time
        pathlib.Path(\(String(reflecting: pidFile.path))).write_text(str(os.getpid()))
        time.sleep(10)
        """
        try Data(source.utf8).write(to: script)

        var pid: Int32 = 0
        var fetches = 0
        var stopping = false
        let service = ServiceController(statusFetcher: { _ in
            guard pid != 0 else { return nil }
            let snapshot = ServiceSnapshot(data: Data(
                "{\"instance\":{\"pid\":\(pid)},\"ready\":true,\"http\":{\"requests\":{\"active\":1}}}".utf8
            ))
            guard stopping, !Task.isCancelled else { return snapshot }
            fetches += 1
            guard fetches <= 2 || fetches == 4 else { return nil }
            return snapshot
        })
        let port = try XCTUnwrap((22001...23000).first(where: PortAvailability.isFree))
        try service.start(command: SplashCommand(executable: URL(fileURLWithPath: "/usr/bin/python3"),
                                                 prefixArguments: ["-u", script.path], environment: [:]),
                          modelID: "owner/repo", port: port, idleRelease: .tenMinutes)
        defer { Task { await service.stop(force: true) } }
        for _ in 0..<100 where !FileManager.default.fileExists(atPath: pidFile.path) {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        pid = try XCTUnwrap(Int32(String(contentsOf: pidFile, encoding: .utf8)))

        stopping = true
        let stoppedAt = Date()
        await service.stop()
        XCTAssertLessThan(Date().timeIntervalSince(stoppedAt), 7)
        XCTAssertEqual(service.status, .stopping)
        XCTAssertEqual(service.runningPort, port)
        XCTAssertGreaterThanOrEqual(fetches, 8)
        await service.stop(force: true)
        XCTAssertEqual(service.status, .stopped)
    }

    @MainActor
    func testFirstStatusFailureDoesNotTerminateOwnedService() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = directory.appendingPathComponent("service.py")
        try Data("import time\ntime.sleep(30)\n".utf8).write(to: script)
        let service = ServiceController(statusFetcher: { _ in nil })
        let port = try XCTUnwrap((23001...24000).first(where: PortAvailability.isFree))
        try service.start(command: SplashCommand(executable: URL(fileURLWithPath: "/usr/bin/python3"),
                                                 prefixArguments: ["-u", script.path], environment: [:]),
                          modelID: "owner/repo", port: port, idleRelease: .tenMinutes)
        let stopped = await service.stop()
        XCTAssertFalse(stopped)
        XCTAssertEqual(service.status, .stopping)
        XCTAssertEqual(service.runningPort, port)
        let forced = await service.stop(force: true)
        XCTAssertTrue(forced)
        XCTAssertEqual(service.status, .stopped)
    }

    @MainActor
    func testAPIKeyProtectedStatusCanBecomeReady() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = directory.appendingPathComponent("service.py")
        let source = """
        import argparse, http.server, json, os
        parser = argparse.ArgumentParser()
        parser.add_argument('--port', type=int)
        args, _ = parser.parse_known_args()
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.headers.get('Authorization') != 'Bearer test-desktop-key':
                    self.send_response(401)
                    self.end_headers()
                    return
                body = json.dumps({'instance': {'pid': os.getpid()}, 'ready': True}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_):
                pass
        http.server.HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
        """
        try Data(source.utf8).write(to: script)
        let port = try XCTUnwrap((24001...25000).first(where: PortAvailability.isFree))
        let service = ServiceController()
        try service.start(command: SplashCommand(executable: URL(fileURLWithPath: "/usr/bin/python3"),
                                                 prefixArguments: ["-u", script.path],
                                                 environment: ["SPLASH_API_KEY": "test-desktop-key"]),
                          modelID: "owner/repo", port: port, idleRelease: .tenMinutes)
        defer { Task { await service.stop(force: true) } }
        for _ in 0..<40 where service.status != .ready(modelID: "owner/repo", weightsReleased: false) {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(service.status, .ready(modelID: "owner/repo", weightsReleased: false))
        await service.stop(force: true)
    }

    @MainActor
    func testReadinessExpiresWhenStatusIsUnavailableAndRecovers() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let pidFile = directory.appendingPathComponent("pid")
        let script = directory.appendingPathComponent("service.py")
        let source = """
        import os, pathlib, time
        pathlib.Path(\(String(reflecting: pidFile.path))).write_text(str(os.getpid()))
        time.sleep(30)
        """
        try Data(source.utf8).write(to: script)
        var available = true
        let service = ServiceController(statusFetcher: { _ in
            guard available, let pid = try? String(contentsOf: pidFile, encoding: .utf8) else { return nil }
            return ServiceSnapshot(data: Data("{\"instance\":{\"pid\":\(pid)},\"ready\":true}".utf8))
        })
        let port = try XCTUnwrap((25001...26000).first(where: PortAvailability.isFree))
        try service.start(command: SplashCommand(executable: URL(fileURLWithPath: "/usr/bin/python3"),
                                                 prefixArguments: ["-u", script.path], environment: [:]),
                          modelID: "owner/repo", port: port, idleRelease: .tenMinutes)
        defer { Task { await service.stop(force: true) } }
        for _ in 0..<50 where service.status != .ready(modelID: "owner/repo", weightsReleased: false) {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(service.status, .ready(modelID: "owner/repo", weightsReleased: false))

        available = false
        for _ in 0..<60 {
            if case .failed = service.status { break }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        guard case .failed = service.status else { return XCTFail("Stale Ready status was retained") }
        XCTAssertEqual(service.runningPort, port)

        available = true
        for _ in 0..<30 where service.status != .ready(modelID: "owner/repo", weightsReleased: false) {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(service.status, .ready(modelID: "owner/repo", weightsReleased: false))
        await service.stop(force: true)
    }
}

@MainActor
private final class DelayedStatusProbe {
    var pid: Int32 = 0
    var requestCount = 0
    var firstRequest: CheckedContinuation<ServiceSnapshot?, Never>?
    private var firstResponseReturned = false
    private var firstResponseWaiter: CheckedContinuation<Void, Never>?

    func fetch() async -> ServiceSnapshot? {
        requestCount += 1
        if requestCount == 1 {
            let response = await withCheckedContinuation { firstRequest = $0 }
            firstResponseReturned = true
            firstResponseWaiter?.resume()
            firstResponseWaiter = nil
            return response
        }
        return snapshot(activeRequests: 1)
    }

    func waitForFirstResponse() async {
        if firstResponseReturned { return }
        await withCheckedContinuation { firstResponseWaiter = $0 }
    }

    func releaseFirstRequest() {
        firstRequest?.resume(returning: snapshot(activeRequests: 1))
        firstRequest = nil
    }

    private func snapshot(activeRequests: Int) -> ServiceSnapshot? {
        ServiceSnapshot(data: Data("{\"instance\":{\"pid\":\(pid)},\"ready\":true,\"http\":{\"requests\":{\"active\":\(activeRequests)}}}".utf8))
    }
}
