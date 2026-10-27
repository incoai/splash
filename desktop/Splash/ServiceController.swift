import Darwin
import Foundation

struct SplashCommand {
    let executable: URL
    let prefixArguments: [String]
    let environment: [String: String]

    func arguments(modelID: String, port: Int, idleRelease: IdleRelease,
                   languageOnly: Bool = false) -> [String] {
        prefixArguments + ["serve", "--model", modelID] +
            (languageOnly ? ["--language-only"] : []) +
            ["--port", String(port), "--idle-release", idleRelease.cliValue]
    }
}

enum ServiceStatus: Equatable {
    case stopped
    case starting
    case ready(modelID: String, weightsReleased: Bool)
    case stopping
    case failed(String)
    case externalPortOwner
}

struct ServiceSnapshot {
    let pid: Int32
    let ready: Bool
    let weightsReleased: Bool
    let activeRequests: Int

    init?(data: Data) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let instance = json["instance"] as? [String: Any],
              let pid = instance["pid"] as? Int,
              let processID = Int32(exactly: pid),
              let ready = json["ready"] as? Bool else { return nil }
        self.pid = processID
        self.ready = ready
        let weights = json["weights"] as? [String: Any]
        weightsReleased = weights?["released"] as? Bool ?? false
        let http = json["http"] as? [String: Any]
        let requests = http?["requests"] as? [String: Any]
        let tokens = http?["token_counts"] as? [String: Any]
        let active = requests?["active"] as? Int ?? 0
        let activeTokens = tokens?["active"] as? Int ?? 0
        let (total, overflow) = active.addingReportingOverflow(activeTokens)
        guard active >= 0, activeTokens >= 0, !overflow else { return nil }
        activeRequests = total
    }
}

enum PortAvailability {
    static func isFree(_ port: Int) -> Bool {
        guard (1...65535).contains(port) else { return false }
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        var reuse: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout.size(ofValue: reuse)))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = UInt16(port).bigEndian
        address.sin_addr = in_addr(s_addr: INADDR_LOOPBACK.bigEndian)
        return withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
    }
}

@MainActor
final class ServiceController: ObservableObject {
    @Published private(set) var status: ServiceStatus = .stopped
    @Published private(set) var log = ""
    private(set) var runningPort: Int?
    private var process: Process?
    private var pollTask: Task<Void, Never>?
    private var pipe: Pipe?
    private let statusFetcher: (@MainActor (Int) async -> ServiceSnapshot?)?
    private var statusAPIKey: String?
    @Published private(set) var runningModelID: String?

    init(statusFetcher: (@MainActor (Int) async -> ServiceSnapshot?)? = nil) {
        self.statusFetcher = statusFetcher
    }

    func start(command: SplashCommand, modelID: String, port: Int, idleRelease: IdleRelease,
               modelCacheURL: URL? = nil, languageOnly: Bool = false) throws {
        guard process == nil else { throw ServiceError.alreadyRunning }
        guard PortAvailability.isFree(port) else {
            status = .externalPortOwner
            throw ServiceError.portOccupied(port)
        }
        let child = Process()
        child.executableURL = command.executable
        child.arguments = command.arguments(modelID: modelID, port: port,
                                            idleRelease: idleRelease, languageOnly: languageOnly)
        var environment = ProcessInfo.processInfo.environment.merging(command.environment) { _, new in new }
        if let modelCacheURL { environment["HF_HUB_CACHE"] = modelCacheURL.path }
        child.environment = environment
        let output = Pipe()
        child.standardOutput = output
        child.standardError = output
        log = ""
        status = .starting
        do {
            try child.run()
        } catch {
            status = .failed(error.localizedDescription)
            throw error
        }
        process = child
        statusAPIKey = environment["SPLASH_API_KEY"]
        pipe = output
        runningPort = port
        runningModelID = modelID
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else { return }
            Task { @MainActor [weak self] in
                self?.appendLog(String(decoding: chunk, as: UTF8.self))
            }
        }
        child.terminationHandler = { [weak self] exited in
            Task { @MainActor [weak self] in self?.didExit(exited) }
        }
        pollTask = Task { [weak self] in await self?.poll() }
    }

    @discardableResult
    func stop(force: Bool = false) async -> Bool {
        guard let child = process else { return true }
        status = .stopping
        pollTask?.cancel()
        if !force, let port = runningPort {
            var failedStatusChecks = 0
            while child.isRunning {
                if let snapshot = await loadSnapshot(port), snapshot.pid == child.processIdentifier {
                    failedStatusChecks = 0
                    if snapshot.activeRequests == 0 { break }
                } else {
                    failedStatusChecks += 1
                    if failedStatusChecks >= 4 {
                        appendLog("\nCould not verify active requests. Use Force Stop to interrupt the service.\n")
                        return false
                    }
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
        if child.isRunning {
            if force { kill(child.processIdentifier, SIGKILL) }
            else { child.terminate() }
        }
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                child.waitUntilExit()
                continuation.resume()
            }
        }
        didExit(child)
        return true
    }

    private func poll() async {
        var failedStatusChecks = 0
        while !Task.isCancelled, let child = process, child.isRunning,
              let port = runningPort {
            let snapshot = await loadSnapshot(port)
            guard process === child, child.isRunning, !Task.isCancelled,
                  status != .stopping else { return }
            if let snapshot {
                failedStatusChecks = 0
                let nextStatus: ServiceStatus
                if snapshot.pid != child.processIdentifier {
                    nextStatus = .externalPortOwner
                } else if snapshot.ready, let modelID = runningModelID {
                    nextStatus = .ready(modelID: modelID, weightsReleased: snapshot.weightsReleased)
                } else {
                    nextStatus = .starting
                }
                if status != nextStatus { status = nextStatus }
            } else if case .ready = status {
                failedStatusChecks += 1
                if failedStatusChecks >= 3 {
                    status = .failed("Cannot reach the Splash service.")
                }
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }

    private func loadSnapshot(_ port: Int) async -> ServiceSnapshot? {
        if let statusFetcher { return await statusFetcher(port) }
        return await Self.fetchStatus(port: port, apiKey: statusAPIKey)
    }

    private static func fetchStatus(port: Int, apiKey: String?) async -> ServiceSnapshot? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/status") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return ServiceSnapshot(data: data)
    }

    private func appendLog(_ text: String) {
        log = String((log + text).suffix(24_000))
    }

    private func didExit(_ child: Process) {
        guard process === child else { return }
        let wasStopping = status == .stopping
        pipe?.fileHandleForReading.readabilityHandler = nil
        pipe = nil
        process = nil
        statusAPIKey = nil
        pollTask?.cancel()
        pollTask = nil
        runningPort = nil
        runningModelID = nil
        let exitDetail = child.terminationReason == .uncaughtSignal
            ? "Splash was terminated by signal \(child.terminationStatus)."
            : "Splash exited with code \(child.terminationStatus)."
        status = wasStopping ? .stopped : .failed(exitDetail)
    }
}

enum ServiceError: LocalizedError {
    case alreadyRunning
    case portOccupied(Int)

    var errorDescription: String? {
        switch self {
        case .alreadyRunning: return "A Splash service is already running."
        case .portOccupied(let port): return "Port \(port) is in use."
        }
    }
}
