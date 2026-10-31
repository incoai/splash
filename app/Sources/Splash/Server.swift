import Darwin
import Foundation
import Observation

/// The one `splash serve` this app runs. It starts an installed model, follows it to Ready, stops
/// it as Ctrl+C would, and switches models by stopping one and starting the next.
@MainActor @Observable
final class Server {
    enum Phase: Equatable {
        case stopped, starting, ready, stopping
        case failed(String)
    }

    /// What a start does, as the server's log tells it: the server starts, reads the model's
    /// weights into memory, then runs the model once at full size so that its first answer comes
    /// quickly; the first start after an update also prepares the GPU's programs then, which can
    /// take a minute.
    enum Step: String {
        case starting = "Starting"
        case loading = "Loading weights"
        case warming = "Warming up"

        /// The step `line` of the server's log begins, if it begins one.
        init?(_ line: String) {
            if line.contains("Loading · ") {
                self = .loading
            } else if line.contains("Weights loaded") {
                self = .warming
            } else {
                return nil
            }
        }
    }

    /// What a ready server does: runs the model, sleeps with its weights given back, or recovers
    /// while the engine, having failed, starts again.
    enum Serving: String {
        case running = "Running"
        case sleeping = "Sleeping"
        case recovering = "Recovering…"
    }

    private(set) var phase = Phase.stopped
    /// While starting: what the start does, and since when.
    private(set) var step = Step.starting
    private(set) var stepSince = Date.now
    /// The model starting, served or last failed.
    private(set) var model: String?
    /// The start to make once the model stopping has stopped.
    private var queued: (model: String, settings: ServeSettings)?
    /// The options of the model starting, served or last failed.
    private(set) var settings = ServeSettings()
    /// When the model starting, served or last failed began to start.
    private(set) var since = Date.now
    /// Once ready: whether the engine has given the weights back while idle.
    private(set) var asleep = false
    /// Once ready: whether the engine failed and the server is starting it again.
    private(set) var recovering = false
    /// Once ready: the tokens the model's context holds, as the option set it or Auto fitted it.
    private(set) var context: Int?
    /// Once ready: whether the model takes images.
    private(set) var acceptsImages = false
    /// Once ready: what a reasoning effort does to the model.
    private(set) var reasoning: Reasoning?
    /// Once ready: the work the server reports, as its status gives it.
    private(set) var activity = Activity()
    /// Once ready: what the server does.
    var serving: Serving { recovering ? .recovering : asleep ? .sleeping : .running }
    /// Whether the model answers a request now: ready, and not recovering.
    var answers: Bool { phase == .ready && !recovering }

    /// Once ready: each request the server finished, as its log line gives it, the latest last.
    private(set) var requestLog: [RequestEntry] = []

    /// The server's work since it started, from `/status`: requests served and running, the average
    /// speed of answers and of reading prompts in tokens a second, the median time to an answer's
    /// first token, the tokens it generated, read and took from its cache, and the memory the
    /// engine holds.
    struct Activity: Equatable {
        var served = 0
        var running = 0
        var generation: Double?
        var reading: Double?
        var firstToken: Duration?
        var generated = 0
        var read = 0
        var cached = 0
        var memory: Int64?
    }

    /// What a reasoning effort does to the served model, as `/v1/models` reports it from its
    /// template.
    struct Reasoning: Decodable, Equatable {
        /// Whether a message that asks for no effort makes the model think; nil for a model that
        /// decides itself.
        var thinks: Bool?
        /// The efforts besides none the model tells apart, least first; none where its thinking
        /// is only on or off.
        var efforts: [String]
        /// The one of them a message that asks for no effort gets.
        var `default`: String?
    }

    static let log = URL.libraryDirectory.appending(path: "Logs/Splash/server.log")

    private let runtime: Runtime
    private let log: URL
    private var child: ChildProcess?
    private let record = ChildRecord(key: Preference.serverPID)
    private var tail: LogTail?
    private var stopRequested = false
    private var firstError: String?
    private var watch: Task<Void, Never>?

    init(runtime: Runtime, log: URL = Server.log) {
        self.runtime = runtime
        self.log = log
    }

    var running: Bool { child != nil }

    /// The model to start once the one stopping has stopped: another one, or the same one again.
    var next: String? { queued?.model }

    /// Starts installed `model` with `settings`, stopping the one being served first.
    func start(_ model: String, settings: ServeSettings) {
        if child != nil {
            stop()
            queued = (model, settings)
        } else {
            launch(model, settings: settings)
        }
    }

    /// Stops the model, and drops a switch to another one: SIGINT to the process group, as Ctrl+C
    /// sends; a second one after 10 s, on which the server stops the engine at once; SIGKILL if
    /// anything still runs 20 s later.
    func stop() {
        queued = nil
        guard let child, phase != .stopping else { return }
        stopRequested = true
        phase = .stopping
        watch?.cancel()
        child.signal(SIGINT)
        Task { [weak self] in
            for (delay, signal) in [(10, SIGINT), (20, SIGKILL)] {
                try? await Task.sleep(for: .seconds(delay))
                guard self?.child === child else { return }
                child.signal(signal)
            }
        }
    }

    /// Clears a failure the user has read, stopping a server that still runs but serves nothing.
    func dismissFailure() {
        guard case .failed = phase else { return }
        if child != nil { stop() } else { phase = .stopped }
    }

    /// Stops the server an app started before it quit or crashed, if it still serves, and waits
    /// until it has given its port and memory back. It stops by itself once that app has ended
    /// (`SPLASH_APP_PID`); this stops one that has not yet, or that an older app started.
    func stopLeftover() async {
        await record.stopLeftover()
    }

    private func launch(_ model: String, settings: ServeSettings) {
        self.model = model
        self.settings = settings
        asleep = false
        recovering = false
        context = nil
        activity = Activity()
        requestLog = []
        acceptsImages = false
        reasoning = nil
        firstError = nil
        stopRequested = false
        resetLog(log)
        if portInUse(settings.port) {
            return refuse("Port \(settings.port) is in use by another app. Quit that app, or choose another port in Launch Options › Advanced.")
        }
        since = .now
        step = .starting
        stepSince = .now
        phase = .starting
        do {
            // It stops as Ctrl+C stops it once this app has ended, a crash too, giving its memory
            // back then rather than at the app's next launch.
            child = try ChildProcess(
                command: runtime.command(settings.serveArguments(model: model)),
                environment: runtime.environment.merging(settings.environment) { $1 }
                    .merging(["SPLASH_APP_PID": String(getpid())]) { $1 },
                log: log,
                exited: { [weak self] in self?.exited($0) }
            )
        } catch {
            phase = .failed("Splash couldn’t start it: \(error.localizedDescription)")
            return
        }
        tail = LogTail(log) { [weak self] in self?.received($0) }
        let pid = child!.pid
        record.keep(child!)
        watch = Task { [weak self] in await self?.follow(pid) }
    }

    /// Fails a start before it runs anything, saying why in the log too.
    private func refuse(_ message: String) {
        try? Data((message + "\n").utf8).write(to: log)
        phase = .failed(message)
    }

    private func received(_ line: String) {
        // Why a start fails; once serving, error lines are about requests.
        if phase == .starting, firstError == nil { firstError = errorMessage(line) }
        if phase == .starting, let next = Step(line) {
            step = next
            stepSince = .now
        }
        if phase == .ready, let entry = RequestEntry(line) {
            requestLog.append(entry)
            if requestLog.count > 1000 { requestLog.removeFirst(requestLog.count - 1000) }
        }
    }

    private func exited(_ status: Int32) {
        tail?.finish()
        tail = nil
        child = nil
        watch?.cancel()
        record.forget()
        if let queued {
            self.queued = nil
            launch(queued.model, settings: queued.settings)
        } else if stopRequested {
            phase = .stopped
        } else if phase == .starting, let firstError, let model {
            phase = .failed(Self.memoryProblem(firstError, settings: settings) ?? plainProblem(firstError, model: model))
        } else {
            phase = .failed("It stopped unexpectedly. The log has the details.")
        }
    }

    /// A start the engine refuses for want of memory, in the words of the launch options: the
    /// model, with its reserves, needs more than the budget, the memory limit set or this Mac's own;
    /// or the context length set is more than the memory left, or the SSD cache, holds.
    nonisolated static func memoryProblem(_ message: String, settings: ServeSettings) -> String? {
        if let match = message.firstMatch(of: /require (\d+) bytes but the Metal memory budget is (\d+) bytes/),
           let need = Int64(match.1), let budget = Int64(match.2) {
            let size = { (bytes: Int64) in
                "\((Double(bytes) / Double(1 << 30)).formatted(.number.precision(.fractionLength(0...1)))) GB"
            }
            if let limit = settings.memoryLimit, Int64(limit) << 30 == budget {
                return "It needs \(size(need)) of memory but the memory limit is \(size(budget)). Raise the memory limit in Launch Options."
            }
            return "It needs \(size(need)) of memory but this Mac can give it \(size(budget)). Choose a smaller model."
        }
        if let match = message.firstMatch(of: /exceeds the (\d+) tokens the model and (.+) allow/), let most = Int(match.1) {
            let tokens = most >= 1024 ? "\(most / 1024)K" : "\(most)"
            if match.2.contains("--max-memory") {
                return "Its context fits up to \(tokens) tokens within the memory limit. Choose a shorter context length, or raise the memory limit."
            }
            if match.2.contains("SSD cache") {
                return "Its context fits up to \(tokens) tokens in the SSD cache. Choose a shorter context length, or a larger SSD cache."
            }
            return "Its context fits up to \(tokens) tokens on this Mac. Choose a shorter context length, or Auto."
        }
        return nil
    }

    /// Waits until this start's server answers, then follows its status. The port's lock names the
    /// process serving it, and the server answers only once ready.
    private func follow(_ pid: pid_t) async {
        let health = settings.url.appending(path: "health")
        while !Task.isCancelled, phase == .starting {
            if runtime.serveLockOwner(port: settings.port) == pid,
               let (_, response) = try? await URLSession.shared.data(from: health),
               (response as? HTTPURLResponse)?.statusCode == 200,
               !Task.isCancelled, phase == .starting {
                // What the model takes and holds, known as it shows as ready.
                let served = await servedModel()
                guard !Task.isCancelled, phase == .starting else { break }
                acceptsImages = served?.input_modalities?.contains("image") ?? false
                reasoning = served?.reasoning
                context = served?.context_length
                phase = .ready
                break
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        // Then its status, every 2 s while it serves.
        while !Task.isCancelled, phase == .ready {
            if let report = await status(), phase == .ready {
                asleep = report.asleep
                recovering = report.recovering
                activity = report.activity
                if let stopped = report.stopped { phase = .failed(stopped) }
            }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    /// What the server's status says: the work it has done, whether it has given the weights back
    /// while idle, whether it is starting a failed engine again, and, once it serves nothing more,
    /// why.
    struct Report: Equatable {
        var activity: Activity
        var asleep: Bool
        var recovering = false
        var stopped: String?
    }

    /// The report of `/status`'s body, if it is one.
    nonisolated static func report(_ data: Data) -> Report? {
        guard let status = try? JSONDecoder().decode(Status.self, from: data) else { return nil }
        let metrics = status.metrics
        return Report(
            activity: Activity(
                served: status.requests?.completed ?? 0,
                running: status.active_requests?.count ?? 0,
                generation: metrics?.decode_tokens_per_second.flatMap { $0 > 0 ? $0 : nil },
                reading: metrics?.prefill_tokens_per_second.flatMap { $0 > 0 ? $0 : nil },
                firstToken: metrics?.ttft_ms?.p50.flatMap { $0 > 0 ? .milliseconds($0) : nil },
                generated: metrics?.decode_output_tokens ?? 0,
                read: metrics?.prefill_input_tokens ?? 0,
                cached: status.cache?.reused_tokens ?? 0,
                memory: status.memory_actual?.current_bytes
            ),
            asleep: status.weights?.released ?? false,
            recovering: status.transport?.recovering ?? false,
            // The engine failed again and again; the server stays up but serves nothing.
            stopped: status.transport?.stopped == true ? status.transport?.error ?? "It stopped after failing repeatedly." : nil
        )
    }

    private struct Status: Decodable {
        struct Weights: Decodable { let released: Bool? }
        struct Transport: Decodable { let recovering: Bool?; let stopped: Bool?; let error: String? }
        struct Requests: Decodable { let completed: Int? }
        struct Request: Decodable {}
        struct Metrics: Decodable {
            struct Percentiles: Decodable { let p50: Double? }
            let decode_tokens_per_second: Double?
            let prefill_tokens_per_second: Double?
            let ttft_ms: Percentiles?
            let decode_output_tokens: Int?
            let prefill_input_tokens: Int?
        }
        struct Cache: Decodable { let reused_tokens: Int? }
        struct Memory: Decodable { let current_bytes: Int64? }
        let weights: Weights?
        let transport: Transport?
        let requests: Requests?
        let active_requests: [Request]?
        let metrics: Metrics?
        let cache: Cache?
        let memory_actual: Memory?
    }

    private struct ServedModel: Decodable {
        /// Text, and images and PDFs with vision.
        let input_modalities: [String]?
        let reasoning: Reasoning?
        let context_length: Int?
    }

    private func servedModel() async -> ServedModel? {
        struct Models: Decodable { let data: [ServedModel] }
        guard let (data, _) = try? await URLSession.shared.data(for: settings.request("v1/models")),
              let models = try? JSONDecoder().decode(Models.self, from: data)
        else { return nil }
        return models.data.first
    }

    private func status() async -> Report? {
        var request = settings.request("status")
        request.timeoutInterval = 2
        guard let (data, _) = try? await URLSession.shared.data(for: request) else { return nil }
        return Self.report(data)
    }
}

/// A request the server finished, as its log line gives it: when, how it ended, and, for one that
/// answered, the tokens it read, took from the cache and wrote, how soon it began and how fast it
/// went; for one that failed, why.
struct RequestEntry: Equatable {
    enum Outcome: Equatable { case done, cancelled, error }

    let time: String
    let outcome: Outcome
    /// The line itself.
    let raw: String
    var input: Int?
    var cached: Int?
    var output: Int?
    /// Seconds to the first token, and tokens a second.
    var firstToken: Double?
    var rate: Double?
    var message: String?

    /// The request `line` of the server's log finishes, if it finishes one.
    init?(_ line: String) {
        guard let match = line.wholeMatch(of: /(\d\d:\d\d:\d\d) (Done|Cancelled|Error) · (.*)/) else { return nil }
        time = String(match.1)
        outcome = match.2 == "Done" ? .done : match.2 == "Cancelled" ? .cancelled : .error
        raw = line
        guard outcome != .error else {
            message = String(match.3)
            return
        }
        for part in match.3.components(separatedBy: " · ") {
            if let count = part.wholeMatch(of: /(input|cached|output) ([\d,]+)/) {
                let tokens = Int(count.2.replacing(",", with: ""))
                switch count.1 {
                case "input": input = tokens
                case "cached": cached = tokens
                default: output = tokens
                }
            } else if let seconds = part.wholeMatch(of: /TTFT ([\d.]+)s/) {
                firstToken = Double(seconds.1)
            } else if let speed = part.wholeMatch(of: /([\d.]+) tok\/s/) {
                rate = Double(speed.1)
            }
        }
    }
}
