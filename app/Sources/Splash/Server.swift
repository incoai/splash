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

    /// What a start does, as the server's serve record gives it: the server starts, reads the
    /// model's weights into memory, then runs the model once at full size so that its first answer
    /// comes quickly; the first start after an update also prepares the GPU's programs then, which
    /// can take a minute.
    enum Step: String {
        case starting = "Starting"
        case loading = "Loading weights"
        case warming = "Warming up"

        /// The step of a start at `phase`; nil once the start has ended.
        init?(_ phase: ServeRecord.Phase) {
            switch phase {
            case .starting: self = .starting
            case .loading: self = .loading
            case .warming: self = .warming
            case .ready, .failed: return nil
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

    private(set) var phase = Phase.stopped {
        didSet { phaseChanged() }
    }
    /// Called each time `phase` is set.
    @ObservationIgnored var phaseChanged: () -> Void = {}
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

    /// What the server is at, as the window shows it: its phase and the model a stop makes way
    /// for, the model, what a ready one does, and what that model takes and how it thinks. A
    /// change of it changes the window as one (`MainView`); a start's step changes words alone.
    struct Stage: Equatable {
        var phase: Phase
        var next: String?
        var model: String?
        var serving: Serving
        var acceptsImages: Bool
        var reasoning: Reasoning?
    }

    var stage: Stage {
        Stage(phase: phase, next: next, model: model, serving: serving, acceptsImages: acceptsImages, reasoning: reasoning)
    }

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

    /// How long a stop gives the server before its process group is killed: the longest its own
    /// stop of the engine takes.
    nonisolated static let killAfter = Supervision.stop * 2

    private let runtime: Runtime
    /// Its log, which each start begins again.
    let log: URL
    private let killAfter: Duration
    private var child: ChildProcess?
    /// The process of the server starting or serving, as its serve record names it.
    private var pid: pid_t?
    /// SIGTERM, which a running server takes for Ctrl+C and one already stopping ignores, where a
    /// second Ctrl+C would stop its engine at once.
    let record: ChildRecord
    private var stopRequested = false
    private var watch: Task<Void, Never>?

    init(runtime: Runtime, log: URL, defaults: UserDefaults, killAfter: Duration = Server.killAfter) {
        self.runtime = runtime
        self.log = log
        self.killAfter = killAfter
        record = ChildRecord(defaults: defaults, key: Preference.serverPID, stop: SIGTERM, killAfter: killAfter)
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

    /// Stops the model, and drops a switch to another one: SIGINT, as Ctrl+C sends, on which the
    /// server stops its engine, its flush included (`ChildProcess.stop`).
    func stop() {
        queued = nil
        guard let child, phase != .stopping else { return }
        stopRequested = true
        phase = .stopping
        watch?.cancel()
        child.stop(SIGINT, killAfter: killAfter)
    }

    /// Clears a failure the user has read, stopping a server that still runs but serves nothing.
    func dismissFailure() {
        guard case .failed = phase else { return }
        if child != nil { stop() } else { phase = .stopped }
    }

    /// Stops the server an app started before it quit or crashed, if it still runs, and waits
    /// until it has given its port and memory back. It stops by itself once that app has ended
    /// (`SPLASH_APP_PID`), and is left to finish; this stops one that has not yet. An app before
    /// 1.6 kept its server's ID alone, with its port: the process that still holds that port's
    /// lock under the ID is that server, which nothing stops by itself.
    func stopLeftover() async {
        let port = record.defaults.integer(forKey: Preference.olderServerPort)
        await record.stopLeftover { pid in port > 0 && runtime.serveLockHolder(port: port) == pid }
        record.defaults.removeObject(forKey: Preference.olderServerPort)
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
        stopRequested = false
        resetLog(log)
        if portInUse(settings.port) {
            return refuse(Self.portTaken(settings.port))
        }
        since = .now
        step = .starting
        stepSince = .now
        phase = .starting
        do {
            // It stops as SIGTERM stops it once this app has ended, a crash too, giving its memory
            // back then rather than at the app's next launch.
            child = try record.start(
                runtime.command(settings.serveArguments(model: model)),
                environment: runtime.environment.merging(settings.environment) { $1 }
                    .merging(["SPLASH_APP_PID": String(getpid())]) { $1 },
                log: log,
                line: { [weak self] in self?.received($0) },
                exited: { [weak self] in self?.exited($0) }
            )
        } catch {
            phase = .failed("Splash couldn’t start it: \(error.localizedDescription)")
            return
        }
        let pid = child!.pid
        self.pid = pid
        watch = Task { [weak self] in await self?.follow(pid) }
    }

    /// Fails a start before it runs anything, saying why in the log too.
    private func refuse(_ message: String) {
        try? Data((message + "\n").utf8).write(to: log)
        phase = .failed(message)
    }

    private func received(_ line: String) {
        if phase == .ready, let entry = RequestEntry(line) {
            requestLog.append(entry)
            if requestLog.count > 1000 { requestLog.removeFirst(requestLog.count - 1000) }
        }
    }

    private func exited(_ status: Int32) {
        child = nil
        watch?.cancel()
        if let queued {
            self.queued = nil
            launch(queued.model, settings: queued.settings)
        } else if stopRequested {
            phase = .stopped
        } else if phase == .starting, let pid {
            phase = .failed(Self.startProblem(record: runtime.serveRecord(port: settings.port), pid: pid,
                                              port: settings.port, holder: runtime.serveLockHolder(port: settings.port)))
        } else {
            phase = .failed("It stopped unexpectedly. The log has the details.")
        }
    }

    /// Why a start by process `pid` on `port` failed, as this start's serve record says; that another
    /// Splash server serves the port, where `holder`, the process holding the port's lock, is not
    /// this start's: the launcher refuses it and leaves that server's record; else that it stopped,
    /// where it recorded nothing: it crashed or was killed.
    nonisolated static func startProblem(record: ServeRecord?, pid: pid_t, port: Int, holder: pid_t?) -> String {
        if let record, record.pid == pid, let failure = record.failure {
            return problem(failure, port: port)
        }
        if let holder, holder != pid {
            return "Another Splash server is using port \(port). Stop it, or choose another port in Launch Options › Advanced."
        }
        return "It stopped unexpectedly. The log has the details."
    }

    /// That another app holds `port`.
    nonisolated static func portTaken(_ port: Int) -> String {
        "Port \(port) is in use by another app. Quit that app, or choose another port in Launch Options › Advanced."
    }

    /// A start failure on `port` in the words of the launch options: the model, with its reserves,
    /// needs more memory than the limit set or this Mac gives; the context length set is more than
    /// the memory left, or the SSD cache, holds; other apps hold the memory, GPU memory or port it
    /// needs; an update holds Splash; its files could not be prepared, or Splash cannot serve them;
    /// a failure of no class, which the log tells more of; else the failure's own message.
    nonisolated static func problem(_ failure: ServeRecord.Failure, port: Int) -> String {
        switch (failure.kind, failure.bound) {
        case ("memory_budget", "max_memory"):
            return "It needs \(Format.memory(failure.required_bytes ?? 0)) of memory but the memory limit is "
                + "\(Format.memory(failure.budget_bytes ?? 0)). Raise the memory limit in Launch Options."
        case ("memory_budget", "device_memory"):
            return "It needs \(Format.memory(failure.required_bytes ?? 0)) of memory but this Mac can give it "
                + "\(Format.memory(failure.budget_bytes ?? 0)). Choose a smaller model."
        case ("context_budget", let bound?):
            let tokens = ContextLength.figure(failure.plan_context_tokens ?? 0)
            switch bound {
            case "max_memory":
                return "Its context fits up to \(tokens) tokens within the memory limit. Choose a shorter context length, or raise the memory limit."
            case "disk":
                return "Its context fits up to \(tokens) tokens in the SSD cache. Choose a shorter context length, or a larger SSD cache."
            default:
                return "Its context fits up to \(tokens) tokens on this Mac. Choose a shorter context length, or Auto."
            }
        case ("host_memory", _):
            return "Other apps are using the memory it needs. Quit some, then start it again."
        case ("command_memory", _):
            return "Another app is using the GPU memory this model needs, such as another Splash server. Quit it, then start again."
        case ("port", _):
            return portTaken(port)
        case ("upgrade", _):
            return "Splash is being updated. Wait for the update to finish, then start it again."
        case ("install", _):
            return "Splash couldn’t prepare its files: \(failure.message)"
        case ("model", _):
            return "Splash can’t serve this model: \(failure.message)"
        case ("internal", _):
            return "It stopped while starting: \(failure.message). The log has the details."
        default:
            return failure.message
        }
    }

    /// Follows this start's server through its start, as its serve record gives it, until it
    /// answers, then follows its status.
    private func follow(_ pid: pid_t) async {
        while !Task.isCancelled, phase == .starting {
            if let record = runtime.serveRecord(port: settings.port), record.pid == pid {
                if let next = Step(record.phase), next != step {
                    step = next
                    stepSince = .now
                } else if record.phase == .ready {
                    // What the model takes and holds, known as it shows as ready.
                    let served = await servedModel()
                    guard !Task.isCancelled, phase == .starting else { break }
                    acceptsImages = served?.input_modalities?.contains("image") ?? false
                    reasoning = served?.reasoning
                    context = served?.context_length
                    phase = .ready
                    break
                }
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        // Then its status, every 2 s while it serves.
        while !Task.isCancelled, phase == .ready {
            if let report = await status(), phase == .ready {
                asleep = report.asleep
                recovering = report.recovering
                activity = report.activity
                // A relaunched engine may serve less context than the first did.
                if let served = report.context { context = served }
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
        /// The tokens of context the server serves now.
        var context: Int?
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
            // The engine failed in a way restarting cannot help; the server stays up but serves
            // nothing.
            stopped: status.transport?.stopped == true ? status.transport?.error ?? "Its engine stopped." : nil,
            context: status.capabilities?.max_context_tokens
        )
    }

    private struct Status: Decodable {
        struct Capabilities: Decodable { let max_context_tokens: Int? }
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
        let capabilities: Capabilities?
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
