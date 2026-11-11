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

    /// A change of the launch options for a start that failed for want of memory (`remedy`).
    enum Remedy: Equatable {
        /// The memory limit, as `--max-memory` takes it; nil for none, Auto.
        case memoryLimit(String?)
        /// The context length on Auto.
        case autoContext
        /// Model files on this Mac that started with a vision projector or a draft chosen for them,
        /// either of which may be what Splash couldn't prepare or serve: the start without them,
        /// each saying whether it was passed.
        case without(projector: Bool, draft: Bool)
        /// Model files on this Mac Splash can't prepare or serve, which no start would: off the list.
        case removeFromList

        /// Its button: the start with it, or taking the files off the list.
        var title: String {
            switch self {
            case .memoryLimit(let limit?): "Start with \(limit.dropLast()) GB"
            case .memoryLimit(nil): "Start Without a Limit"
            case .autoContext: "Start with Auto Context"
            case .without(projector: true, draft: true): "Start Without Projector and Draft"
            case .without(projector: true, draft: false): "Start Without Projector"
            case .without(projector: false, _): "Start Without Draft"
            case .removeFromList: "Remove from List"
            }
        }
    }

    /// What a start does, as the server's serve record gives it: the server starts, reads the
    /// model's weights into memory, then runs the model once at full size so that its first answer
    /// comes quickly; the first start after an update also prepares the GPU's programs then, which
    /// can take a couple of minutes.
    enum Step: String {
        case starting = "Starting"
        case loading = "Loading weights"
        case warming = "Warming up"

        /// Why the step can take long, which the window says while it runs; nil for a step that
        /// doesn't.
        var note: String? {
            self == .warming ? "The first start after an update prepares the GPU and can take a couple of minutes." : nil
        }

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
    /// While warming: how far the engine's warm-up is, as the serve record last gave it; nil until
    /// it says.
    private(set) var warmup: ServeRecord.Warmup?
    /// While starting: its step as the window words it, with how far the warm-up is once it says.
    var stepPhrase: String {
        guard step == .warming, let warmup else { return "\(step.rawValue)…" }
        return "\(step.rawValue) · \(warmup.steps_done) of \(warmup.steps)"
    }
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
    /// Once ready: whether the model takes images and PDFs, which come together (`input_modalities`).
    private(set) var acceptsImages = false
    /// Once ready: what a request to it may carry, as `/v1/models` says.
    private(set) var limits: Limits?

    /// What a request may carry (`/v1/models`' `limits`): its bytes, its images, a PDF's pages
    /// counted among them, an image's bytes, and a PDF's bytes, which a request's PDFs share, and
    /// pages.
    struct Limits: Decodable, Equatable, Sendable {
        let request_bytes: Int
        let images: Int
        let image_bytes: Int
        let pdf_bytes: Int
        let pdf_pages: Int
    }
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

    /// Once ready: the requests the server has taken, a row each, in the order they came
    /// (`ServedRequest`).
    private(set) var requests: [ServedRequest] = []

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
    /// Its log: each start goes on under the starts before it, from a line that names it, for the
    /// one that failed to be read after the next one begins (`beginLog`).
    let log: URL
    private let killAfter: Duration
    private var child: ChildProcess?
    /// The process of the server this object started last, as its serve record names it: an
    /// exit it did not ask for is worded from that record.
    private var pid: pid_t = 0
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

    /// Whether neither the window nor the menu bar's panel shows the server's status: it is asked
    /// every 10 s rather than every 2 s then, for the menu bar's icon alone.
    @ObservationIgnored var unwatched = false

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

    /// A change of the launch options that would let the start that failed start: the failure says
    /// what the model needs, a memory limit that holds it or the context length on Auto; nil
    /// otherwise.
    private(set) var remedy: Remedy?

    /// Clears a failure the user has read, and what it offered.
    func dismissFailure() {
        guard case .failed = phase else { return }
        remedy = nil
        phase = .stopped
    }

    /// Stops the server an app started before it quit or crashed, if it still runs, and waits
    /// until it has given its port and memory back. It stops by itself once that app has ended
    /// (`SPLASH_APP_PID`), and is left to finish; this stops one that has not yet. An app before
    /// 1.6 kept its server's ID alone, with its port: the process that still holds that port's
    /// lock under the ID is that server, which nothing stops by itself.
    func stopLeftover(stopping: () -> Void = {}) async {
        let port = record.defaults.integer(forKey: Preference.olderServerPort)
        await record.stopLeftover(olderApps: { pid in port > 0 && runtime.serveLockHolder(port: port) == pid }, stopping: stopping)
        record.defaults.removeObject(forKey: Preference.olderServerPort)
    }

    private func launch(_ model: String, settings: ServeSettings) {
        self.model = model
        self.settings = settings
        asleep = false
        recovering = false
        context = nil
        activity = Activity()
        requests = []
        acceptsImages = false
        limits = nil
        reasoning = nil
        stopRequested = false
        remedy = nil
        let start = beginLog(log, heading: "── Starting \(model), \(Date.now.formatted(date: .abbreviated, time: .standard))")
        if portInUse(settings.port) {
            return refuse(Self.portTaken(settings.port))
        }
        since = .now
        step = .starting
        stepSince = .now
        warmup = nil
        phase = .starting
        do {
            // It stops as SIGTERM stops it once this app has ended, a crash too, giving its memory
            // back then rather than at the app's next launch.
            child = try record.start(
                runtime.command(settings.serveArguments(model: model)),
                environment: runtime.environment.merging(settings.environment) { $1 }
                    .merging(["SPLASH_APP_PID": String(getpid())]) { $1 },
                log: log,
                from: start,
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
        appendLog(log, message)
        phase = .failed(message)
    }

    private func received(_ line: String) {
        if phase == .ready, let entry = RequestEntry(line) {
            requests = ServedRequest.taking(entry, at: .now, into: requests)
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
        } else {
            let record = runtime.serveRecord(port: settings.port)
            phase = .failed(Self.exitProblem(record: record, pid: pid, port: settings.port,
                                             holder: runtime.serveLockHolder(port: settings.port)))
            // A failure that a change of the launch options would let start offers it.
            remedy = record.flatMap { $0.pid == pid ? $0.failure : nil }
                .flatMap { Self.remedy(for: $0, settings: settings, memory: Catalog.memory, local: model?.hasPrefix("/") == true) }
        }
    }

    /// Why process `pid` serving `port` ended unasked, as its serve record says: the failure it
    /// recorded, at its start or once ready; that another Splash server serves the
    /// port, where `holder`, the process holding the port's lock, is not this one: the launcher
    /// refuses it and leaves that server's record; else that it stopped, where it recorded
    /// nothing: it crashed or was killed.
    nonisolated static func exitProblem(record: ServeRecord?, pid: pid_t, port: Int, holder: pid_t?) -> String {
        if let record, record.pid == pid, let failure = record.failure {
            return problem(failure, port: port)
        }
        if let holder, holder != pid {
            return "Another Splash server is using port \(port). Stop it, or choose another port in Launch Options › Advanced."
        }
        return "It stopped unexpectedly. The log has the details."
    }

    /// What the launch options would need for a start that failed with `settings` on a Mac with
    /// `memory` GB to start: a memory limit that holds what the model needs, the least one the
    /// options offer, or none where none does; the context length on Auto, which the server fits to
    /// the memory, where one was set that doesn't fit; for `local` model files Splash couldn't
    /// prepare or can't serve, the start without the projector or draft chosen for them, where it
    /// passed either, else taking them off the list. Other failures it can't tell a change for.
    nonisolated static func remedy(for failure: ServeRecord.Failure, settings: ServeSettings, memory: Int,
                                   local: Bool = false) -> Remedy? {
        switch (failure.kind, failure.bound) {
        case (let kind, _) where local && (kind == "model" || kind == "install"):
            let projector = settings.passedProjector != nil, draft = settings.draft != nil
            return projector || draft ? .without(projector: projector, draft: draft) : .removeFromList
        case ("memory_budget", "max_memory"):
            guard let needed = failure.required_bytes else { return nil }
            let limit = ServeSettings.memoryLimits(memory: memory).first { Int64($0) << 30 >= needed }
            return .memoryLimit(limit.map { "\($0)G" })
        case ("context_budget", _) where settings["--max-context"] != nil:
            return .autoContext
        default:
            return nil
        }
    }

    /// That another app holds `port`.
    nonisolated static func portTaken(_ port: Int) -> String {
        "Port \(port) is in use by another app. Quit that app, or choose another port in Launch Options › Advanced."
    }

    /// A failure on `port` in the words of the launch options: the model, with its reserves,
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
            return "It stopped: \(failure.message). The log has the details."
        default:
            return failure.message
        }
    }

    /// Follows this start's server through its start, as its serve record gives it, until it
    /// answers, then follows its status.
    private func follow(_ pid: pid_t) async {
        while !Task.isCancelled, phase == .starting {
            if let record = runtime.serveRecord(port: settings.port), record.pid == pid {
                // Kept through the ready record, which has none, until the start ends.
                if let counted = record.warmup, counted != warmup { warmup = counted }
                if let next = Step(record.phase), next != step {
                    step = next
                    stepSince = .now
                } else if record.phase == .ready {
                    // What the model takes and holds, known as it shows as ready.
                    let served = await servedModel()
                    guard !Task.isCancelled, phase == .starting else { break }
                    acceptsImages = served?.input_modalities?.contains("image") ?? false
                    limits = served?.limits
                    reasoning = served?.reasoning
                    context = served?.context_length
                    phase = .ready
                    break
                }
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        // Then its status while it serves: every 2 s, or every 10 s while nothing but the menu bar's
        // icon shows it.
        var asked = ContinuousClock.now - .seconds(10)
        while !Task.isCancelled, phase == .ready {
            if !unwatched || ContinuousClock.now - asked >= .seconds(10) {
                asked = .now
                await askStatus()
            }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    /// Asks the server's status and takes what it reports.
    private func askStatus() async {
        guard let report = await status(), phase == .ready else { return }
        asleep = report.asleep
        recovering = report.recovering
        activity = report.activity
        requests = ServedRequest.taking(report.live, taken: report.taken, into: requests)
        // A relaunched engine may serve less context than the first did.
        if let served = report.context { context = served }
    }

    /// What the server's status says: the work it has done, whether it has given the weights back
    /// while idle, and whether it is starting a failed engine again.
    struct Report: Equatable {
        var activity: Activity
        var asleep: Bool
        var recovering = false
        /// The tokens of context the server serves now.
        var context: Int?
        /// The requests the engine works on, as they were at `taken`.
        var live: [LiveRequest] = []
        /// When the engine's requests were as `live` has them, at the earliest: as the status was
        /// asked, less the age of the snapshot the server answered from, as while a long prefill
        /// keeps the engine from answering (`transport.status_age_ms`); nil where it lists none.
        var taken: Date?
    }

    /// The report of `/status`'s body, if it is one, asked `now`.
    nonisolated static func report(_ data: Data, at now: Date = .now) -> Report? {
        guard let status = try? JSONDecoder().decode(Status.self, from: data) else { return nil }
        let metrics = status.metrics
        let taken = now.addingTimeInterval(-(status.transport?.status_age_ms ?? 0) / 1000)
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
            context: status.capabilities?.max_context_tokens,
            // A request without the engine's id has none: no line or later status could be told
            // from it.
            live: (status.active_requests ?? []).compactMap { request in
                request.id.map { id in
                    LiveRequest(id: id, time: LiveRequest.time(of: taken.addingTimeInterval(-(request.age_ms ?? 0) / 1000)),
                                input: request.prompt_tokens ?? 0,
                                stage: LiveRequest.stage(phase: request.phase, read: request.prompt_processed ?? 0,
                                                         written: request.generated_tokens ?? 0))
                }
            },
            taken: status.active_requests == nil ? nil : taken
        )
    }

    private struct Status: Decodable {
        struct Capabilities: Decodable { let max_context_tokens: Int? }
        struct Weights: Decodable { let released: Bool? }
        struct Transport: Decodable {
            let recovering: Bool?
            /// How long before it answered the server took the snapshot it answered from: none
            /// where the engine answered then.
            let status_age_ms: Double?
        }
        struct Requests: Decodable { let completed: Int? }
        struct Request: Decodable {
            let id: Int?
            let phase: String?
            let prompt_tokens: Int?
            let prompt_processed: Int?
            let generated_tokens: Int?
            let age_ms: Double?
        }
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
        let limits: Limits?
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
        let asked = Date.now
        var request = settings.request("status")
        request.timeoutInterval = 2
        guard let (data, _) = try? await URLSession.shared.data(for: request) else { return nil }
        return Self.report(data, at: asked)
    }
}

/// A request the server works on, as its status gives it: when it came, the tokens of its prompt,
/// and what it does now.
struct LiveRequest: Equatable, Identifiable {
    enum Stage: Equatable {
        /// Not begun, and why: queued, or waiting for memory or for a prefix another request reads.
        case waiting(String)
        /// Reading its prompt, so many of its tokens read, those taken from the cache among them.
        case reading(Int)
        /// Answering, so many tokens written.
        case answering(Int)
    }

    let id: Int
    /// When it came, as the status's age of it says, as the server's lines give the time of day.
    let time: String
    let input: Int
    let stage: Stage

    /// What a request in the engine's `phase` does, having read `read` tokens of its prompt and
    /// written `written`.
    nonisolated static func stage(phase: String?, read: Int, written: Int) -> Stage {
        switch phase {
        case "queued": .waiting("Queued")
        case "waiting_resources": .waiting("Waiting for memory")
        case "waiting_prefix": .waiting("Waiting for a shared prefix")
        case "prefill": .reading(read)
        default: .answering(written)
        }
    }

    /// `date`'s time of day as the server's lines give it.
    nonisolated static func time(of date: Date) -> String {
        let parts = Calendar.current.dateComponents([.hour, .minute, .second], from: date)
        return String(format: "%02d:%02d:%02d", parts.hour ?? 0, parts.minute ?? 0, parts.second ?? 0)
    }
}

extension LiveRequest.Stage {
    /// How far on it is: waiting, then reading, then answering, each by its tokens.
    fileprivate var progress: (Int, Int) {
        switch self {
        case .waiting: (0, 0)
        case .reading(let read): (1, read)
        case .answering(let written): (2, written)
        }
    }
}

/// A row of the Server's request list: one for each request the engine took, from the line that
/// started it or the status that had it first, which goes on as each status taken since says, back
/// too, until the line that ends it ends it in its place, or a status taken since no longer has
/// it; and one for each line that names no request, as a request refused before the engine took
/// it.
struct ServedRequest: Equatable, Identifiable {
    /// The engine's id of the request, as its lines and the status name it; below zero for a line
    /// that names none.
    let id: Int
    /// When it came: its first line's time of day, or the time the status first had it come.
    let time: String
    /// The tokens of its prompt, where they are known.
    var input: Int?
    /// How far it has got while it runs.
    var stage: LiveRequest.Stage
    /// When it was so: as the line that said so came, or, at the earliest, as the snapshot of the
    /// status that said so was taken. A status taken no later says nothing of it.
    var seen = Date.distantPast
    /// The line that ended it; nil while it runs, or where it ended without one (`gone`).
    var end: RequestEntry?
    /// Whether a status taken since it was seen running no longer has it: it ended, its line not
    /// come, as the server's console drops lines while 1024 wait. Its line, should it come later,
    /// still ends it.
    var gone = false

    /// Whether it still runs: no line has ended it, and the statuses still have it.
    var running: Bool { end == nil && !gone }

    /// The rows a list keeps, the latest.
    static let kept = 1000

    /// `requests` with `line` taken, as it came `now`: a request's start adds its row, unless the
    /// status had it first, which it moves on to reading where the status had it no further; its
    /// end ends its row, or adds one ended where it was never seen; an end that names no request
    /// adds a row of its own, as nothing else could end it.
    static func taking(_ line: RequestEntry, at now: Date, into requests: [ServedRequest]) -> [ServedRequest] {
        var requests = requests
        let index = line.request.flatMap { id in requests.firstIndex { $0.id == id } }
        switch (line.outcome, index) {
        case (.started, let index?):
            requests[index].input = line.input ?? requests[index].input
            if requests[index].running { requests[index].moveOn(to: .reading(line.cached ?? 0), seen: now) }
        case (.started, nil):
            guard let id = line.request else { return requests }
            requests.append(ServedRequest(id: id, time: line.time, input: line.input, stage: .reading(line.cached ?? 0), seen: now))
        case (_, let index?):
            requests[index].input = line.input ?? requests[index].input
            requests[index].end = line
        case (_, nil):
            let id = line.request ?? min(-1, (requests.map(\.id).min() ?? 0) - 1)
            requests.append(ServedRequest(id: id, time: line.time, input: line.input, stage: .answering(line.output ?? 0),
                                          seen: now, end: line))
        }
        return Array(requests.suffix(kept))
    }

    /// `requests` with a status's `live` ones taken, as they were at `taken`: where each row
    /// running that was last seen before is, back too, as the engine suspends a request for memory
    /// and later reads its prompt again, and that one it no longer has has ended (`gone`); a
    /// request it has first is added, at the time it came. A row seen since, as a line started it
    /// while the status was asked, or one ended, stays as it is. A status that lists no requests
    /// (`taken` nil) says nothing of them.
    static func taking(_ live: [LiveRequest], taken: Date?, into requests: [ServedRequest]) -> [ServedRequest] {
        guard let taken else { return requests }
        var requests = requests
        for index in requests.indices where requests[index].running && requests[index].seen < taken {
            if let request = live.first(where: { $0.id == requests[index].id }) {
                requests[index].input = requests[index].input ?? request.input
                requests[index].stage = request.stage
                requests[index].seen = taken
            } else {
                requests[index].gone = true
            }
        }
        for request in live where !requests.contains(where: { $0.id == request.id }) {
            requests.append(ServedRequest(id: request.id, time: request.time, input: request.input, stage: request.stage,
                                          seen: taken))
        }
        return Array(requests.suffix(kept))
    }

    /// Takes a start's `stage`, as its line came `now`, where the row has it no further on: a
    /// status may have had it further on by the time its line comes.
    private mutating func moveOn(to stage: LiveRequest.Stage, seen now: Date) {
        guard stage.progress >= self.stage.progress else { return }
        self.stage = stage
        seen = now
    }
}

/// A request line of the server's log: when, that the request started or how it ended, and the
/// request it names; for one that started, its prompt and the part of it cached; for one that
/// answered, the tokens it read, took from the cache and wrote, how soon it began and how fast it
/// went; for one that failed, why.
struct RequestEntry: Equatable {
    enum Outcome: Equatable { case started, done, cancelled, error }

    let time: String
    let outcome: Outcome
    /// The line itself.
    let raw: String
    /// The engine's id of the request; nil for one refused before the engine took it.
    var request: Int?
    var input: Int?
    var cached: Int?
    var output: Int?
    /// Seconds to the first token, and tokens a second.
    var firstToken: Double?
    var rate: Double?
    var message: String?

    /// The request `line` of the server's log starts or ends, if it is a request's line.
    init?(_ line: String) {
        guard let match = line.wholeMatch(of: /(\d\d:\d\d:\d\d) (Started|Done|Cancelled|Error) · (.*)/) else { return nil }
        time = String(match.1)
        outcome = switch match.2 {
        case "Started": .started
        case "Done": .done
        case "Cancelled": .cancelled
        default: .error
        }
        raw = line
        var why: [String] = []
        for part in match.3.components(separatedBy: " · ") {
            if let id = part.wholeMatch(of: /request (\d+)/) {
                request = Int(id.1)
            } else if outcome == .error {
                why.append(part)
            } else if let count = part.wholeMatch(of: /(input|cached|output) ([\d,]+)/) {
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
        if outcome == .error { message = why.joined(separator: " · ") }
    }
}
