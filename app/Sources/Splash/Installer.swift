import Darwin
import Foundation
import Observation

/// Downloads and installs models, one at a time, with the installer `splash serve` runs before it
/// serves. A model can download while another one serves.
@MainActor @Observable
final class Installer {
    /// What the installer reports while it downloads one repository.
    struct Progress: Equatable {
        let repo: String
        let done: Int64
        let total: Int64

        var fraction: Double { total > 0 ? Double(done) / Double(total) : 0 }

        /// The progress a line of the installer's output reports, if it is a download event.
        init?(line: String) {
            struct Event: Decodable {
                let event: String
                let repo: String
                let done: Int64
                let total: Int64
            }
            guard line.hasPrefix("{"),
                  let event = try? JSONDecoder().decode(Event.self, from: Data(line.utf8)),
                  event.event == "download"
            else { return nil }
            (repo, done, total) = (event.repo, event.done, event.total)
        }
    }

    /// The model installing.
    private(set) var model: String?
    /// The download of the model's own files. A model's draft downloads first; it is small, and
    /// a bar that filled for it and started over would mislead, so it shows as preparation.
    private(set) var progress: Progress?
    /// The last installation that failed: its model and why.
    private(set) var failure: (model: String, message: String)?

    /// Gets each model once it is installed.
    @ObservationIgnored var installed: (String) -> Void = { _ in }

    static let log = URL.libraryDirectory.appending(path: "Logs/Splash/install.log")

    private let runtime: Runtime
    private var child: ChildProcess?
    private let record = ChildRecord(key: Preference.installerPID)
    private var tail: LogTail?
    private var cancelled = false
    private var firstError: String?
    private var activity: NSObjectProtocol?
    /// When and at how many bytes the model's own download was first reported.
    private var began: (date: Date, done: Int64)?

    init(runtime: Runtime) {
        self.runtime = runtime
    }

    var running: Bool { child != nil }

    /// About how long the download has left, once it has run for a few seconds.
    var remaining: Duration? {
        guard let progress, let began, progress.done > began.done else { return nil }
        let elapsed = Date.now.timeIntervalSince(began.date)
        guard elapsed >= 5 else { return nil }
        return .seconds(Double(progress.total - progress.done) * elapsed / Double(progress.done - began.done))
    }

    func install(_ model: String) {
        guard child == nil else { return }
        self.model = model
        progress = nil
        began = nil
        failure = nil
        firstError = nil
        cancelled = false
        resetLog(Self.log)
        do {
            child = try ChildProcess(
                command: runtime.installCommand(model),
                environment: runtime.environment,
                log: Self.log,
                exited: { [weak self] in self?.exited($0) }
            )
        } catch {
            failure = (model, "The download couldn’t start: \(error.localizedDescription)")
            self.model = nil
            return
        }
        record.keep(child!)
        tail = LogTail(Self.log) { [weak self] in self?.received($0) }
        // A download takes long; the Mac must not sleep through it.
        activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled], reason: "Downloading \(model)")
    }

    /// Stops the download: SIGINT, as Ctrl+C sends, then SIGKILL if it is still running 5 s later.
    /// The installer removes what a killed download left behind the next time it downloads.
    func cancel() {
        guard let child else { return }
        cancelled = true
        child.signal(SIGINT)
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(5))
            if self?.child === child { child.signal(SIGKILL) }
        }
    }

    /// Clears a failure the user has read.
    func dismissFailure() {
        failure = nil
    }

    /// Stops the download this app started before it quit or crashed, if it still runs; its record
    /// stays until then, for the next launch should this one end first.
    func stopLeftover() async {
        await record.stopLeftover()
    }

    private func received(_ line: String) {
        guard let event = Progress(line: line) else {
            if firstError == nil { firstError = errorMessage(line) }
            return
        }
        if event.done == 0 { checkSpace(event) }
        guard let model, event.repo == model.split(separator: ":")[0] else { return }
        if began == nil { began = (.now, event.done) }
        progress = event
    }

    /// Stops a download the disk has no room for, before it fills the disk. macOS frees purgeable
    /// space as it is needed, so what counts is the capacity for important use.
    private func checkSpace(_ event: Progress) {
        let cache = URL.homeDirectory.appending(path: ".cache")
        guard let available = try? cache.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage,
            event.total > available, let model
        else { return }
        cancel()
        failure = (model, "Not enough disk space: the download needs \(Format.size(event.total)), "
            + "and \(Format.size(available)) is available.")
    }

    private func exited(_ status: Int32) {
        tail?.finish()
        tail = nil
        child = nil
        record.forget()
        activity.map(ProcessInfo.processInfo.endActivity)
        activity = nil
        guard let model else { return }
        self.model = nil
        progress = nil
        if status == 0 {
            installed(model)
        } else if !cancelled {
            failure = (model, firstError.map { plainProblem($0, model: model) }
                ?? "The download stopped unexpectedly. The log has the details.")
        }
    }
}
