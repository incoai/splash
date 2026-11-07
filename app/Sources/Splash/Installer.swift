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

    /// A failure the installer reports as it ends: its class, which the app words, and its own
    /// message for the rest; for a disk without room, that names the bytes needed and free.
    struct Failure: Decodable, Equatable {
        /// "not_found", "offline" and "auth" are worded by the app; any other shows the message.
        let kind: String?
        let message: String

        /// The failure a line of the installer's output reports, if it is an error event.
        init?(line: String) {
            struct Event: Decodable {
                let event: String
                let kind: String?
                let message: String
            }
            guard line.hasPrefix("{"),
                  let event = try? JSONDecoder().decode(Event.self, from: Data(line.utf8)),
                  event.event == "error"
            else { return nil }
            (kind, message) = (event.kind, event.message)
        }
    }

    /// A failure installing `model`, in words a user can act on.
    nonisolated static func problem(_ failure: Failure, model: String) -> String {
        let name = model.split(separator: ":")[0]
        switch failure.kind {
        case "not_found":
            return "Hugging Face has no public model named \(name). Check the name, then try again."
        case "offline":
            return "Splash couldn’t reach Hugging Face. Check the internet connection, then try again."
        case "auth":
            return "Hugging Face refused access to \(name). Sign in with an account that can read it (hf auth login) and, "
                + "if the model asks, accept its terms at huggingface.co/\(name), then try again."
        default:
            return failure.message
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

    /// The downloads' log, which each download begins again.
    let log: URL
    private let runtime: Runtime
    private var child: ChildProcess?
    /// A leftover download gets Ctrl+C, which removes its temporary files as it ends.
    let record: ChildRecord
    private var cancelled = false
    private var reported: Failure?
    private var activity: NSObjectProtocol?
    /// When and at how many bytes the model's own download was first reported.
    private var began: (date: Date, done: Int64)?

    init(runtime: Runtime, log: URL, defaults: UserDefaults) {
        self.runtime = runtime
        self.log = log
        record = ChildRecord(defaults: defaults, key: Preference.installerPID, stop: SIGINT, killAfter: .seconds(20))
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
        reported = nil
        cancelled = false
        resetLog(log)
        do {
            child = try record.start(
                runtime.installCommand(model),
                environment: runtime.environment,
                log: log,
                line: { [weak self] in self?.received($0) },
                exited: { [weak self] in self?.exited($0) }
            )
        } catch {
            failure = (model, "The download couldn’t start: \(error.localizedDescription)")
            self.model = nil
            return
        }
        // A download takes long; the Mac must not sleep through it.
        activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled], reason: "Downloading \(model)")
    }

    /// Stops the download: SIGINT, as Ctrl+C sends, then SIGKILL if it is still running 5 s later.
    /// The installer removes what a killed download left behind the next time it downloads.
    func cancel() {
        guard let child else { return }
        cancelled = true
        child.stop(SIGINT, killAfter: .seconds(5))
    }

    /// Clears a failure the user has read.
    func dismissFailure() {
        failure = nil
    }

    /// Stops the download this app started before it quit or crashed, if it still runs; its record
    /// stays until then, for the next launch should this one end first. An app before 1.6 kept its
    /// download's ID alone: the process that still runs the installer under the ID as an app
    /// starts it is that download.
    func stopLeftover() async {
        await record.stopLeftover(olderApps: runtime.runsInstaller)
    }

    private func received(_ line: String) {
        if let reported = Failure(line: line) {
            self.reported = reported
            return
        }
        guard let event = Progress(line: line) else { return }
        guard let model, event.repo == model.split(separator: ":")[0] else { return }
        if began == nil { began = (.now, event.done) }
        progress = event
    }

    private func exited(_ status: Int32) {
        child = nil
        activity.map(ProcessInfo.processInfo.endActivity)
        activity = nil
        guard let model else { return }
        self.model = nil
        progress = nil
        if status == 0 {
            installed(model)
        } else if !cancelled {
            failure = (model, reported.map { Self.problem($0, model: model) }
                ?? "The download stopped unexpectedly. The log has the details.")
        }
    }
}
