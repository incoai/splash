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

    /// What a check for updates found, as the installer reports it last (`prepared`,
    /// install/upstream.py).
    enum Outcome: String, Decodable {
        /// It downloaded what had moved on, which the next start uses.
        case updated
        /// Hugging Face has nothing newer.
        case upToDate = "up_to_date"
        /// It couldn't check: Hugging Face didn't answer, or wasn't asked (HF_HUB_OFFLINE).
        case unreachable
        /// What had moved on couldn't be installed, and the installed version stays.
        case kept

        /// The outcome a line of the installer's output reports, if it is a `prepared` event.
        init?(line: String) {
            struct Event: Decodable {
                let event: String
                let outcome: Outcome
            }
            guard line.hasPrefix("{"),
                  let event = try? JSONDecoder().decode(Event.self, from: Data(line.utf8)),
                  event.event == "prepared"
            else { return nil }
            self = event.outcome
        }
    }

    /// What a check for updates that ended did, which the Server shows until read.
    enum Result: Equatable {
        case checked(Outcome)
    }

    /// What the Server says of `result` for the model `name`: a title, the line under it, and
    /// whether it did what was asked.
    nonisolated static func words(_ result: Result, name: String) -> (title: String, detail: String, done: Bool) {
        switch result {
        case .checked(.updated): ("\(name) is updated", "Its next start uses the new files.", true)
        case .checked(.upToDate): ("\(name) is up to date", "Hugging Face has nothing newer.", true)
        case .checked(.unreachable): ("Splash couldn’t check \(name) for updates", "Hugging Face was not reachable.", false)
        case .checked(.kept): ("Splash couldn’t update \(name)", "It kept the installed version.", false)
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
    /// What the installer does with a model: downloads it, checks an installed one for updates,
    /// which downloads only what has moved on, if anything, or removes an installed one.
    enum Work {
        case download, check, removal

        /// What it does to the model `name`: a check, once it finds something to download,
        /// downloads it.
        func phrase(_ name: String, downloading: Bool) -> String {
            switch self {
            case .download: "Downloading \(name)"
            case .check: downloading ? "Downloading \(name)" : "Checking \(name) for updates"
            case .removal: "Deleting \(name)"
            }
        }
    }
    /// What the installation running does, or else what the last one did.
    private(set) var work = Work.download
    /// The last check for updates that ended, as the installer reported it: its model, and what
    /// it found.
    private(set) var checked: (model: String, result: Result)?

    /// Gets each model once it is downloaded.
    @ObservationIgnored var installed: (String) -> Void = { _ in }
    /// Gets each model once a check or a deletion of it has changed its files.
    @ObservationIgnored var changed: (String) -> Void = { _ in }
    /// Gets each model once it is removed.
    @ObservationIgnored var removed: (String) -> Void = { _ in }

    /// The downloads' log, which each download begins again.
    let log: URL
    private let runtime: Runtime
    private var child: ChildProcess?
    /// A leftover download gets Ctrl+C, which removes its temporary files as it ends.
    let record: ChildRecord
    private var cancelled = false
    private var reported: Failure?
    /// What the check running found, as reported.
    private var outcome: Outcome?
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

    /// Downloads `model`, or checks it for updates (`Work`).
    func install(_ model: String, _ work: Work = .download) {
        run(model, work)
    }

    /// Removes `model`, an installed one, with the files of it that no other installation uses
    /// (`install/models.py remove`), which a server serving it refuses.
    func remove(_ model: String) {
        run(model, .removal)
    }

    private func run(_ model: String, _ work: Work) {
        guard child == nil else { return }
        self.model = model
        self.work = work
        progress = nil
        began = nil
        failure = nil
        checked = nil
        reported = nil
        outcome = nil
        cancelled = false
        resetLog(log)
        do {
            child = try record.start(
                work == .removal ? runtime.removeCommand(model) : runtime.installCommand(model),
                environment: runtime.environment,
                log: log,
                line: { [weak self] in self?.received($0) },
                exited: { [weak self] in self?.exited($0) }
            )
        } catch {
            failure = (model, "Splash couldn’t start: \(error.localizedDescription)")
            self.model = nil
            return
        }
        guard work != .removal else { return }
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

    /// Clears what a check found, once read.
    func dismissChecked() {
        checked = nil
    }

    /// Stops the download this app started before it quit or crashed, if it still runs; its record
    /// stays until then, for the next launch should this one end first. An app before 1.6 kept its
    /// download's ID alone: the process that still runs the installer under the ID as an app
    /// starts it is that download.
    func stopLeftover(stopping: () -> Void = {}) async {
        await record.stopLeftover(olderApps: runtime.runsInstaller, stopping: stopping)
    }

    private func received(_ line: String) {
        if let reported = Failure(line: line) {
            self.reported = reported
            return
        }
        if let outcome = Outcome(line: line) {
            self.outcome = outcome
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
            switch work {
            case .download: installed(model)
            case .check:
                checked = outcome.map { (model, .checked($0)) }
                changed(model)
            case .removal: changed(model)
            }
        } else if !cancelled {
            failure = (model, reported.map { Self.problem($0, model: model) }
                ?? "It stopped unexpectedly. The log has the details.")
        }
    }
}
