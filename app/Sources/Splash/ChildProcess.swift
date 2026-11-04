import Darwin
import Foundation

/// A program run as a terminal runs a foreground job: in a process group of its own, so a signal
/// reaches it and every program it starts (the installer, the engine), the way Ctrl+C does. Its
/// output goes to a log file rather than a pipe, so it never waits on this app, and keeps working
/// if the app quits or crashes. Every program the app runs but the drawer's shells, which need a
/// terminal of their own, runs as one.
final class ChildProcess: @unchecked Sendable {
    let pid: pid_t
    /// When it started (`processStart`), which tells it apart from a later process given its ID.
    let started: Int?
    private let exitSource: DispatchSourceProcess
    /// Main queue only: whether it has ended, and whether it was asked to stop.
    private var ended = false
    private var stopping = false

    /// Starts `command` with `environment`, standard input from /dev/null and standard output and
    /// error appended to `log`. `exited` gets the exit status, as a shell reports it, on the main
    /// queue once the program has ended.
    init(
        command: [String],
        environment: [String: String],
        log: URL,
        exited: @escaping @MainActor (Int32) -> Void
    ) throws {
        let output = open(log.path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0o644)
        guard output >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        defer { close(output) }

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_adddup2(&actions, output, 1)
        posix_spawn_file_actions_adddup2(&actions, output, 2)

        var attributes: posix_spawnattr_t?
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        // Its own process group; no descriptor but 0-2 inherited; default signal handling and an
        // empty signal mask, whatever this app set for itself.
        posix_spawnattr_setflags(
            &attributes,
            Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGDEF
                | POSIX_SPAWN_SETSIGMASK))
        posix_spawnattr_setpgroup(&attributes, 0)
        var signals = sigset_t()
        sigemptyset(&signals)
        posix_spawnattr_setsigmask(&attributes, &signals)
        sigfillset(&signals)
        posix_spawnattr_setsigdefault(&attributes, &signals)

        let argv = command.map { strdup($0) } + [nil]
        let envp = environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { (argv + envp).forEach { free($0) } }
        var pid: pid_t = 0
        let status = posix_spawn(&pid, command[0], &actions, &attributes, argv, envp)
        guard status == 0 else { throw POSIXError(.init(rawValue: status) ?? .EIO) }
        self.pid = pid
        started = processStart(pid)

        exitSource = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit, queue: .main)
        exitSource.setEventHandler { [self, exitSource] in
            exitSource.cancel()
            ended = true
            // The event can come just before the program can be reaped, and comes once: wait for
            // it, which takes no time now that it has ended.
            var status: Int32 = 0
            var reaped: pid_t
            repeat { reaped = waitpid(pid, &status, 0) } while reaped == -1 && errno == EINTR
            let code = reaped != pid ? -1 : (status & 0x7f) == 0 ? (status >> 8) & 0xff : 128 + (status & 0x7f)
            MainActor.assumeIsolated { exited(code) }
        }
        exitSource.resume()
    }

    /// Stops the program as a terminal stops a job: `signal` to its process group, once, as a
    /// second one could cut what the first lets it finish; SIGKILL to the group if the program
    /// still runs `killAfter` later.
    @MainActor
    func stop(_ signal: Int32, killAfter: Duration) {
        guard !stopping, !ended else { return }
        stopping = true
        killpg(pid, signal)
        Task { @MainActor [self] in
            try? await Task.sleep(for: killAfter)
            if !ended { killpg(pid, SIGKILL) }
        }
    }

    /// Runs `command` to its end, as `init` does: its exit status, as a shell reports it; nil where
    /// it does not start, or runs past `timeout`, which kills it. One whose caller no longer waits
    /// for it is killed too.
    @MainActor
    static func run(_ command: [String], environment: [String: String], log: URL, timeout: Duration) async -> Int32? {
        let (statuses, end) = AsyncStream.makeStream(of: Int32.self)
        guard let child = try? ChildProcess(command: command, environment: environment, log: log, exited: {
            end.yield($0)
            end.finish()
        }) else { return nil }
        let deadline = Task {
            try await Task.sleep(for: timeout)
            child.stop(SIGKILL, killAfter: .zero)
        }
        let status: Int32? = await withTaskCancellationHandler {
            for await status in statuses { return status }
            return nil
        } onCancel: {
            Task { @MainActor in child.stop(SIGKILL, killAfter: .zero) }
        }
        deadline.cancel()
        // The deadline passed only where it was not cancelled.
        return (try? await deadline.value) == nil ? status : nil
    }
}

/// Follows a log file as it grows, one call per complete line, on the main queue.
@MainActor
final class LogTail {
    private let handle: FileHandle
    private let source: DispatchSourceFileSystemObject
    private let line: (String) -> Void
    private var pending = Data()

    init?(_ url: URL, line: @escaping (String) -> Void) {
        let descriptor = open(url.path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else { return nil }
        handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        self.line = line
        source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: descriptor, eventMask: [.extend, .write], queue: .main)
        source.setEventHandler { [weak self] in MainActor.assumeIsolated { self?.read() } }
        // The source watches the descriptor until it is cancelled, so only then is it closed.
        source.setCancelHandler { close(descriptor) }
        source.resume()
    }

    /// Reads what the file holds now, the last line too if it lacks its newline, and stops.
    func finish() {
        read()
        if !pending.isEmpty { line(String(decoding: pending, as: UTF8.self)) }
        pending.removeAll()
        source.cancel()
    }

    private func read() {
        guard let data = try? handle.readToEnd(), !data.isEmpty else { return }
        pending.append(data)
        while let end = pending.firstIndex(of: 0x0A) {
            let text = String(decoding: pending[pending.startIndex..<end], as: UTF8.self)
            pending.removeSubrange(pending.startIndex...end)
            if !text.isEmpty { line(text) }
        }
    }
}

/// The engine's stop budget (runtime/Supervision.hpp `kStop`, which
/// dev/tests/engine/protocol_golden.txt pins): from a stop signal to the engine's exit, its flush
/// included. A server stops its engine with SIGTERM and kills it past this, then waits this long
/// more for it to exit, so a server stopping takes at most twice it.
enum Supervision {
    static let stop = Duration.seconds(15)
}

/// A program this app started, as `defaults` keep it for the next launch should this one end
/// without stopping it: its process ID and when it started. Together they name it whichever copy
/// of the app started it, and never a later process given the same ID.
@MainActor
struct ChildRecord {
    let defaults: UserDefaults
    let key: String
    /// The signal a later launch stops it with, once: a second one could cut what the first lets
    /// it finish.
    let stop: Int32
    /// How long after that signal its process group is killed if it still runs.
    let killAfter: Duration

    /// Starts `command` as `ChildProcess` does and keeps it as the program this record names until
    /// it exits, giving `line` each line of `log` as it is written; `exited` gets its status once
    /// the log has been read to its end.
    func start(_ command: [String], environment: [String: String], log: URL, line: @escaping (String) -> Void,
               exited: @escaping @MainActor (Int32) -> Void) throws -> ChildProcess {
        var tail: LogTail?
        let child = try ChildProcess(command: command, environment: environment, log: log) { status in
            tail?.finish()
            forget()
            exited(status)
        }
        tail = LogTail(log, line: line)
        keep(child)
        return child
    }

    /// Keeps `child` as the program this record names.
    func keep(_ child: ChildProcess) {
        guard let started = child.started else { return forget() }
        defaults.set([Int(child.pid), started], forKey: key)
    }

    func forget() {
        defaults.removeObject(forKey: key)
    }

    /// Stops the program kept, if it still runs, and forgets it once it has ended; one still
    /// running stays kept, for the launch after. A record of the ID alone, as an older app kept
    /// it, is forgotten, and its program left running.
    func stopLeftover() async {
        guard let kept = defaults.array(forKey: key) as? [Int], kept.count == 2,
              let pid = pid_t(exactly: kept[0])
        else { return forget() }
        if processStart(pid) == kept[1] { await stopProcessGroup(pid, with: stop, killAfter: killAfter) }
        if processStart(pid) != kept[1] { forget() }
    }
}

/// When process `pid` started, in microseconds since 1970, or nil where no such process runs.
func processStart(_ pid: pid_t) -> Int? {
    var info = proc_bsdinfo()
    let size = Int32(MemoryLayout<proc_bsdinfo>.size)
    guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else { return nil }
    return Int(info.pbi_start_tvsec) * 1_000_000 + Int(info.pbi_start_tvusec)
}

/// Stops the process group `pid` leads, as `ChildProcess.stop` does, when this app no longer
/// holds it: `signal`, then SIGKILL if it still runs `killAfter` later; returns once its leader
/// has ended, or 10 s after the SIGKILL.
func stopProcessGroup(_ pid: pid_t, with signal: Int32, killAfter: Duration) async {
    killpg(pid, signal)
    let killAt = ContinuousClock.now + killAfter
    var killed = false
    while kill(pid, 0) == 0, ContinuousClock.now < killAt + .seconds(10) {
        if !killed, ContinuousClock.now >= killAt {
            killpg(pid, SIGKILL)
            killed = true
        }
        try? await Task.sleep(for: .milliseconds(100))
    }
}

/// Whether something listens on, or holds, the loopback `port`, as the launcher checks it.
func portInUse(_ port: Int) -> Bool {
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = in_port_t(UInt16(port).bigEndian)
    address.sin_addr.s_addr = inet_addr("127.0.0.1")
    func attempt(_ operation: (Int32, UnsafePointer<sockaddr>, socklen_t) -> Int32) -> Bool {
        let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        defer { close(socket) }
        var reuse: Int32 = 1
        setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        return withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                operation(socket, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
    }
    return !attempt(bind) || attempt(connect)
}

/// Creates `url` empty, with its folder, for a run's log.
func resetLog(_ url: URL) {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: url.path, contents: nil)
}
