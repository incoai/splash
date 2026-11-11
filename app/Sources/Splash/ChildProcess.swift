import Darwin
import Foundation

/// A program run as a terminal runs a foreground job: in a process group of its own, so a signal
/// reaches it and every program it starts (the installer, the engine), the way Ctrl+C does. Its
/// output goes to a log file rather than a pipe, so it never waits on this app, and keeps working
/// if the app quits or crashes; but for output only this app may read, which no file may hold,
/// as the login shell's settings (`Runtime.readShellSettings`). Every program the app runs but the
/// drawer's shells, which need a terminal of their own, runs as one.
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
    convenience init(
        command: [String],
        environment: [String: String],
        log: URL,
        exited: @escaping @MainActor (Int32) -> Void
    ) throws {
        let output = open(log.path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0o644)
        guard output >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        defer { close(output) }
        try self.init(command: command, environment: environment, output: output, exited: exited)
    }

    /// Starts `command` as `init(command:environment:log:exited:)` does, but with standard output
    /// and error to `output`, an open descriptor, as a pipe's writing end, which stays open here.
    init(
        command: [String],
        environment: [String: String],
        output: Int32,
        exited: @escaping @MainActor (Int32) -> Void
    ) throws {
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

/// A log read as it grows, in whole lines, so that no read decodes part of a line or of a
/// character: each read gives the lines it ends and keeps the bytes after its last newline for the
/// next. Reads that start inside the log start after its first newline, where a line does.
struct LogLines {
    private var pending = Data()
    /// Whether the bytes up to the next newline end a line begun before the reads.
    private var midLine: Bool

    /// `inside`: whether the reads start inside the log rather than at its start.
    init(inside: Bool = false) {
        midLine = inside
    }

    /// The whole lines `data` ends, each with its newline.
    mutating func take(_ data: Data) -> Data {
        pending.append(data)
        if midLine {
            guard let end = pending.firstIndex(of: 0x0A) else {
                pending.removeAll()
                return Data()
            }
            pending.removeSubrange(...end)
            midLine = false
        }
        guard let end = pending.lastIndex(of: 0x0A) else { return Data() }
        defer { pending.removeSubrange(...end) }
        return pending[...end]
    }

    /// The bytes after the last newline: a last line that lacks its newline.
    mutating func rest() -> Data {
        defer { pending.removeAll() }
        return pending
    }

    /// The line being written, as far as its characters are whole: the bytes after the last
    /// newline but those of a character still being written. None while the reads are inside the
    /// line they started in.
    var unfinished: Data {
        guard !midLine else { return Data() }
        // A character still being written: the lead byte of a UTF-8 sequence missing some of the
        // continuation bytes after it.
        let bytes = [UInt8](pending)
        guard let lead = bytes.indices.reversed().prefix(4).first(where: { bytes[$0] & 0xC0 != 0x80 }) else { return pending }
        let length = bytes[lead] >= 0xF0 ? 4 : bytes[lead] >= 0xE0 ? 3 : bytes[lead] >= 0xC0 ? 2 : 1
        return bytes.count - lead < length ? Data(bytes[..<lead]) : pending
    }
}

/// Follows a log file as it grows, from `offset`, where the run it follows starts, one call per
/// complete line, on the main queue.
@MainActor
final class LogTail {
    private let handle: FileHandle
    private let source: DispatchSourceFileSystemObject
    private let line: (String) -> Void
    private var lines = LogLines()

    init?(_ url: URL, from offset: UInt64 = 0, line: @escaping (String) -> Void) {
        let descriptor = open(url.path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else { return nil }
        handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        try? handle.seek(toOffset: offset)
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
        let last = lines.rest()
        if !last.isEmpty { line(String(decoding: last, as: UTF8.self)) }
        source.cancel()
    }

    private func read() {
        guard let data = try? handle.readToEnd(), !data.isEmpty else { return }
        for text in lines.take(data).split(separator: 0x0A) { line(String(decoding: text, as: UTF8.self)) }
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
    func start(_ command: [String], environment: [String: String], log: URL, from offset: UInt64 = 0,
               line: @escaping (String) -> Void, exited: @escaping @MainActor (Int32) -> Void) throws -> ChildProcess {
        var tail: LogTail?
        let child = try ChildProcess(command: command, environment: environment, log: log) { status in
            tail?.finish()
            forget()
            exited(status)
        }
        tail = LogTail(log, from: offset, line: line)
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
    /// running stays kept, for the launch after. A record of the ID alone, as an app before 1.6
    /// kept it, gives no start to tell its program from a later process given the ID: that
    /// program is stopped only where `olderApps` shows the ID is still its own, and the record is
    /// forgotten. `stopping` runs as the stop of a program that runs begins, which can take long.
    func stopLeftover(olderApps: (pid_t) -> Bool, stopping: () -> Void = {}) async {
        if let older = defaults.object(forKey: key) as? Int, let pid = pid_t(exactly: older) {
            if olderApps(pid) {
                stopping()
                await stopProcessGroup(pid, with: stop, killAfter: killAfter)
            }
            return forget()
        }
        guard let kept = defaults.array(forKey: key) as? [Int], kept.count == 2,
              let pid = pid_t(exactly: kept[0])
        else { return forget() }
        if processStart(pid) == kept[1] {
            stopping()
            await stopProcessGroup(pid, with: stop, killAfter: killAfter)
        }
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

/// The arguments process `pid` started with, its program first, as the kernel keeps them; nil
/// where no such process of this user runs.
func processArguments(_ pid: pid_t) -> [String]? {
    var name = [CTL_KERN, KERN_ARGMAX]
    var largest: Int32 = 0
    var size = MemoryLayout<Int32>.size
    guard sysctl(&name, u_int(name.count), &largest, &size, nil, 0) == 0 else { return nil }
    name = [CTL_KERN, KERN_PROCARGS2, pid]
    var buffer = [UInt8](repeating: 0, count: Int(largest))
    size = buffer.count
    guard sysctl(&name, u_int(name.count), &buffer, &size, nil, 0) == 0, size > MemoryLayout<Int32>.size
    else { return nil }
    // Their count, then the program's path and the 0s that pad it, then each argument and a 0.
    let count = Int(buffer.withUnsafeBytes { $0.load(as: Int32.self) })
    let arguments = buffer[MemoryLayout<Int32>.size..<size].split(separator: 0, omittingEmptySubsequences: false)
        .dropFirst().drop(while: \.isEmpty).prefix(count)
    guard arguments.count == count else { return nil }
    return arguments.map { String(decoding: $0, as: UTF8.self) }
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

/// Begins a run in `url`'s log, after the runs before it and a blank line, with `heading`, a line
/// that names the run, and returns where the run's own lines start. The log keeps its last `kept`
/// bytes, from the start of a line: 8 MiB, 16 times what the drawer shows at first, keeps earlier
/// starts' failures and is written again in a moment. It is made, with its folder, where there is
/// none. Only its end is read, however long a run grew it: a log past `kept` is written again from
/// its last `kept` bytes, and one within it has the heading added.
@discardableResult
func beginLog(_ url: URL, heading: String, kept: Int = 8 << 20) -> UInt64 {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let file = try? FileHandle(forUpdating: url)
    defer { try? file?.close() }
    let size = (try? file?.seekToEnd()) ?? 0
    let trims = size > UInt64(kept)
    // Its last `kept` bytes, from the start of a line, where it trims; otherwise its last byte,
    // which says whether its last line ended.
    var tail = Data()
    if let file, size > 0 {
        try? file.seek(toOffset: trims ? size - UInt64(kept) : size - 1)
        tail = (try? file.readToEnd()) ?? Data()
        if trims, let newline = tail.firstIndex(of: 0x0A) { tail = Data(tail[tail.index(after: newline)...]) }
    }
    var added = Data()
    if !tail.isEmpty {
        if tail.last != 0x0A { added.append(0x0A) }
        added.append(0x0A)
    }
    added.append(Data((heading + "\n").utf8))
    guard let file, !trims else {
        let log = (trims ? tail : Data()) + added
        try? log.write(to: url, options: .atomic)
        return UInt64(log.count)
    }
    _ = try? file.seekToEnd()
    try? file.write(contentsOf: added)
    return size + UInt64(added.count)
}

/// Adds `line` at the end of `url`'s log.
func appendLog(_ url: URL, _ line: String) {
    guard let handle = try? FileHandle(forWritingTo: url) else { return }
    defer { try? handle.close() }
    _ = try? handle.seekToEnd()
    try? handle.write(contentsOf: Data((line + "\n").utf8))
}
