import Darwin
import Foundation

/// A program run as a terminal runs a foreground job: in a process group of its own, so a signal
/// reaches it and every program it starts (the installer, the engine), the way Ctrl+C does. Its
/// output goes to a log file rather than a pipe, so it never waits on this app, and keeps working
/// if the app quits or crashes.
final class ChildProcess: @unchecked Sendable {
    let pid: pid_t
    /// When it started (`processStart`), which tells it apart from a later process given its ID.
    let started: Int?
    private let exitSource: DispatchSourceProcess

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
        exitSource.setEventHandler { [exitSource] in
            exitSource.cancel()
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

    /// Sends `signal` to the program's process group.
    func signal(_ signal: Int32) {
        killpg(pid, signal)
    }
}

/// A program run to its end for what it says on its error output, as the engine's device check
/// is: its exit status and that output, read as it writes it, so that it never waits on a pipe
/// nobody reads; nil where it does not start. One still running after `timeout` is ended, and so
/// is one whose caller no longer waits for it. The read and the wait for its end hold no thread of
/// Swift's own pool: waitUntilExit there once never returned for a process that had exited.
enum Program {
    static func run(_ command: [String], timeout: Duration) async -> (status: Int32, error: String)? {
        let process = Process(), errors = Pipe()
        process.executableURL = URL(filePath: command[0])
        process.arguments = Array(command.dropFirst())
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = errors
        let ended = AsyncStream<Void> { continuation in process.terminationHandler = { _ in continuation.finish() } }
        guard (try? process.run()) != nil else { return nil }
        let deadline = Task {
            try? await Task.sleep(for: timeout)
            if process.isRunning { process.terminate() }
        }
        defer { deadline.cancel() }
        return await withTaskCancellationHandler {
            let error = String(decoding: await reading(errors.fileHandleForReading), as: UTF8.self)
            await end(ended)
            return (process.terminationStatus, error)
        } onCancel: {
            if process.isRunning { process.terminate() }
        }
    }

    /// Waits for `ended`, the stream a process's termination handler finishes, to finish: in a task
    /// of its own, which the caller's cancellation does not reach. A cancelled task's `for await`
    /// returns at once, before the process has exited, and terminationStatus, read then, raises
    /// "task still running".
    private static func end(_ ended: AsyncStream<Void>) async {
        await Task { for await _ in ended {} }.value
    }

    /// Reads what a program writes to `handle` as it writes it, until the program ends: on a thread
    /// of its own, blocking there, rather than on FileHandle's async bytes, which read every handle
    /// on one actor.
    private static func reading(_ handle: FileHandle) async -> Data {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                var data = Data()
                var chunk = [UInt8](repeating: 0, count: 65536)
                while true {
                    let count = read(handle.fileDescriptor, &chunk, chunk.count)
                    if count < 0 && errno == EINTR { continue }
                    guard count > 0 else { break }
                    data.append(contentsOf: chunk[..<count])
                }
                continuation.resume(returning: data)
            }
        }
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

/// A program this app started, as the defaults keep it for the next launch should this one end
/// without stopping it: its process ID and when it started. Together they name it whichever copy
/// of the app started it, and never a later process given the same ID.
struct ChildRecord {
    let key: String

    /// Keeps `child` as the program this record names.
    func keep(_ child: ChildProcess) {
        guard let started = child.started else { return forget() }
        UserDefaults.standard.set([Int(child.pid), started], forKey: key)
    }

    func forget() {
        UserDefaults.standard.removeObject(forKey: key)
    }

    /// Stops the program kept, if it still runs, and forgets it once it has ended; one still
    /// running stays kept, for the launch after.
    func stopLeftover() async {
        guard let kept = UserDefaults.standard.array(forKey: key) as? [Int], kept.count == 2,
              let pid = pid_t(exactly: kept[0])
        else { return forget() }
        if processStart(pid) == kept[1] { await stopProcessGroup(pid) }
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

/// Stops the process group `pid` leads, as `ChildProcess.signal` would, when this app no longer
/// holds it: SIGINT, then SIGKILL if it still runs 20 s later; returns once it has ended.
func stopProcessGroup(_ pid: pid_t) async {
    killpg(pid, SIGINT)
    for tick in 1...300 where kill(pid, 0) == 0 {
        if tick == 200 { killpg(pid, SIGKILL) }
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

/// A failure of `model`'s in words a user can act on; the log keeps the original.
func plainProblem(_ message: String, model: String) -> String {
    if let reason = message.firstMatch(of: /^cannot prepare .+?: (.+)$/)?.1 {
        return "Splash can’t serve these files: \(reason)"
    }
    if message.contains("Repository Not Found") || message.contains("404 Client Error") {
        return "Hugging Face has no public model named \(model.split(separator: ":")[0]). Check the name, then try again."
    }
    if message.contains("ConnectionError") || message.contains("Max retries exceeded") || message.contains("nodename nor servname") {
        return "Splash couldn’t reach Hugging Face. Check the internet connection, then try again."
    }
    return message
}

/// The message of an error line, without its prefix: `error: ...` from the launcher, installer
/// and engine, `HH:MM:SS Error · ...` from the server, `PROGRAM: error: ...` from an argument
/// parser refusing its command line.
func errorMessage(_ line: String) -> String? {
    line.firstMatch(of: /^(?:error: |\d\d:\d\d:\d\d Error · |[\w.\- ]+: error: )(.+)/).map { String($0.1) }
}
