import AppKit
import Darwin
import Foundation
import SwiftUI
import Testing

@testable import Splash

/// The drawer's terminals and logs, as its tabs, the menus and the app's end drive them; each
/// terminal runs `Shell.bare`.
@MainActor
@Suite(.serialized)
struct DrawerTests {
    /// Closing a terminal ends its shell and the job it runs, and leaves no zombie.
    @Test func closingATerminalEndsItsShellAndWhatItRuns() async throws {
        let (state, folder) = AppState.forTests()
        defer { state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.newTerminal()
        let session = try #require(state.drawer.terminals.first)
        let shell = session.view.process.shellPid
        let job = try await Self.run("sleep 1000", in: session)
        state.drawer.close(session.id)
        try await waitFor(Self.gone(shell) && Self.gone(job), seconds: 5)
    }

    /// Closing a tab ends its shell and picks the tab after it, or the one before where it was
    /// the last; closing another keeps the pick; closing the only one leaves none.
    @Test func closingATabPicksTheOneAfterOrBefore() throws {
        let (state, folder) = AppState.forTests()
        defer { state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        let drawer = state.drawer
        for _ in 0..<3 { state.newTerminal() }
        let (a, b, c) = (drawer.terminals[0], drawer.terminals[1], drawer.terminals[2])
        #expect(drawer.terminal == c.id)
        drawer.terminal = b.id
        drawer.close(b.id)
        #expect(b.ended && drawer.terminals.map(\.id) == [a.id, c.id] && drawer.terminal == c.id)
        drawer.close(c.id)
        #expect(drawer.terminal == a.id)
        state.newTerminal()
        let d = try #require(drawer.terminals.last)
        drawer.close(a.id)
        #expect(drawer.terminal == d.id)
        drawer.close(d.id)
        #expect(drawer.terminals.isEmpty && drawer.terminal == nil)
        #expect([a, b, c, d].allSatisfy { $0.ended })
    }

    /// The drawer, opened on its terminals, starts one where it has none, and only there; opened
    /// on its logs, it starts none.
    @Test func theDrawerStartsATerminalOnlyWhereItHasNone() {
        let (state, folder) = AppState.forTests()
        defer { state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.showDrawer(.logs)
        #expect(state.showsDrawer && state.drawer.terminals.isEmpty)
        state.toggleDrawer()
        #expect(!state.showsDrawer && state.drawer.terminals.isEmpty)
        state.drawer.part = .terminal
        state.toggleDrawer()
        #expect(state.showsDrawer && state.drawer.terminals.count == 1)
        state.toggleDrawer()
        state.toggleDrawer()
        state.showDrawer(.terminal)
        #expect(state.drawer.terminals.count == 1)
    }

    /// A failure's Show Log opens the drawer on that failure's log, wherever it was, and starts no
    /// terminal.
    @Test func showLogOpensTheDrawerOnThatLog() {
        let (state, folder) = AppState.forTests()
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(state.drawer.part == .terminal && state.drawer.log == .server)
        state.showLog(.downloads)
        #expect(state.showsDrawer && state.drawer.part == .logs && state.drawer.log == .downloads)
        #expect(state.drawer.terminals.isEmpty)
        state.drawer.part = .terminal
        state.showLog(.server)
        #expect(state.drawer.part == .logs && state.drawer.log == .server)
        #expect(state.drawer.terminals.isEmpty)
    }

    /// Only the app holds a terminal's master, so that its end, a crash too, hangs up every shell:
    /// a later terminal's shell, and what it runs, which may outlive it, do not hold it.
    @Test func aTerminalsMasterStaysWithTheApp() async throws {
        let (state, folder) = AppState.forTests()
        defer { state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.newTerminal()
        state.newTerminal()
        let later = try #require(state.drawer.terminals.last)
        let job = try await Self.run("sleep 1000", in: later)
        #expect(Self.masters(heldBy: getpid()) >= 2)
        #expect(Self.masters(heldBy: later.view.process.shellPid) == 0 && Self.masters(heldBy: job) == 0)
    }

    /// Option types what the keyboard layout gives, as `|` is Option-7 on a German keyboard, and
    /// not the key after an escape.
    @Test func optionTypesWhatTheLayoutGives() async throws {
        let (state, folder) = AppState.forTests()
        let window = Self.window(WithDrawer { Color.clear }.environment(state))
        defer { Self.close(window); state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.toggleDrawer()
        let session = try #require(state.drawer.terminals.first)
        try await waitFor(window.firstResponder === session.view)
        // The last line typed, kept in a file as the terminal gets it.
        let typed = FileManager.default.temporaryDirectory.appending(path: "splash-drawer-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: typed) }
        _ = try await Self.run("tail -n 1 > \(typed.path)", in: session)
        window.sendEvent(try #require(NSEvent.keyEvent(
            with: .keyDown, location: .zero, modifierFlags: .option, timestamp: 0, windowNumber: window.windowNumber,
            context: nil, characters: "|", charactersIgnoringModifiers: "7", isARepeat: false, keyCode: 0x1A)))
        session.view.send(txt: "\n\u{4}")
        try await waitFor(!((try? Data(contentsOf: typed)) ?? Data()).isEmpty)
        #expect(try String(contentsOf: typed, encoding: .utf8) == "|\n")
    }

    /// A terminal runs a program while one other than its shell has it, not at its prompt, and one
    /// closed runs nothing; the drawer counts them for what a quit would cut off.
    @Test func aTerminalRunsAProgramWhileOneHasIt() async throws {
        let (state, folder) = AppState.forTests()
        defer { state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.newTerminal()
        state.newTerminal()
        let busy = try #require(state.drawer.terminals.first)
        let idle = try #require(state.drawer.terminals.last)
        #expect(state.drawer.runningPrograms == 0 && state.quitCost.message == nil)
        _ = try await Self.run("sleep 1000", in: busy)
        try await waitFor(busy.runsProgram, "the program to have the terminal")
        #expect(!idle.runsProgram && state.drawer.runningPrograms == 1)
        #expect(state.quitCost.message == "A terminal is running a program.")
        // Its close asks first, as closing ends what runs; the other's closes at once.
        state.drawer.askToClose(busy.id)
        #expect(state.drawer.closing == busy.id && !busy.ended)
        state.drawer.closing = nil
        state.drawer.askToClose(idle.id)
        #expect(state.drawer.closing == nil && idle.ended && state.drawer.terminals.map(\.id) == [busy.id])
        state.drawer.close(busy.id)
        #expect(!busy.runsProgram && state.drawer.runningPrograms == 0)
    }

    /// The drawer showing terminals closes as its last one does; showing logs, it stays.
    @Test func closingTheLastTerminalClosesTheDrawer() throws {
        let (state, folder) = AppState.forTests()
        defer { state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.toggleDrawer()
        let only = try #require(state.drawer.terminals.first)
        state.drawer.close(only.id)
        #expect(!state.showsDrawer)
        state.newTerminal()
        state.showDrawer(.logs)
        state.drawer.close(try #require(state.drawer.terminals.first).id)
        #expect(state.showsDrawer && state.drawer.part == .logs)
    }

    /// A shell that exits cleanly, as `exit` ends it, takes its tab with it, as in Terminal; one
    /// that fails stays, ended, with what it said.
    @Test func aShellThatExitsCleanlyTakesItsTab() async throws {
        let (state, folder) = AppState.forTests()
        defer { state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.newTerminal()
        state.newTerminal()
        let (clean, failing) = (state.drawer.terminals[0], state.drawer.terminals[1])
        clean.view.send(txt: "exit\n")
        failing.view.send(txt: "exit 3\n")
        try await waitFor(failing.ended && clean.ended, seconds: 10)
        try await waitFor(state.drawer.terminals.map(\.id) == [failing.id])
    }

    /// A line typed into a terminal as it starts, as an agent's command is, waits for the shell's
    /// prompt: it runs, and shows once, at the prompt, rather than echoed ahead of it too.
    @Test func aLineTypedAsATerminalStartsShowsOnceAtThePrompt() async throws {
        let (state, folder) = AppState.forTests()
        defer { state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        let session = TerminalSession(shell: .bare, environment: ["HOME": NSHomeDirectory(), "PATH": "/usr/bin:/bin"],
                                      typing: "echo typed-$((6 * 7))")
        state.drawer.add(session)
        func screen() -> String { String(decoding: session.view.getTerminal().getBufferAsData(), as: UTF8.self) }
        try await waitFor(screen().contains("typed-42"), seconds: 10)
        #expect(screen().components(separatedBy: "echo typed-").count == 2)
    }

    /// Quitting ends every shell.
    @Test func quittingEndsEveryShell() async throws {
        let (state, folder) = AppState.forTests()
        defer { try? FileManager.default.removeItem(at: folder) }
        state.newTerminal()
        state.newTerminal()
        let shells = state.drawer.terminals.map(\.view.process.shellPid)
        state.quit()
        #expect(state.drawer.terminals.allSatisfy { $0.ended })
        try await waitFor(shells.allSatisfy(Self.gone), seconds: 5)
    }

    /// The terminals get the app's environment without what tells the server its key and the app
    /// its runtime, and find the user's site-packages; the programs the app runs do not.
    @Test func theTerminalsGetTheAppsEnvironmentWithoutTheServersKey() {
        let kept = ["SPLASH_API_KEY", "SPLASH_RUNTIME"].map { ($0, ProcessInfo.processInfo.environment[$0]) }
        defer {
            for (name, value) in kept {
                if let value { setenv(name, value, 1) } else { unsetenv(name) }
            }
        }
        setenv("SPLASH_API_KEY", "secret", 1)
        setenv("SPLASH_RUNTIME", kept[1].1 ?? Showcase.runtime.root.path, 1)
        let runtime = Showcase.runtime
        let user = runtime.userEnvironment
        #expect(user["SPLASH_API_KEY"] == nil && user["SPLASH_RUNTIME"] == nil && user["PYTHONNOUSERSITE"] == nil)
        #expect(user["HOME"] == ProcessInfo.processInfo.environment["HOME"])
        #expect(runtime.environment["SPLASH_API_KEY"] == nil && runtime.environment["PYTHONNOUSERSITE"] == "1")
    }

    /// The programs the app runs get the Hugging Face settings of the user's login shell, as its
    /// startup files make them, an app getting none: the variables the app takes alone, whatever
    /// else the shell prints; none from a shell that doesn't answer in time. No other variable
    /// leaves the shell, whatever its value holds.
    @Test func theProgramsGetTheLoginShellsHuggingFaceSettings() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-zdotdir-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let zsh = Shell(path: "/bin/zsh", arguments: [], name: "zsh")
        var environment = ProcessInfo.processInfo.environment.filter { !Runtime.takesShellSetting($0.key) }
        environment["ZDOTDIR"] = folder.path
        try """
            echo a greeting
            export HF_HUB_CACHE=/Volumes/Models/hub HUGGINGFACE_HUB_VERBOSITY=info OTHER=1
            export HUGGING_FACE_HUB_TOKEN=hf_fake XDG_CACHE_HOME=/Volumes/Cache SECRET=fake-secret
            export NOTES=$'a line\\nHF_LEAK=fake-leak'

            """.write(to: folder.appending(path: ".zshrc"), atomically: true, encoding: .utf8)
        let settings = await Runtime.readShellSettings(zsh, environment: environment)
        #expect(settings == ["HF_HUB_CACHE": "/Volumes/Models/hub", "HUGGINGFACE_HUB_VERBOSITY": "info",
                             "HUGGING_FACE_HUB_TOKEN": "hf_fake", "XDG_CACHE_HOME": "/Volumes/Cache"])
        let runtime = Runtime(root: folder)
        runtime.shellSettings.set(settings)
        #expect(runtime.environment["HF_HUB_CACHE"] == "/Volumes/Models/hub" && runtime.environment["OTHER"] == nil)
        // What the shell gives: the variables taken, a value's newline and all, and nothing else.
        let filter = Process()
        filter.executableURL = URL(filePath: "/bin/sh")
        filter.arguments = ["-c", Runtime.shellSettingsCommand]
        filter.environment = ["HF_TOKEN": "hf_fake", "HF_NOTE": "one\ntwo", "SECRET": "fake-secret",
                              "NOTES": "a line\nHF_LEAK=fake-leak", "XDG_CACHE_HOMES": "/elsewhere"]
        let given = Pipe()
        filter.standardOutput = given
        try filter.run()
        let records = given.fileHandleForReading.readDataToEndOfFile().split(separator: 0).map { String(decoding: $0, as: UTF8.self) }
        filter.waitUntilExit()
        #expect(Set(records) == ["HF_TOKEN=hf_fake", "HF_NOTE=one\ntwo"] && records.count == 2)
        // A shell that hangs is stopped at the limit.
        try "sleep 30\n".write(to: folder.appending(path: ".zshrc"), atomically: true, encoding: .utf8)
        let start = ContinuousClock.now
        #expect(await Runtime.readShellSettings(zsh, environment: environment, limit: .milliseconds(500)).isEmpty)
        #expect(ContinuousClock.now - start < .seconds(5))
        let output = Data("motd\0SPLASH-SHELL-SETTINGS\0HF_HOME=/h\0PATH=/bin\0HF_TOKEN=a=b\0XDG_CACHE_HOME=/c\0SPLASH-SHELL-SETTINGS\0after".utf8)
        #expect(Runtime.shellSettings(in: output, between: "SPLASH-SHELL-SETTINGS") == ["HF_HOME": "/h", "HF_TOKEN": "a=b", "XDG_CACHE_HOME": "/c"])
    }

    /// A terminal runs the login shell the user database names; where it names none, the one
    /// SHELL names, else zsh, each under its name as a login shell.
    @Test func aTerminalRunsTheLoginShellOrElseSHELLOrZsh() {
        let named = Shell.login("/bin/bash", environment: ["SHELL": "/bin/ksh"])
        #expect(named.path == "/bin/bash" && named.arguments.isEmpty && named.name == "-bash")
        let fromSHELL = Shell.login(nil, environment: ["SHELL": "/bin/ksh"])
        #expect(fromSHELL.path == "/bin/ksh" && fromSHELL.name == "-ksh")
        let zsh = Shell.login(nil, environment: [:])
        #expect(zsh.path == "/bin/zsh" && zsh.name == "-zsh")
        #expect(FileManager.default.isExecutableFile(atPath: Shell.login.path))
    }

    /// Once the app has taken SIGTERM for itself, as it does at launch, what it starts, as the
    /// drawer's terminals start their shells, ends on SIGTERM.
    @Test func whatTheAppStartsEndsOnSIGTERM() async throws {
        var before = sigaction()
        sigaction(SIGTERM, nil, &before)
        defer { sigaction(SIGTERM, &before, nil) }
        AppDelegate.catchTerminate()
        let arguments: [UnsafeMutablePointer<CChar>?] = [strdup("sleep"), strdup("30"), nil]
        defer { arguments.forEach { free($0) } }
        var pid: pid_t = 0
        try #require(posix_spawn(&pid, "/bin/sleep", nil, nil, arguments, environ) == 0)
        var status: Int32 = 0
        var reaped = false
        defer { if !reaped { kill(pid, SIGKILL); waitpid(pid, nil, 0) } }
        kill(pid, SIGTERM)
        try await waitFor({ reaped = waitpid(pid, &status, WNOHANG) == pid; return reaped }(), seconds: 5)
        #expect(status & 0x7f == SIGTERM)
    }

    /// Closing the drawer leaves its terminals their size: what runs in them, a full-screen
    /// program among them, keeps its screen.
    @Test func closingTheDrawerKeepsItsTerminalsSize() async throws {
        let (state, folder) = AppState.forTests()
        let window = Self.window(WithDrawer { Color.clear }.environment(state))
        defer { Self.close(window); state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.toggleDrawer()
        let session = try #require(state.drawer.terminals.first)
        try await waitFor(session.view.window === window)
        // Past the slide.
        try await Task.sleep(for: .seconds(1))
        Self.update(window)
        let rows = session.view.getTerminal().rows
        #expect(rows > 10)
        state.toggleDrawer()
        // Past the slide.
        try await Task.sleep(for: .seconds(1))
        Self.update(window)
        #expect(!state.showsDrawer && session.view.getTerminal().rows == rows)
    }

    /// The terminal shown takes the keyboard: one added, one picked, the drawer opened on it. One
    /// hidden, as another is picked or the drawer closes, gives it up; the logs shown take it, and
    /// give it up as the drawer closes.
    @Test func theTerminalShownHasTheKeyboardAndWhatHidesGivesItUp() async throws {
        let (state, folder) = AppState.forTests()
        let window = Self.window(WithDrawer { Color.clear }.environment(state))
        defer { Self.close(window); state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.toggleDrawer()
        let first = try #require(state.drawer.terminals.first)
        try await waitFor(window.firstResponder === first.view)
        state.newTerminal()
        let second = try #require(state.drawer.terminals.last)
        try await waitFor(window.firstResponder === second.view)
        state.drawer.terminal = first.id
        try await waitFor(window.firstResponder === first.view)
        state.toggleDrawer()
        try await waitFor(window.firstResponder === window)
        state.toggleDrawer()
        try await waitFor(window.firstResponder === first.view)
        state.showDrawer(.logs)
        try await waitFor(Self.textView(in: window) != nil)
        let log = try #require(Self.textView(in: window))
        try await waitFor(window.firstResponder === log)
        state.toggleDrawer()
        try await waitFor(window.firstResponder === window)
    }

    /// The drawer has the keyboard to give the composer as it hides where a terminal or the log
    /// has it, or nothing does, as when the terminal that had it has closed; not where a field
    /// elsewhere has it, as chat search or a rename field.
    @Test func theDrawerHasTheKeyboardOnlyWhereItsViewsOrNothingHaveIt() async throws {
        let (state, folder) = AppState.forTests()
        let window = Self.window(WithDrawer { Color.clear }.environment(state))
        defer { Self.close(window); state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.toggleDrawer()
        let terminal = try #require(state.drawer.terminals.first)
        try await waitFor(window.firstResponder === terminal.view)
        #expect(state.drawer.hasKeyboard(in: window))
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 120, height: 22))
        window.contentView?.addSubview(field)
        #expect(window.makeFirstResponder(field))
        #expect(!state.drawer.hasKeyboard(in: window))
        #expect(window.makeFirstResponder(nil))
        #expect(state.drawer.hasKeyboard(in: window))
        state.showDrawer(.logs)
        try await waitFor(Self.textView(in: window) != nil)
        let log = try #require(Self.textView(in: window))
        try await waitFor(window.firstResponder === log)
        #expect(state.drawer.hasKeyboard(in: window))
    }

    /// Find… shows the find bar of what the drawer shows, whose field takes the keyboard, the
    /// keyboard elsewhere.
    @Test func findShowsTheFindBarOfTheLogTheDrawerShows() async throws {
        let (state, folder) = AppState.forTests()
        let window = Self.window(WithDrawer { Color.clear }.environment(state))
        defer { Self.close(window); state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        // Open before the logs show: off the screen, the window may not run the drawer's slide.
        state.toggleDrawer()
        let terminal = try #require(state.drawer.terminals.first)
        try await waitFor(window.firstResponder === terminal.view)
        state.showDrawer(.logs)
        try await waitFor(Self.textView(in: window) != nil)
        let log = try #require(Self.textView(in: window))
        try await waitFor(window.firstResponder === log)
        window.makeFirstResponder(nil)
        state.drawer.find()
        #expect(log.enclosingScrollView?.isFindBarVisible == true)
        #expect((window.firstResponder as? NSTextView)?.isFieldEditor == true)
    }

    /// A log shows the lines that start in its last 512 KB, then what is written to it.
    @Test func aLogShowsTheLinesOfItsLast512KBThenWhatIsWritten() async throws {
        // 12 bytes a line: the last 512 KiB of 600,000 start inside line 6,310.
        let (folder, file) = try Self.log((1...50_000).map(Self.line).joined())
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        let text = try await Self.shownText(in: window)
        #expect(text.string == (6_311...50_000).map(Self.line).joined())
        try Self.append("appended\n", to: file)
        try await waitFor(text.string.hasSuffix("line 050000\nappended\n"))
    }

    /// A character written in two parts, as a write can end inside one, shows whole.
    @Test func aCharacterWrittenInTwoPartsShowsWhole() async throws {
        let (folder, file) = try Self.log("first\n")
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        let text = try await Self.shownText(in: window)
        let dot = Array("·".utf8)
        try Self.append(Data("second\nStarted ".utf8) + dot.prefix(1), to: file)
        try await waitFor(text.string.contains("second\n"))
        try Self.append(Data(dot.dropFirst()) + Data(" a/b\n".utf8), to: file)
        try await waitFor(text.string.hasSuffix(" a/b\n"))
        #expect(text.string == "first\nsecond\nStarted · a/b\n")
    }

    /// A log read in parts gives whole lines, a character split between two reads whole, the line
    /// being written without a character still being written, and the bytes after the last newline
    /// last; read from inside it, its lines from its first newline.
    @Test func aLogReadInPartsGivesWholeLines() {
        let dot = Array("·".utf8)
        var lines = LogLines()
        #expect(lines.take(Data("a\nb".utf8) + dot.prefix(1)) == Data("a\n".utf8))
        #expect(lines.unfinished == Data("b".utf8))
        #expect(lines.take(Data(dot.dropFirst()) + Data("c\nd".utf8)) == Data("b·c\n".utf8))
        #expect(lines.rest() == Data("d".utf8) && lines.rest().isEmpty)
        var inside = LogLines(inside: true)
        #expect(inside.take(Data("ne beg".utf8)).isEmpty && inside.unfinished.isEmpty)
        #expect(inside.take(Data("un before\nnext\nla".utf8)) == Data("next\n".utf8))
        #expect(inside.rest() == Data("la".utf8))
    }

    /// A log followed line by line, as the app follows the server's and the installer's, gives
    /// each line once it ends, a character split between two writes whole, and no empty line;
    /// finished, the last line too, though it lacks its newline.
    @Test func aLogTailGivesEachLineOnceItEnds() async throws {
        let (folder, file) = try Self.log("")
        defer { try? FileManager.default.removeItem(at: folder) }
        var lines: [String] = []
        let tail = try #require(LogTail(file) { lines.append($0) })
        let dot = Array("·".utf8)
        try Self.append(Data("first\n\nStarted ".utf8) + dot.prefix(1), to: file)
        try await waitFor(lines == ["first"])
        try Self.append(Data(dot.dropFirst()) + Data(" a/b\nlast".utf8), to: file)
        try await waitFor(lines.count == 2)
        tail.finish()
        #expect(lines == ["first", "Started · a/b", "last"])
    }

    /// The server's log goes on under the starts before the one that begins: from a blank line and
    /// a line that names it, its oldest lines going past what it keeps; the start's tail gives only
    /// the start's own lines.
    @Test func aStartsLogGoesOnUnderTheStartsBeforeIt() async throws {
        let (folder, file) = try Self.log("an earlier start\nits failure")
        defer { try? FileManager.default.removeItem(at: folder) }
        let start = beginLog(file, heading: "── Starting a/b")
        #expect(try String(contentsOf: file, encoding: .utf8) == "an earlier start\nits failure\n\n── Starting a/b\n")
        var lines: [String] = []
        let tail = try #require(LogTail(file, from: start) { lines.append($0) })
        try Self.append("its first line\n", to: file)
        try await waitFor(lines == ["its first line"])
        tail.finish()
        #expect(lines == ["its first line"])
        beginLog(file, heading: "── Starting c/d", kept: 24)
        #expect(try String(contentsOf: file, encoding: .utf8) == "its first line\n\n── Starting c/d\n")
    }

    /// A start reads only the end of the log before it, however long a run grew it: one past what
    /// it keeps is cut to its last lines there, and one within it has the start's line added, the
    /// file kept.
    @Test func aStartReadsOnlyTheEndOfALongLog() throws {
        let (folder, file) = try Self.log("")
        defer { try? FileManager.default.removeItem(at: folder) }
        // 4 GiB, all but its last line a hole, which reading would fill.
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: 4 << 30)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data("\nthe run's last line\n".utf8))
        try handle.close()
        let start = beginLog(file, heading: "── Starting a/b", kept: 1 << 20)
        #expect(try Data(contentsOf: file) == Data("the run's last line\n\n── Starting a/b\n".utf8))
        #expect(start == UInt64(try Data(contentsOf: file).count))
        let number = try FileManager.default.attributesOfItem(atPath: file.path)[.systemFileNumber] as? Int
        let next = beginLog(file, heading: "── Starting c/d", kept: 1 << 20)
        #expect(try String(contentsOf: file, encoding: .utf8)
            == "the run's last line\n\n── Starting a/b\n\n── Starting c/d\n")
        #expect(next == UInt64(try Data(contentsOf: file).count))
        #expect(try FileManager.default.attributesOfItem(atPath: file.path)[.systemFileNumber] as? Int == number)
    }

    /// As a log grows, its oldest lines go: past 512 K characters, it keeps the lines that start
    /// in its last seven eighths of that.
    @Test func aLogKeepsAtMostTheLinesOfItsLast512KCharactersAsItGrows() async throws {
        let (folder, file) = try Self.log((1...50_000).map(Self.line).joined())
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        let text = try await Self.shownText(in: window)
        try Self.append((50_001...60_000).map(Self.line).joined(), to: file)
        try await waitFor(text.string.hasSuffix(Self.line(60_000)))
        // Shown from line 6,311, 644,280 characters: the last 448 Ki start inside line 21,771.
        #expect(text.string == (21_772...60_000).map(Self.line).joined())
    }

    /// A log shows its end, and keeps it in view as each line is written, the line being written
    /// too, in place as it grows.
    @Test func aLogAtItsEndShowsEachLineThereAsItIsWritten() async throws {
        let (folder, file) = try Self.log((1...2_000).map(Self.wrappingLine).joined())
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        let text = try await Self.shownText(in: window)
        Self.update(window)
        #expect(Self.endInView(text))
        for (number, written) in [(2_001, "line 2001\n"), (2_002, "line 2002 being"), (2_002, " written\n")] {
            try Self.append(written, to: file)
            try await waitFor(text.string.hasSuffix(written), "line \(number)")
            Self.update(window)
            #expect(Self.endInView(text), "line \(number)")
        }
        #expect(text.string.hasSuffix("line 2001\nline 2002 being written\n"))
    }

    /// A log scrolled up stays where it is as lines are written, the line being written too.
    @Test func aLogScrolledUpStaysWhereItIsAsLinesAreWritten() async throws {
        let (folder, file) = try Self.log((1...2_000).map(Self.wrappingLine).joined())
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        let text = try await Self.shownText(in: window)
        let clip = try #require(text.enclosingScrollView?.contentView)
        Self.update(window)
        // Some 20 lines above the end.
        clip.scroll(to: NSPoint(x: 0, y: text.bounds.maxY - clip.bounds.height - 300))
        Self.update(window)
        let shown = Self.shownLines(text), origin = clip.bounds.origin
        for written in ["line 2001\n", "line 2002 being", " written\n"] {
            try Self.append(written, to: file)
            try await waitFor(text.string.hasSuffix(written))
            Self.update(window)
            #expect(Self.shownLines(text) == shown && clip.bounds.origin == origin, "\(written)")
        }
    }

    /// A log scrolled up stays on the lines it shows as its oldest lines go.
    @Test func aLogScrolledUpStaysOnItsLinesAsTheOldestGo() async throws {
        let (folder, file) = try Self.log((1...50_000).map(Self.line).joined())
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        let text = try await Self.shownText(in: window)
        let clip = try #require(text.enclosingScrollView?.contentView)
        Self.update(window)
        clip.scroll(to: NSPoint(x: 0, y: text.bounds.midY))
        Self.update(window)
        let shown = Self.shownLines(text)
        try Self.append((50_001...51_000).map(Self.line).joined(), to: file)
        try await waitFor(text.string.hasSuffix(Self.line(51_000)))
        Self.update(window)
        // Shown from line 6,311, 536,280 characters: the last 448 Ki start inside line 12,771.
        #expect(text.string.hasPrefix(Self.line(12_772)))
        #expect(Self.shownLines(text) == shown)
    }

    /// A log written again from its start, emptied in place or replaced as each start replaces
    /// the server's, shows only what is new.
    @Test func aLogWrittenAgainFromItsStartShowsOnlyWhatIsNew() async throws {
        let (folder, file) = try Self.log(String(repeating: "an older line\n", count: 100))
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        let text = try await Self.shownText(in: window)
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: 0)
        try handle.write(contentsOf: Data("emptied\n".utf8))
        try handle.close()
        try await waitFor(text.string == "emptied\n")
        resetLog(file)
        try Self.append("replaced\n", to: file)
        try await waitFor(text.string == "replaced\n")
    }

    /// A log that does not exist yet, nor its folder, as on a first start, shows what is written
    /// once a start writes it.
    @Test func aLogNotWrittenYetShowsWhatIsWrittenOnceItIs() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-drawer-\(UUID().uuidString)")
        let file = folder.appending(path: "Logs/Splash/server.log")
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        try await waitFor(Self.textView(in: window) != nil)
        let text = try #require(Self.textView(in: window))
        resetLog(file)
        try Self.append("started\n", to: file)
        try await waitFor(text.string == "started\n")
    }

    /// Find, as the Edit menu's Find… sends it to what has the keyboard, shows a log's find bar.
    @Test func findShowsTheFindBarOfTheLogWithTheKeyboard() async throws {
        let (folder, file) = try Self.log("a line to find\n")
        let window = Self.window(LogText(url: file))
        defer { Self.close(window); try? FileManager.default.removeItem(at: folder) }
        let text = try await Self.shownText(in: window)
        let scroll = try #require(text.enclosingScrollView)
        window.makeFirstResponder(text)
        #expect(!scroll.isFindBarVisible)
        window.firstResponder?.showFindBar()
        #expect(scroll.isFindBarVisible)
    }

    /// A window showing `view`, off the screen.
    private static func window<V: View>(_ view: V) -> NSWindow {
        _ = NSApplication.shared
        let window = NSWindow(contentViewController: NSHostingController(rootView: view))
        window.setContentSize(CGSize(width: 800, height: 600))
        window.setFrameOrigin(NSPoint(x: -30000, y: -30000))
        window.orderFrontRegardless()
        return window
    }

    /// Takes the window's views down, as closing it does: a log they follow, which makes its file
    /// again once it is removed, lets the test's folder go.
    private static func close(_ window: NSWindow) {
        window.contentViewController = nil
        window.orderOut(nil)
    }

    /// What SwiftUI owes the window, done now.
    private static func update(_ window: NSWindow) {
        window.contentView?.layoutSubtreeIfNeeded()
        window.displayIfNeeded()
    }

    private static func textView(in window: NSWindow) -> NSTextView? {
        func textViews(in view: NSView) -> [NSTextView] {
            view.subviews.flatMap { ($0 as? NSTextView).map { [$0] } ?? textViews(in: $0) }
        }
        return window.contentView.flatMap { textViews(in: $0).first { !$0.isFieldEditor } }
    }

    /// The window's log, once it shows some text.
    private static func shownText(in window: NSWindow) async throws -> NSTextView {
        try await waitFor(textView(in: window).map { !$0.string.isEmpty } ?? false)
        return try #require(textView(in: window))
    }

    /// Whether the end of `text` is in view.
    private static func endInView(_ text: NSTextView) -> Bool {
        text.visibleRect.maxY >= text.bounds.maxY - 1
    }

    /// The lines at the top and the bottom of the view.
    private static func shownLines(_ text: NSTextView) -> [String] {
        let visible = text.visibleRect, string = text.string as NSString
        return [visible.minY + 1, visible.maxY - 1].map { y in
            let index = text.characterIndexForInsertion(at: NSPoint(x: 20, y: y))
            return string.substring(with: string.lineRange(for: NSRange(location: min(index, string.length - 1), length: 0)))
        }
    }

    /// Line `number` of a test's log, every seventh long enough to wrap.
    private static func wrappingLine(_ number: Int) -> String {
        "line \(number)" + (number % 7 == 0 ? String(repeating: " wrapping", count: 40) : "") + "\n"
    }

    /// A log holding `text`, in a folder of its own.
    private static func log(_ text: String) throws -> (URL, URL) {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-drawer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let file = folder.appending(path: "test.log")
        try Data(text.utf8).write(to: file)
        return (folder, file)
    }

    /// Line `number` of a test's log, 12 bytes.
    private static func line(_ number: Int) -> String {
        String(format: "line %06d\n", number)
    }

    private static func append(_ text: String, to file: URL) throws {
        try append(Data(text.utf8), to: file)
    }

    private static func append(_ data: Data, to file: URL) throws {
        let handle = try FileHandle(forWritingTo: file)
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try handle.close()
    }

    /// Runs `command` in the terminal, as typed, once its shell reads what is typed; the process it
    /// starts, not one of what starting the shell runs.
    private static func run(_ command: String, in session: TerminalSession) async throws -> pid_t {
        let shell = session.view.process.shellPid
        let program = String(command.prefix { $0 != " " })
        let deadline = ContinuousClock.now + .seconds(20)
        while true {
            // Typed again until the shell, still starting, takes it.
            session.view.send(txt: command + "\n")
            for _ in 0..<100 {
                if let job = children(of: shell).first(where: { name(of: $0) == program }) { return job }
                try #require(ContinuousClock.now < deadline, "the shell ran nothing")
                try await Task.sleep(for: .milliseconds(20))
            }
        }
    }

    private static func children(of pid: pid_t) -> [pid_t] {
        var pids = [pid_t](repeating: 0, count: 64)
        let count = proc_listchildpids(pid, &pids, Int32(pids.count * MemoryLayout<pid_t>.size))
        return Array(pids.prefix(Int(max(count, 0))))
    }

    private static func name(of pid: pid_t) -> String {
        var name = [CChar](repeating: 0, count: 256)
        proc_name(pid, &name, UInt32(name.count))
        return String(cString: name)
    }

    /// How many pseudo-terminal masters process `pid` holds.
    private static func masters(heldBy pid: pid_t) -> Int {
        var open = [proc_fdinfo](repeating: proc_fdinfo(), count: 4096)
        let bytes = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, &open, Int32(open.count * MemoryLayout<proc_fdinfo>.stride))
        return open.prefix(Int(bytes) / MemoryLayout<proc_fdinfo>.stride).count { descriptor in
            var file = vnode_fdinfowithpath()
            let size = Int32(MemoryLayout<vnode_fdinfowithpath>.stride)
            guard proc_pidfdinfo(pid, descriptor.proc_fd, PROC_PIDFDVNODEPATHINFO, &file, size) == size else { return false }
            return withUnsafeBytes(of: file.pvip.vip_path) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) } == "/dev/ptmx"
        }
    }

    /// Whether process `pid` has ended and been waited for.
    private static func gone(_ pid: pid_t) -> Bool {
        kill(pid, 0) != 0 && errno == ESRCH
    }
}

extension DrawerTests {
    /// The drawer slides with what is above it and what is in it: each frame drawn as it opens and
    /// closes over the Server screen has the screen's control bar and the drawer's tabs as they show
    /// once it has opened, together, moved as one with the drawer's edge; not the bar's words where
    /// the drawer is going, ahead of the drawer and of the bar's own background, nor the tab of the
    /// terminal it opens with fading in under the New Terminal button.
    @Test func theDrawerSlidesWithWhatIsAboveItAndInIt() async throws {
        let fake = try FakeRuntime("splash-drawer-slide", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await ServerScreenTests.launchedWithModels(fake)
        defer { state.quit() }
        state.options["--port"] = String(fake.port)
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.start("unsloth/flint-GGUF:UD-Q4_K_XL")
        try await waitFor(state.server.phase == .ready, "ready")
        let window = await Snapshot.window(MainView().environment(state), size: CGSize(width: 1120, height: 780), appearance: .aqua)
        defer { window.orderOut(nil); state.drawer.endAll() }
        pump(0.5)
        /// The window drawn now, each row of its pixels as a hash, from the top: drawn at once,
        /// so that the frames come some 20 ms apart.
        func rows() throws -> [Int] {
            let frame = try #require(window.contentView?.superview)
            let bitmap = try #require(frame.bitmapImageRepForCachingDisplay(in: frame.bounds))
            frame.cacheDisplay(in: frame.bounds, to: bitmap)
            let data = try #require(bitmap.bitmapData)
            return (0..<bitmap.pixelsHigh).map { Data(bytes: data + $0 * bitmap.bytesPerRow, count: bitmap.bytesPerRow).hashValue }
        }
        /// The frames drawn through `change` that differ from the one before it, and the one once
        /// it has stopped changing.
        func frames(of change: () -> Void) throws -> (drawn: [[Int]], after: [Int]) {
            let before = try rows()
            change()
            var drawn: [[Int]] = []
            for _ in 0..<15 {
                pump(0.02)
                drawn.append(try rows())
            }
            pump(1)
            return (drawn.filter { $0 != before }, try rows())
        }
        let scale = CGFloat(try rows().count) / window.frame.height
        let opening = try frames { state.toggleDrawer() }
        // Opened: the drawer's edge, 280 points up from the window's bottom; the band from the bar's
        // last 60 points to the drawer's tabs and the divider under them.
        let edge = Int((window.frame.height - min(state.drawerHeight, 280)) * scale)
        let band = Array(opening.after[(edge - Int(60 * scale))..<(edge + Int(38 * scale))])
        /// How much of the band `frame` shows whole, moved down as far as it shows most of it.
        func shown(in frame: [Int]) -> Double {
            let start = edge - Int(60 * scale)
            return (0...Int(300 * scale)).map { offset in
                let compared = band.indices.filter { start + offset + $0 < frame.count }
                guard compared.count >= Int(20 * scale) else { return 0 }
                return Double(compared.count { band[$0] == frame[start + offset + $0] }) / Double(compared.count)
            }.max() ?? 0
        }
        #expect(!opening.drawn.isEmpty && opening.drawn.allSatisfy { shown(in: $0) >= 0.9 },
                "opening: \(opening.drawn.map { shown(in: $0) })")
        let closing = try frames { state.toggleDrawer() }
        #expect(!closing.drawn.isEmpty && closing.drawn.allSatisfy { shown(in: $0) >= 0.9 },
                "closing: \(closing.drawn.map { shown(in: $0) })")
    }
}

extension DrawerTests {
    /// The tab picked shows, past the tabs that fill the drawer's width: one added, the last, with
    /// New Terminal after it, at the strip's end, and the first, picked again, at its start.
    @Test func theTabPickedShowsPastTheTabsThatFillTheDrawer() async throws {
        let (state, folder) = AppState.forTests()
        let window = Self.window(WithDrawer { Color.clear }.environment(state))
        defer { Self.close(window); state.drawer.endAll(); try? FileManager.default.removeItem(at: folder) }
        state.toggleDrawer()
        for _ in 0..<11 { state.newTerminal() }
        /// The tabs' strip, once they are wider than it.
        func strip() -> NSScrollView? {
            func scrollViews(in view: NSView) -> [NSScrollView] {
                view.subviews.flatMap { ($0 as? NSScrollView).map { [$0] } ?? scrollViews(in: $0) }
            }
            return window.contentView.flatMap { scrollViews(in: $0).first { ($0.documentView?.frame.width ?? 0) > $0.contentView.bounds.width + 100 } }
        }
        try await waitFor(strip() != nil, "the tabs wider than their strip")
        let scroll = try #require(strip())
        let document = try #require(scroll.documentView)
        try await waitFor(scroll.contentView.bounds.maxX >= document.frame.width - 1, "the strip's end")
        state.drawer.terminal = state.drawer.terminals.first?.id
        try await waitFor(scroll.contentView.bounds.minX <= 1, "the strip's start")
    }
}
