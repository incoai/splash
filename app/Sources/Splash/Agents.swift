import AppKit

/// The coding agents `splash` connects to the model being served. Each is a terminal program, so
/// the app opens Terminal in a project folder running `splash AGENT --port PORT`, which finds the
/// agent on the shell's PATH and tells how to install it when it is missing.
enum Agent: String, CaseIterable, Identifiable {
    case claude, codex, opencode, hermes, pi

    var id: String { rawValue }
    var title: String {
        switch self {
        case .claude: "Claude Code"
        case .codex: "Codex"
        case .opencode: "OpenCode"
        case .hermes: "Hermes"
        case .pi: "Pi"
        }
    }

    /// Asks for the project folder, starting where `defaults` keep the last one picked, then opens
    /// the agent there in Terminal, on the model served with `settings`.
    @MainActor
    func open(runtime: Runtime, settings: ServeSettings, defaults: UserDefaults) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Open \(title)"
        panel.message = "Choose the project folder \(title) works in."
        if let folder = defaults.url(forKey: Preference.agentFolder) {
            panel.directoryURL = folder
        }
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        defaults.set(folder, forKey: Preference.agentFolder)
        let script = script(runtime: runtime, port: settings.port, folder: folder, apiKey: settings.apiKey)
        do {
            let directory = FileManager.default.temporaryDirectory.appending(path: "Splash")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            // A name of its own, as a script removes itself when it starts.
            let url = directory.appending(path: "\(rawValue)-\(UUID().uuidString.prefix(8)).command")
            try Data(script.utf8).write(to: url, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
            let terminal = URL(filePath: "/System/Applications/Utilities/Terminal.app")
            NSWorkspace.shared.open([url], withApplicationAt: terminal, configuration: NSWorkspace.OpenConfiguration())
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    /// The Terminal script that runs the agent in `folder`, with the key the server asks for. It
    /// removes itself as it starts, so that the key, in a file only this user reads, stays no longer.
    func script(runtime: Runtime, port: Int, folder: URL, apiKey: String? = nil) -> String {
        let key = apiKey.map { "export SPLASH_API_KEY=\(Self.quoted($0))\n" } ?? ""
        return """
        #!/bin/sh
        rm -f -- "$0"
        cd \(Self.quoted(folder.path)) || exit
        \(key)clear
        exec \(runtime.command([rawValue, "--port", String(port)]).map(Self.quoted).joined(separator: " "))

        """
    }

    private static func quoted(_ word: String) -> String {
        "'" + word.replacing("'", with: #"'\''"#) + "'"
    }
}
