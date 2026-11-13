import Foundation

/// The coding agents `splash` connects to the model being served. Each is a terminal program, so
/// the app opens one in a terminal of the window's drawer, in the folder the agents work in, typing
/// `splash AGENT --port PORT`, which finds the agent on the shell's PATH and tells how to install
/// it when it is missing; the shell stays once the agent ends.
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

    /// The command line a terminal types to run the agent on the server at `port`: this runtime's
    /// `splash`, each word quoted for the shell.
    func command(runtime: Runtime, port: Int) -> String {
        runtime.command([rawValue, "--port", String(port)]).map(Self.quoted).joined(separator: " ")
    }

    private static func quoted(_ word: String) -> String {
        "'" + word.replacing("'", with: #"'\''"#) + "'"
    }
}
