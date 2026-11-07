import Foundation
import Security

/// The keys of what the app keeps in UserDefaults.
enum Preference {
    /// The model picked on the Server screen, the one Start starts.
    static let model = "model"
    /// The model files on this Mac the user added, by path, which `splash serve` prepares.
    static let localModels = "localModels"
    static let agentFolder = "agentFolder"
    /// Whether the Chat shows its sidebar.
    static let sidebar = "sidebar"
    /// The reasoning effort chosen for each model, by its ID; none for its default.
    static let efforts = "efforts"
    /// The launch options set, by flag (`ServeSettings`).
    static let launchOptions = "launchOptions"
    /// The server and the download the app runs (`ChildRecord`), kept until they exit, to stop
    /// them on the next launch should the app quit without stopping them.
    static let serverPID = "serverPID"
    static let installerPID = "installerPID"
    /// The port of the server an app before 1.6 kept beside its ID alone (`Server.stopLeftover`).
    static let olderServerPort = "serverPort"
}

/// How a model starts: `splash serve --model MODEL --port PORT --offline` and the launch options
/// set, each an option of `LaunchOption.all`; one not set is the server's default. The API key is
/// kept in the Keychain (`AppState` reads and writes it) and reaches the server in its environment,
/// never its command line.
struct ServeSettings: Equatable {
    static let defaultPort = 8000

    /// The value of each option set, by its flag: "true" for a switch that is on, the value
    /// `splash serve` takes otherwise.
    var values: [String: String] = [:]
    var apiKey: String?

    subscript(flag: String) -> String? {
        get { values[flag] }
        set { values[flag] = newValue }
    }

    var port: Int { values["--port"].flatMap(Int.init) ?? Self.defaultPort }

    /// The options UserDefaults keeps, of the flags the app offers and values they take; no key.
    static func current(_ defaults: UserDefaults) -> ServeSettings {
        let kept = defaults.dictionary(forKey: Preference.launchOptions) as? [String: String] ?? [:]
        return ServeSettings(values: kept.filter { flag, value in LaunchOption.all.contains { $0.flag == flag && $0.accepts(value) } })
    }

    /// Keeps the options in UserDefaults; the key is kept apart.
    func save(_ defaults: UserDefaults) {
        defaults.set(values, forKey: Preference.launchOptions)
    }

    /// What `splash serve` takes to serve `model`, a Hugging Face ID or the path of model files on
    /// this Mac, with these options.
    func serveArguments(model: String) -> [String] {
        let source = model.hasPrefix("/") ? ["--model", Self.localID(model), "--model-dir", model] : ["--model", model]
        var arguments = ["serve"] + source + ["--port", String(port), "--offline"]
        for option in LaunchOption.all where option.flag != "--port" && option.applies(self) {
            arguments += option.arguments(values[option.flag])
        }
        return arguments
    }

    /// The ID the API reports for the model files at `path`: local/ and the file's or folder's name,
    /// made a repository name as the server takes one: each run of other characters than letters,
    /// digits and _ becomes one -, unless it is a single . or -; neither begins or ends the name, of
    /// at most 96 characters, nor does .git end it.
    static func localID(_ path: String) -> String {
        var name = URL(filePath: path).lastPathComponent
        if name.hasSuffix(".gguf") { name.removeLast(5) }
        name = name.replacing(/[^A-Za-z0-9_]{2,}|[^A-Za-z0-9_.\-]/, with: "-")
        let edges = CharacterSet(charactersIn: "-.")
        name = String(name.trimmingCharacters(in: edges).prefix(96)).trimmingCharacters(in: edges)
        name = name.replacing(/(\.git)+$/, with: "")
        return "local/" + (name.isEmpty ? "model" : name)
    }

    /// What the server's environment adds: the API key it asks for.
    var environment: [String: String] {
        apiKey.map { ["SPLASH_API_KEY": $0] } ?? [:]
    }

    /// The options a model of `family` starts with: those set but the ones it does not take
    /// (`ModelFamily.takes`), whose values stay kept for a model that does.
    func usable(by family: ModelFamily) -> ServeSettings {
        ServeSettings(values: values.filter { family.takes($0.key, $0.value) }, apiKey: apiKey)
    }

    /// These options with the ones a model of `family` can use, those the card shows, back at their
    /// defaults, the API key too; values kept for the others stay, for a model that can use them.
    func defaults(for family: ModelFamily) -> ServeSettings {
        ServeSettings(values: values.filter { !family.uses($0.key) })
    }

    /// Whether a model of `family` started with these options would start otherwise with `other`.
    func differs(from other: ServeSettings, model: String, family: ModelFamily) -> Bool {
        usable(by: family).serveArguments(model: model) != other.usable(by: family).serveArguments(model: model)
            || apiKey != other.apiKey
    }

    /// The server's address: it listens on the loopback address, alone or with every other one, and
    /// the app's chat and status requests and the coding agents go there.
    var url: URL { URL(string: "http://127.0.0.1:\(port)")! }

    /// A request to `path` on the server, with the API key it asks for.
    func request(_ path: String) -> URLRequest {
        var request = URLRequest(url: url.appending(path: path))
        if let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        return request
    }

    /// The memory limits the menu offers on a Mac with `memory` GB: below what it has.
    static func memoryLimits(memory: Int) -> [Int] {
        [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384].filter { $0 < memory }
    }
}

/// A `splash serve` option the Server screen offers: its flag, its words, its part of the screen,
/// and how its value is chosen. `all` holds every option `splash serve` takes but those the app
/// sets itself (`--model`, `--model-dir`, `--offline`) and those of one model rather than every
/// start (`--revision`, `--draft-model`, `--mmproj`).
struct LaunchOption: Identifiable, Sendable {
    enum Section: String, CaseIterable {
        case main = "Launch Options", memory = "Memory", cache = "SSD Cache"
        case network = "Network and Access", api = "API", requests = "Requests"
    }

    enum Kind {
        /// On or off, the flag present or absent; `on` is whether the flag's presence is what
        /// the switch's words say.
        case toggle(on: Bool)
        /// One of `choices`, or, with `custom`, a value typed in; none is the server's default,
        /// named `standard`.
        case choice(standard: String, choices: [(value: String, title: String)], custom: Value?)
        /// A value typed in; with `list`, the option once for each value, split at commas.
        case text(Value, list: Bool)
        /// A folder picked.
        case folder
        /// The API key, typed in a field that hides it and kept apart (`ServeSettings.apiKey`).
        case secret
    }

    /// A value typed in: the words that ask for it, for a model of a family, and the value
    /// `splash serve` takes for a text, nil where it takes none.
    struct Value: Sendable {
        let prompt: String
        let message: @Sendable (ModelFamily) -> String
        let parse: @Sendable (String) -> String?
        /// A value as the field shows it.
        var text: @Sendable (String) -> String = { $0 }
        /// A value as the menu names it.
        var title: @Sendable (String) -> String = { $0 }

        init(prompt: String, message: @escaping @Sendable (ModelFamily) -> String, parse: @escaping @Sendable (String) -> String?,
             text: @escaping @Sendable (String) -> String = { $0 }, title: @escaping @Sendable (String) -> String = { $0 }) {
            self.prompt = prompt
            self.message = message
            self.parse = parse
            self.text = text
            self.title = title
        }

        /// One whose words are the same for every model.
        init(prompt: String, message: String, parse: @escaping @Sendable (String) -> String?,
             text: @escaping @Sendable (String) -> String = { $0 }, title: @escaping @Sendable (String) -> String = { $0 }) {
            self.init(prompt: prompt, message: { _ in message }, parse: parse, text: text, title: title)
        }
    }

    let flag: String
    let title: String
    let help: String
    let section: Section
    let kind: Kind
    /// What the option needs of the others to have effect; one that hasn't shows dimmed, a switch
    /// as off, says what it lacks, and doesn't reach the server.
    var requires: [Requirement] = []
    /// What to beware of with the options set as they are, if anything.
    var warning: @Sendable (ServeSettings) -> String? = { _ in nil }

    var id: String { flag }

    /// What an option needs of the others, in a sentence, and whether they have it.
    struct Requirement: Sendable {
        let reason: String
        let met: @Sendable (ServeSettings) -> Bool

        init(_ reason: String, met: @escaping @Sendable (ServeSettings) -> Bool) {
            self.reason = reason
            self.met = met
        }
    }

    /// What the option lacks to have effect with the others set as they are; nil while it has it.
    func lacks(_ settings: ServeSettings) -> String? {
        let reasons = requires.filter { !$0.met(settings) }.map(\.reason)
        return reasons.isEmpty ? nil : reasons.joined(separator: " ")
    }

    /// Whether the option has effect with the others set as they are.
    func applies(_ settings: ServeSettings) -> Bool {
        lacks(settings) == nil
    }

    /// What `splash serve` takes for `value`: the option, its value, or nothing.
    func arguments(_ value: String?) -> [String] {
        guard let value else { return [] }
        switch kind {
        case .toggle: return value == "true" ? [flag] : []
        case .text(_, list: true): return Self.items(value).flatMap { [flag, $0] }
        case .choice, .text, .folder: return [flag, value]
        case .secret: return []
        }
    }

    /// Whether `value` is one this option takes: a choice, or a value typed in as its field shows it.
    func accepts(_ value: String) -> Bool {
        switch kind {
        case .toggle: value == "true"
        case .choice(_, let choices, let custom):
            choices.contains { $0.value == value } || custom.map { $0.parse($0.text(value)) == value } ?? false
        case .text(let custom, _): custom.parse(custom.text(value)) == value
        case .folder: value.hasPrefix("/")
        case .secret: false
        }
    }

    /// The values of a list typed in: split at commas and spaces.
    static func items(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == "," || $0.isWhitespace }).map(String.init)
    }
}

/// What the models of one family make of the launch options, as the runtime reports it for each
/// family (`install/families.py`): the flags of the options none of them can use, which the Server
/// screen leaves out and a start doesn't pass, and the context they serve: their own window, and
/// the most past it, as far as YaRN stretches it; a longer context length set stays kept for a
/// model that serves it, and a start passes Auto. Every option is used by a family not known.
struct ModelFamily: Decodable, Equatable, Sendable {
    var unusable: Set<String> = []
    var contextTokens: Int?
    var maxContextTokens: Int?

    static let unknown = ModelFamily()

    enum CodingKeys: String, CodingKey {
        case unusable, contextTokens = "context_tokens", maxContextTokens = "max_context_tokens"
    }

    /// Whether its models can use the option `flag`.
    func uses(_ flag: String) -> Bool {
        !unusable.contains(flag)
    }

    /// Whether its models take `value` for the option `flag`: one they can use, and no refusal.
    func takes(_ flag: String, _ value: String) -> Bool {
        uses(flag) && refusal(flag, value) == nil
    }

    /// Why its models don't take `value` for the option `flag`: a context length past the most
    /// they serve; nil for any other.
    func refusal(_ flag: String, _ value: String) -> String? {
        guard flag == "--max-context", let tokens = Int(value), let most = maxContextTokens, tokens > most else { return nil }
        return "The model picked serves at most \(ContextLength.title(most)); it starts on Auto."
    }
}

extension LaunchOption {
    static let all: [LaunchOption] = main + memory + cache + network + api + requests

    private static let main: [LaunchOption] = [
        LaunchOption(
            flag: "--max-context", title: "Context length",
            help: "How much of a conversation the model keeps in mind. Auto fits it to this Mac’s memory.",
            section: .main,
            kind: .choice(standard: "Auto", choices: ContextLength.lengths.map { (String($0), ContextLength.title($0)) },
                          custom: Value(prompt: "100K", message: ContextLength.message,
                                        parse: { ContextLength.parse($0).map(String.init) },
                                        text: { Int($0).map(ContextLength.text) ?? $0 },
                                        title: { Int($0).map(ContextLength.title) ?? $0 }))),
        LaunchOption(
            flag: "--max-memory", title: "Memory limit",
            help: "The most memory the model may use, to leave room for other apps. Auto gives it what this Mac can.",
            section: .main,
            kind: .choice(standard: "Auto", choices: ServeSettings.memoryLimits(memory: Catalog.memory).map { ("\($0)G", "\($0) GB") },
                          custom: Value(prompt: "40", message: "Gigabytes, a whole number. This Mac has \(Catalog.memory) GB.",
                                        parse: { Int($0.trimmingCharacters(in: .whitespaces)).flatMap { $0 > 0 ? "\($0)G" : nil } },
                                        text: { String($0.dropLast()) }, title: { "\($0.dropLast()) GB" }))),
        LaunchOption(
            flag: "--idle-release", title: "Free memory when idle",
            help: "An idle model frees its memory and loads again with the next message.",
            section: .main,
            kind: .choice(standard: "After 10 Minutes",
                          choices: [("30m", "After 30 Minutes"), ("1h", "After 1 Hour"), ("off", "Never")],
                          custom: duration(message: "Seconds, or with s, m or h, such as 45m."))),
        LaunchOption(
            flag: "--language-only", title: "Text only",
            help: "Leaves the model’s vision out: it loads less, and refuses images and PDFs.",
            section: .main, kind: .toggle(on: true)),
    ]

    private static let memory: [LaunchOption] = [
        LaunchOption(
            flag: "--kv-format", title: "KV cache",
            help: "How the model keeps what it has read in memory. 16-bit takes twice the memory.",
            section: .memory, kind: .choice(standard: "8-bit", choices: [("bf16", "16-bit")], custom: nil)),
        LaunchOption(
            flag: "--state-format", title: "Recurrent state",
            help: "How a hybrid model such as Qwen3.6 keeps its state. 16-bit takes half the memory, so twice as many fit.",
            section: .memory, kind: .choice(standard: "32-bit", choices: [("bf16", "16-bit")], custom: nil)),
        LaunchOption(
            flag: "--expert-memory", title: "Expert cache",
            help: "MoE expert weights kept in memory; the others are read from the SSD as needed. Auto gives them what the context leaves. All keeps every expert in memory.",
            section: .memory, kind: .choice(standard: "Auto", choices: [("off", "All")], custom: size(example: "16G"))),
        LaunchOption(
            flag: "--ple-memory", title: "PLE table",
            help: "Memory for Qwen3.8-Flash-Next’s per-layer n-gram embedding table; the rest is read from the SSD. All loads the whole table, tens of GB.",
            section: .memory, kind: .choice(standard: "Auto", choices: [("off", "All")], custom: size(example: "2G"))),
        LaunchOption(
            flag: "--sparse-kv-memory", title: "Sparse KV cache",
            help: "Memory for the sparse-attention KV blocks Qwen3.8-Flash-Next reads back from the SSD; never below its floor.",
            section: .memory, kind: .choice(standard: "Auto", choices: [], custom: size(example: "1200M")),
            requires: [ssdCache]),
    ]

    private static let cache: [LaunchOption] = [
        LaunchOption(
            flag: "--max-cache-disk", title: "Size",
            help: "Room on the SSD for earlier prompts, so a long conversation goes on without reading it again. Auto follows the free space.",
            section: .cache,
            kind: .choice(standard: "Auto", choices: [("0", "Off"), ("16G", "16 GB"), ("32G", "32 GB"), ("64G", "64 GB")],
                          custom: size(example: "48G"))),
        LaunchOption(
            flag: "--max-cache-disk-writes", title: "Writes a day",
            help: "The most the SSD cache writes in a day, to spare the SSD.",
            section: .cache,
            kind: .choice(standard: "No Limit", choices: [("100G", "100 GB"), ("200G", "200 GB"), ("500G", "500 GB")],
                          custom: size(example: "300G")),
            requires: [ssdCache]),
        LaunchOption(
            flag: "--persistent-cache", title: "Keep after stopping",
            help: "The next start of the model goes on from the prompts it kept.",
            section: .cache, kind: .toggle(on: true), requires: [ssdCache]),
        LaunchOption(
            flag: "--cache-dir", title: "Folder",
            help: "Where the SSD cache it keeps lives.",
            section: .cache, kind: .folder,
            requires: [Requirement("Needs Keep after stopping on.") { $0["--persistent-cache"] == "true" }, ssdCache]),
    ]

    private static let network: [LaunchOption] = [
        LaunchOption(
            flag: "--port", title: "Port",
            help: "Where the model answers, for the chat, coding agents and other apps.",
            section: .network,
            kind: .text(Value(prompt: String(ServeSettings.defaultPort), message: "",
                              parse: { Int($0).flatMap { (1...65535).contains($0) ? String($0) : nil } }), list: false)),
        LaunchOption(
            flag: "--host", title: "Listen on",
            help: "This Mac only, or every network it is on, for other devices to reach the model.",
            section: .network,
            kind: .choice(standard: "This Mac Only", choices: [("0.0.0.0", "All Networks")], custom: nil),
            warning: { $0["--host"] != nil && $0.apiKey == nil
                ? "Any device on these networks can use the model. Set an API key to require one." : nil }),
        LaunchOption(
            flag: "--allowed-host", title: "Host names",
            help: "Other names clients reach this Mac by, such as mymac.local.",
            section: .network,
            kind: .text(Value(prompt: "None", message: "", parse: { items($0).isEmpty ? nil : items($0).joined(separator: ", ") }),
                        list: true)),
        LaunchOption(
            flag: "--allowed-origin", title: "Web origins",
            help: "Origins whose pages may call the API from a browser, such as tauri://localhost; * for any.",
            section: .network,
            kind: .text(Value(prompt: "None", message: "", parse: { items($0).isEmpty ? nil : items($0).joined(separator: ", ") }),
                        list: true),
            warning: { $0["--allowed-origin"].map(items)?.contains("*") == true && $0.apiKey == nil
                ? "Any website open in a browser can use the model. Set an API key to require one." : nil }),
        LaunchOption(
            flag: "--api-key", title: "API key",
            help: "A key every request but the chat page’s must carry. Splash’s own chat and coding agents send it.",
            section: .network, kind: .secret),
        LaunchOption(
            flag: "--no-webui", title: "Browser chat page",
            help: "The chat page at the model’s address, for a browser.",
            section: .network, kind: .toggle(on: false)),
    ]

    private static let api: [LaunchOption] = [
        LaunchOption(
            flag: "--default-reasoning-effort", title: "Thinking",
            help: "How much the model thinks for apps whose requests don’t say. The Chat asks for its own.",
            section: .api,
            kind: .choice(standard: "Model’s Default",
                          choices: [("none", "No Thinking"), ("minimal", "Minimal"), ("low", "Low"), ("medium", "Medium"),
                                    ("high", "High"), ("xhigh", "Extra High"), ("max", "Max")], custom: nil)),
        LaunchOption(
            flag: "--thinking-token-budget", title: "Thinking tokens",
            help: "The most a request that sets none thinks before its thinking closes.",
            section: .api,
            kind: .choice(standard: "No Limit", choices: [], custom: Value(prompt: "4096", message: "Tokens, a whole number.",
                                                                          parse: whole))),
        LaunchOption(
            flag: "--thinking-end-text", title: "Thinking end text",
            help: "What closes thinking where its budget runs out, such as a sentence that turns to the answer.",
            section: .api,
            kind: .text(Value(prompt: "Model’s Default", message: "", parse: { $0.isEmpty ? nil : $0 }), list: false)),
        LaunchOption(
            flag: "--served-model-name", title: "Served model names",
            help: "More names requests may give the model, such as gpt-4o for an app that asks for it.",
            section: .api,
            kind: .text(Value(prompt: "None", message: "", parse: { items($0).isEmpty ? nil : items($0).joined(separator: ", ") }),
                        list: true)),
        LaunchOption(
            flag: "--announce-served-name", title: "Report served name",
            help: "Replies and the model list give the first served model name rather than the model’s own.",
            section: .api, kind: .toggle(on: true),
            requires: [Requirement("Needs a served model name.") { $0["--served-model-name"] != nil }]),
    ]

    private static let requests: [LaunchOption] = [
        LaunchOption(
            flag: "--request-timeout", title: "Request timeout",
            help: "How long a request may wait and run before it ends with an error.",
            section: .requests,
            kind: .choice(standard: "None", choices: [("5m", "5 Minutes"), ("30m", "30 Minutes"), ("1h", "1 Hour")],
                          custom: duration(message: "Seconds, or with s, m or h, such as 10m."))),
        LaunchOption(
            flag: "--queue-size", title: "Queue limit",
            help: "Requests taken at once, running or waiting; more are turned away.",
            section: .requests,
            kind: .choice(standard: "32", choices: [], custom: Value(prompt: "64", message: "A whole number.", parse: whole))),
        LaunchOption(
            flag: "--max-request-size", title: "Request size",
            help: "The largest request body, images and PDFs included.",
            section: .requests, kind: .choice(standard: "128 MB", choices: [], custom: size(example: "256M"))),
        LaunchOption(
            flag: "--max-image-pixels", title: "Image pixels",
            help: "The most pixels an image is resized to, 65,536 to 4,194,304.",
            section: .requests,
            kind: .choice(standard: "4,194,304", choices: [], custom: Value(prompt: "1048576", message: "Pixels, 65,536 to 4,194,304.",
                                                                           parse: whole)),
            requires: [Requirement("Needs the model’s vision (Text only off).") { $0["--language-only"] != "true" }]),
        LaunchOption(
            flag: "--decode-share", title: "Decode share",
            help: "Seconds of answering for each second spent reading a new prompt: more keeps answers flowing, less starts the new one sooner; 0 takes turns step by step.",
            section: .requests,
            kind: .choice(standard: "0.5", choices: [], custom: Value(prompt: "1", message: "A number, 0 or more.",
                                                                     parse: { Double($0).flatMap { $0 >= 0 ? String($0) : nil } }))),
        LaunchOption(
            flag: "--verify-rows", title: "Repeatable answers",
            help: "A MoE model’s greedy and seeded answers repeat when a request runs alone with the same prompts cached, at some cost in speed. A prompt sent again reads from the cache, so its answer can differ.",
            section: .requests, kind: .choice(standard: "Off", choices: [("all", "On")], custom: nil)),
        LaunchOption(
            flag: "--disable-ane", title: "Neural Engine for long prompts",
            help: "Reads long prompts faster on models that can use it.",
            section: .requests, kind: .toggle(on: false)),
        LaunchOption(
            flag: "--allow-idle-sleep", title: "Keep Mac awake while answering",
            help: "The Mac doesn’t sleep in the middle of an answer; the display still may.",
            section: .requests, kind: .toggle(on: false)),
    ]

    /// What the options of the SSD cache need: the cache itself.
    private static let ssdCache = Requirement("Needs the SSD cache (SSD Cache Size not Off).") { $0["--max-cache-disk"] != "0" }

    /// A size typed in, as `splash serve` takes one: bytes, or a number with K, M or G.
    private static func size(example: String) -> Value {
        Value(prompt: example, message: "A size with K, M or G, such as \(example).",
              parse: { text in
                  let value = text.trimmingCharacters(in: .whitespaces).uppercased().replacing(/\s+/, with: "")
                  return value.wholeMatch(of: /\d+(\.\d+)?[KMG]|\d+/) != nil ? value : nil
              },
              title: { $0.replacing(/([KMG])$/) { " \($0.1)B" } })
    }

    /// A time typed in, as `splash serve` takes one: seconds, or a number with s, m or h.
    private static func duration(message: String) -> Value {
        Value(prompt: "45m", message: message,
              parse: { text in
                  let value = text.trimmingCharacters(in: .whitespaces).lowercased()
                  return value.wholeMatch(of: /\d+[smh]?/) != nil ? value : nil
              })
    }

    private static let whole: @Sendable (String) -> String? = { Int($0.trimmingCharacters(in: .whitespaces)).flatMap { $0 > 0 ? String($0) : nil } }
}

/// How long a model's context may grow, in tokens, as `--max-context` takes it; the server fits
/// it to memory on Auto.
enum ContextLength {
    /// The lengths the menu offers besides Auto and one typed in.
    static let lengths = [16, 32, 64, 128, 256].map { $0 * 1024 }

    /// A length as the menu names it: 32K tokens, 1M tokens, 100,000 tokens.
    static func title(_ tokens: Int) -> String {
        "\(figure(tokens)) tokens"
    }

    /// A length in short: 32K, 1M, 100,000.
    static func figure(_ tokens: Int) -> String {
        tokens % 1024 == 0 || tokens % 1_000_000 == 0 ? text(tokens) : tokens.formatted()
    }

    /// A length as it is typed, which `parse` reads back: 32K, 1M, 100000.
    static func text(_ tokens: Int) -> String {
        if tokens % 1_000_000 == 0 { return "\(tokens / 1_000_000)M" }
        if tokens % 1024 == 0 { return "\(tokens / 1024)K" }
        return String(tokens)
    }

    /// What the sheet that asks for a length says, for a model of `family`: the most it serves,
    /// and whether YaRN stretches its own window that far.
    static let message: @Sendable (ModelFamily) -> String = { family in
        let units = "Tokens, with K for 1024 or M for a million"
        guard let window = family.contextTokens, let most = family.maxContextTokens else {
            return "\(units), such as 100K or 1M. Past the model’s own window it stretches it with YaRN, which changes answers at every length."
        }
        guard most > window else { return "\(units), such as 100K; at most \(figure(window)), the model’s own window." }
        return "\(units), such as 100K or 1M; at most \(figure(most)). Past the model’s own window, \(figure(window)), it stretches it with YaRN, which changes answers at every length."
    }

    /// The most context `splash serve` takes, its MAX_YARN_CONTEXT_TOKENS, which a test holds this
    /// to (ServeSettingsTests); dev/tests/install/test_families.py holds that to the most any
    /// family serves.
    static let ceiling = 1_010_000

    /// A length typed as `--max-context` takes it: a token count, with K for 1024 tokens or M for
    /// a million, up to `ceiling`; nil for anything else.
    static func parse(_ text: String) -> Int? {
        guard let match = text.trimmingCharacters(in: .whitespaces).uppercased().wholeMatch(of: /(\d{1,9})\s*([KM]?)/),
              let count = Int(match.1), count > 0
        else { return nil }
        let tokens = count * (match.2 == "K" ? 1024 : match.2 == "M" ? 1_000_000 : 1)
        return tokens <= ceiling ? tokens : nil
    }
}

/// Where the API key is kept: an item of the login Keychain.
struct Keychain: Sendable {
    /// The app's own.
    static let app = Keychain(service: "Splash")

    let service: String
    var account = "API key"

    /// Why the Keychain didn't read or keep the key.
    struct Failure: LocalizedError {
        let status: OSStatus
        var errorDescription: String? { SecCopyErrorMessageString(status, nil) as String? ?? "Keychain error \(status)." }
    }

    private var query: [CFString: Any] {
        [kSecClass: kSecClassGenericPassword, kSecAttrService: service, kSecAttrAccount: account]
    }

    /// The key kept, or nil for none; throws when the Keychain doesn't give it, as when access to it
    /// is denied.
    func read() throws -> String? {
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query.merging([kSecReturnData: true, kSecMatchLimit: kSecMatchLimitOne]) { $1 } as CFDictionary, &item)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = item as? Data else { throw Failure(status: status) }
        return String(decoding: data, as: UTF8.self)
    }

    /// Keeps `key` in place of the one kept; nil or empty removes it.
    func write(_ key: String?) throws {
        var status: OSStatus
        if let key, !key.isEmpty {
            let data: [CFString: Any] = [kSecValueData: Data(key.utf8)]
            status = SecItemUpdate(query as CFDictionary, data as CFDictionary)
            if status == errSecItemNotFound { status = SecItemAdd(query.merging(data) { $1 } as CFDictionary, nil) }
        } else {
            status = SecItemDelete(query as CFDictionary)
            if status == errSecItemNotFound { status = errSecSuccess }
        }
        guard status == errSecSuccess else { throw Failure(status: status) }
    }
}
