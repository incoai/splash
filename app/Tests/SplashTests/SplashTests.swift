import AppKit
import Foundation
import Testing

@testable import Splash

struct CatalogTests {
    let catalog = Catalog(text: """
        # MODEL\tNAME\tMEMORY\tSIZE\tABOUT
        a/Big-GGUF:Q4\tBig\t36\t22\tFast
        a/Big-MLX\tBig\t36\t21\tFast
        a/Big-GGUF:Q2\tBig\t24\t14\tFast
        a/Huge-GGUF:Q4\tHuge\t128\t82\tLargest
        a/Small-GGUF:Q2\tSmall\t24\t12\tSmallest download
        not a line
        a/Bad\tBad\tlots\t1\tNo
        """)

    @Test func parsesDataLinesOnly() {
        #expect(catalog.variants.map(\.id) == ["a/Big-GGUF:Q4", "a/Big-MLX", "a/Big-GGUF:Q2", "a/Huge-GGUF:Q4", "a/Small-GGUF:Q2"])
        #expect(catalog.minimumMemory == 24)
        #expect(catalog.variants[0].size == 22)
        #expect(catalog.about(of: "a/Small-GGUF:Q2") == "Smallest download")
        #expect(catalog.about(of: "someone/else") == nil)
    }

    @Test func suggestsTheLargestVariantTheMacHasMemoryFor() {
        #expect(catalog.suggestions(memory: 36).map(\.id) == ["a/Big-GGUF:Q4", "a/Small-GGUF:Q2"])
        #expect(catalog.suggestions(memory: 24).map(\.id) == ["a/Big-GGUF:Q2", "a/Small-GGUF:Q2"])
        #expect(catalog.suggestions(memory: 128).map(\.id) == ["a/Big-GGUF:Q4", "a/Huge-GGUF:Q4", "a/Small-GGUF:Q2"])
        #expect(catalog.suggestions(memory: 16).isEmpty)
    }

    @Test func namesModels() {
        #expect(catalog.name(of: "a/Big-MLX") == "Big")
        #expect(catalog.name(of: "someone/Other-GGUF:Q8_0") == "Other")
        #expect(catalog.name(of: "someone/Other-4bit") == "Other-4bit")
        #expect(Catalog.variant(of: "someone/Other-GGUF:Q8_0") == "Q8_0")
        #expect(Catalog.variant(of: "mlx-community/Other-4bit") == "MLX")
        #expect(Catalog.variant(of: "someone/Other-4bit") == "Other-4bit")
        // Model files on this Mac, by their folder.
        #expect(catalog.name(of: "/a/mlx-models/Qwen3-8B.gguf") == "Qwen3-8B")
        #expect(Catalog.variant(of: "/a/mlx-models/Qwen3-8B.gguf") == "mlx-models")
    }

    @Test func measuresModelFilesOnThisMac() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-local-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder.appending(path: "Q4"), withIntermediateDirectories: true)
        try Data(count: 12_345).write(to: folder.appending(path: "m.gguf"))
        try Data(count: 1000).write(to: folder.appending(path: "Q4/m-00001-of-00002.gguf"))
        #expect(InstalledModel.local(folder.appending(path: "m.gguf").path)
            == InstalledModel(id: folder.appending(path: "m.gguf").path, bytes: 12_345, gguf: true))
        #expect(InstalledModel.local(folder.path).bytes == 13_345 && InstalledModel.local(folder.path).gguf)
        #expect(InstalledModel.local(folder.appending(path: "gone").path).missing)
    }
}

struct ServeSettingsTests {
    @Test func defaultsServeOfflineOnThePort() {
        #expect(ServeSettings().serveArguments(model: "a/b:Q4")
            == ["serve", "--model", "a/b:Q4", "--port", "8000", "--offline"])
    }

    @Test func servesFilesOnThisMacUnderANameTheServerTakes() {
        #expect(ServeSettings().serveArguments(model: "/models/My Model-UD-Q4_K_M.gguf")
            == ["serve", "--model", "local/My-Model-UD-Q4_K_M", "--model-dir", "/models/My Model-UD-Q4_K_M.gguf",
                "--port", "8000", "--offline"])
        for (path, name) in [("/models/Qwen3.8 27B - Q4_K_M.gguf", "Qwen3.8-27B-Q4_K_M"), ("/models/model..v2", "model-v2"),
                             ("/models/" + String(repeating: "x", count: 95) + "-y.gguf", String(repeating: "x", count: 95)),
                             ("/models/repo.git", "repo"), ("/models/repo.git.git", "repo"), ("/models/ - .gguf", "model")] {
            #expect(ServeSettings.localID(path) == "local/" + name)
        }
    }

    @Test func passesTheOptionsSetInTheTablesOrder() {
        let settings = ServeSettings(values: [
            "--port": "8100", "--max-context": "131072", "--max-memory": "48G", "--idle-release": "off",
            "--language-only": "true", "--kv-format": "bf16", "--max-cache-disk": "16G", "--persistent-cache": "true",
            "--host": "0.0.0.0", "--allowed-origin": "a.example, b.example", "--no-webui": "true",
            "--thinking-end-text": "Now the answer. </think>", "--disable-ane": "true",
        ], apiKey: "secret")
        #expect(settings.serveArguments(model: "a/b", localName: nil)
            == ["serve", "--model", "a/b", "--port", "8100", "--offline", "--max-context", "131072",
                "--max-memory", "48G", "--idle-release", "off", "--language-only", "--kv-format", "bf16",
                "--max-cache-disk", "16G", "--persistent-cache", "--host", "0.0.0.0",
                "--allowed-origin", "a.example", "--allowed-origin", "b.example", "--no-webui",
                "--thinking-end-text", "Now the answer. </think>", "--disable-ane"])
        // The key reaches the server in its environment, never its command line.
        #expect(settings.environment == ["SPLASH_API_KEY": "secret"])
        #expect(settings.request("status").value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        #expect(settings.url.absoluteString == "http://127.0.0.1:8100")
    }

    /// Model files on this Mac start with their own projector and draft, which their start alone
    /// passes, after the files; a model Splash installed passes neither. A change of either
    /// starts the model otherwise.
    @Test func passesTheModelFilesOwnProjectorAndDraft() {
        var settings = ServeSettings(values: ["--port": "8100"])
        settings.projector = "/models/mmproj-BF16.gguf"
        settings.draft = "/models/draft"
        #expect(settings.serveArguments(model: "/models/q.gguf", localName: nil)
            == ["serve", "--model", "local/q", "--model-dir", "/models/q.gguf", "--mmproj", "/models/mmproj-BF16.gguf",
                "--draft-model", "/models/draft", "--port", "8100", "--offline"])
        #expect(settings.serveArguments(model: "a/b", localName: nil) == ["serve", "--model", "a/b", "--port", "8100", "--offline"])
        #expect(settings.usable(by: .unknown).projector == settings.projector && settings.usable(by: .unknown).draft == settings.draft)
        var other = settings
        other.projector = nil
        #expect(settings.differs(from: other, model: "/models/q.gguf", family: .unknown))
        // Under Text only, which `splash serve` refuses beside a projector, the projector stays out.
        settings["--language-only"] = "true"
        #expect(settings.textOnly && settings.serveArguments(model: "/models/q.gguf", localName: nil)
            == ["serve", "--model", "local/q", "--model-dir", "/models/q.gguf", "--draft-model", "/models/draft",
                "--port", "8100", "--offline", "--language-only"])
    }

    /// Listening on every network, the server accepts this Mac's .local name, once, which the API
    /// address it gives other devices names; on this Mac alone, neither.
    @Test func givesOtherDevicesThisMacsLocalName() {
        var settings = ServeSettings(values: ["--port": "8100", "--host": "0.0.0.0"])
        #expect(settings.serveArguments(model: "a/b", localName: "mymac.local").suffix(2) == ["--allowed-host", "mymac.local"])
        #expect(settings.address(localName: "mymac.local").absoluteString == "http://mymac.local:8100")
        settings["--allowed-host"] = "proxy.example, mymac.local"
        #expect(settings.serveArguments(model: "a/b", localName: "mymac.local").filter { $0 == "mymac.local" }.count == 1)
        settings["--host"] = nil
        #expect(!settings.serveArguments(model: "a/b", localName: "mymac.local").contains("--host"))
        #expect(settings.address(localName: "mymac.local").absoluteString == "http://127.0.0.1:8100")
        #expect(settings.address(localName: nil) == settings.url)
    }

    @Test func leavesOutOptionsTheOthersLeaveNoEffect() {
        // An SSD cache that is off keeps nothing, writes nothing and has no folder; the first other
        // name needs one.
        let off = ServeSettings(values: ["--max-cache-disk": "0", "--persistent-cache": "true", "--cache-dir": "/tmp/c",
                                         "--max-cache-disk-writes": "100G", "--announce-served-name": "true"])
        #expect(off.serveArguments(model: "a/b") == ServeSettings(values: ["--max-cache-disk": "0"]).serveArguments(model: "a/b"))
        #expect(!off.differs(from: ServeSettings(values: ["--max-cache-disk": "0"]), model: "a/b", family: .unknown))
        #expect(ServeSettings(apiKey: "new").differs(from: ServeSettings(), model: "a/b", family: .unknown))
        // Text only leaves images out, and an SSD cache that is off reads no sparse KV back.
        let text = ServeSettings(values: ["--language-only": "true", "--max-image-pixels": "1048576",
                                          "--max-cache-disk": "0", "--sparse-kv-memory": "2G"])
        #expect(text.serveArguments(model: "a/b") == ["serve", "--model", "a/b", "--port", "8000", "--offline",
                                                      "--language-only", "--max-cache-disk", "0"])
    }

    /// An option the others leave no effect says what it lacks.
    @Test func aDimmedOptionSaysWhatItNeeds() {
        let off = ServeSettings(values: ["--max-cache-disk": "0", "--language-only": "true"])
        let ssd = "Needs the SSD cache (SSD Cache Size not Off)."
        #expect(LaunchOption.all.filter { !$0.applies(off) }.map { [$0.flag, $0.lacks(off)] } == [
            ["--sparse-kv-memory", ssd], ["--max-cache-disk-writes", ssd], ["--persistent-cache", ssd],
            ["--cache-dir", "Needs Keep after stopping on. \(ssd)"],
            ["--announce-served-name", "Needs a served model name."],
            ["--max-image-pixels", "Needs the model’s vision (Text only off)."],
        ])
        // Only what it lacks.
        let cacheDir = LaunchOption.all.first { $0.flag == "--cache-dir" }
        #expect(cacheDir?.lacks(ServeSettings(values: ["--persistent-cache": "true", "--max-cache-disk": "0"])) == ssd)
        let all = ServeSettings(values: ["--persistent-cache": "true", "--served-model-name": "a"])
        #expect(LaunchOption.all.allSatisfy { $0.applies(all) && $0.lacks(all) == nil })
    }

    /// Flint's options, as the runtime reports its family: no vision, recurrent state, PLE table,
    /// sparse KV, Neural Engine or thinking switch.
    static let flint = ModelFamily(unusable: ["--language-only", "--state-format", "--ple-memory", "--sparse-kv-memory",
                                              "--disable-ane", "--max-image-pixels", "--default-reasoning-effort",
                                              "--thinking-token-budget", "--thinking-end-text"],
                                   contextTokens: 262_144, maxContextTokens: 262_144)
    static let moe = ModelFamily(unusable: ["--ple-memory", "--sparse-kv-memory", "--disable-ane"],
                                 contextTokens: 262_144, maxContextTokens: 1_010_000)

    @Test func aModelStartsWithoutTheOptionsItsFamilyCannotUse() {
        let settings = ServeSettings(values: ["--state-format": "bf16", "--default-reasoning-effort": "high",
                                              "--max-image-pixels": "1048576", "--expert-memory": "16G"], apiKey: "k")
        let all = ["serve", "--model", "a/b", "--port", "8000", "--offline", "--state-format", "bf16", "--expert-memory", "16G",
                   "--default-reasoning-effort", "high", "--max-image-pixels", "1048576"]
        #expect(settings.usable(by: Self.flint).serveArguments(model: "a/b")
            == ["serve", "--model", "a/b", "--port", "8000", "--offline", "--expert-memory", "16G"])
        #expect(settings.usable(by: Self.flint).apiKey == "k")
        #expect(settings.usable(by: Self.moe).serveArguments(model: "a/b") == all)
        // A family not known, as of model files on this Mac not prepared yet, takes them all.
        #expect(settings.usable(by: .unknown).serveArguments(model: "a/b") == all)
        // Those it cannot use change nothing of its start; the others do.
        let other = ServeSettings(values: ["--expert-memory": "16G"], apiKey: "k")
        #expect(!settings.differs(from: other, model: "a/b", family: Self.flint))
        #expect(settings.differs(from: other, model: "a/b", family: Self.moe))
        #expect(settings.differs(from: other, model: "a/b", family: .unknown))
        // The card leaves them out, and the bands of parts left empty.
        #expect(LaunchOptions.options(.memory, for: Self.flint).map(\.flag)
            == ["--kv-format", "--expert-memory"])
        #expect(LaunchOptions.options(.memory, for: .unknown).count == 5)
    }

    /// A context length longer than the model picked serves stays kept for a model that serves it,
    /// and a start passes Auto; none is longer than `splash serve` takes.
    @Test func aContextLengthPastTheModelsMostIsKeptButNotPassed() {
        let long = ServeSettings(values: ["--max-context": "1000000"])
        #expect(long.usable(by: Self.flint).serveArguments(model: "a/b") == ["serve", "--model", "a/b", "--port", "8000", "--offline"])
        #expect(long.usable(by: Self.moe) == long && long.usable(by: .unknown) == long)
        #expect(!long.differs(from: ServeSettings(), model: "a/b", family: Self.flint))
        #expect(long.differs(from: ServeSettings(), model: "a/b", family: Self.moe))
        #expect(Self.flint.takes("--max-context", "262144") && !Self.flint.takes("--max-context", "1000000"))
        #expect(Self.flint.refusal("--max-context", "1000000") == "The model picked serves at most 256K tokens; it starts on Auto.")
        #expect(Self.moe.refusal("--max-context", "1000000") == nil && Self.flint.refusal("--max-memory", "1000000G") == nil)
        #expect(ContextLength.parse("1010000") == 1_010_000)
        #expect(ContextLength.parse("5M") == nil && ContextLength.parse("1010001") == nil)
    }

    /// The app takes the context lengths `splash serve` takes: its ceiling is the server's
    /// MAX_YARN_CONTEXT_TOKENS.
    @Test func theAppTakesTheContextLengthsSplashServeTakes() throws {
        let options = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../../server/serve_options.py")
        let source = try String(contentsOf: options, encoding: .utf8)
        let bound = try #require(source.firstMatch(of: /(?m)^MAX_YARN_CONTEXT_TOKENS = ([\d_]+)$/))
        #expect(Int(bound.1.replacing("_", with: "")) == ContextLength.ceiling)
    }

    /// Restore Defaults acts on the options the card shows the model picked; the values kept for
    /// those it leaves out stay, and the API key; alone they don't offer it.
    @Test func defaultsAreRestoredForTheOptionsShown() {
        let hidden = ServeSettings(values: ["--state-format": "bf16"], apiKey: "k")
        let set = ServeSettings(values: ["--state-format": "bf16", "--expert-memory": "16G"], apiKey: "k")
        #expect(set.defaults(for: Self.flint) == hidden && hidden.defaults(for: Self.flint) == hidden)
        #expect(set.defaults(for: Self.moe) == ServeSettings(apiKey: "k") && set.defaults(for: .unknown) == ServeSettings(apiKey: "k"))
    }

    @Test func theContextHintIsTheModels() {
        #expect(ContextLength.message(Self.flint).contains("at most 256K, the model’s own window"))
        #expect(!ContextLength.message(Self.flint).contains("YaRN"))
        #expect(ContextLength.message(Self.moe).contains("at most 1,010,000. Past the model’s own window, 256K, it stretches it with YaRN"))
        #expect(ContextLength.message(.unknown).contains("YaRN"))
    }

    /// The context menu offers the lengths the model picked serves, 1M tokens where YaRN stretches
    /// its window that far, and a length past its window says so under it.
    @Test func theContextMenuOffersTheLengthsTheModelServes() throws {
        let context = try #require(LaunchOption.all.first { $0.flag == "--max-context" })
        #expect(context.choices(for: Self.moe).last?.title == "1M tokens")
        #expect(context.choices(for: .unknown).last?.title == "1M tokens")
        #expect(context.choices(for: Self.flint).last?.title == "256K tokens")
        #expect(Self.moe.caution("--max-context", "1000000")
            == "Past the model’s own window, 256K, it stretches it with YaRN, which changes answers at every length.")
        #expect(Self.moe.caution("--max-context", "262144") == nil && Self.flint.caution("--max-context", "1000000") == nil)
    }

    @Test func readsAndNamesContextLengths() {
        #expect(ContextLength.parse("100K") == 102_400)
        #expect(ContextLength.parse(" 1m ") == 1_000_000)
        #expect(ContextLength.parse("65536") == 65536)
        #expect(ContextLength.parse("0") == nil && ContextLength.parse("12.5K") == nil && ContextLength.parse("lots") == nil)
        #expect(ContextLength.title(32768) == "32K tokens")
        #expect(ContextLength.title(1_000_000) == "1M tokens")
        #expect(ContextLength.title(100_000) == "100,000 tokens")
        // As typed, read back alike.
        #expect([32768, 1_000_000, 100_000].map { ContextLength.parse(ContextLength.text($0)) } == [32768, 1_000_000, 100_000])
    }

    @Test func offersMemoryLimitsBelowTheMacs() {
        #expect(ServeSettings.memoryLimits(memory: 36) == [8, 12, 16, 24, 32])
    }

    @Test func wordsAStartFailureByItsClass() {
        func failure(_ kind: String, _ bound: String? = nil, required: Int64? = nil, budget: Int64? = nil,
                     plan: Int? = nil, message: String = "the engine's words") -> ServeRecord.Failure {
            ServeRecord.Failure(kind: kind, message: message, bound: bound,
                                required_bytes: required, budget_bytes: budget, plan_context_tokens: plan)
        }
        func problem(_ failure: ServeRecord.Failure) -> String { Server.problem(failure, port: 8000) }
        #expect(problem(failure("memory_budget", "max_memory", required: 10_858_659_840, budget: 1_073_741_824))
            == "It needs 10.1 GB of memory but the memory limit is 1 GB. Raise the memory limit in Launch Options.")
        #expect(problem(failure("memory_budget", "device_memory", required: 10_858_659_840, budget: 1_073_741_824))
            == "It needs 10.1 GB of memory but this Mac can give it 1 GB. Choose a smaller model.")
        #expect(problem(failure("context_budget", "device_memory", plan: 73728))
            == "Its context fits up to 72K tokens on this Mac. Choose a shorter context length, or Auto.")
        #expect(problem(failure("context_budget", "max_memory", plan: 73721))
            == "Its context fits up to 73,721 tokens within the memory limit. Choose a shorter context length, or raise the memory limit.")
        #expect(problem(failure("context_budget", "disk", plan: 73721))
            == "Its context fits up to 73,721 tokens in the SSD cache. Choose a shorter context length, or a larger SSD cache.")
        #expect(problem(failure("host_memory"))
            == "Other apps are using the memory it needs. Quit some, then start it again.")
        #expect(problem(failure("command_memory"))
            == "Another app is using the GPU memory this model needs, such as another Splash server. Quit it, then start again.")
        #expect(problem(failure("port"))
            == "Port 8000 is in use by another app. Quit that app, or choose another port in Launch Options › Advanced.")
        #expect(problem(failure("upgrade"))
            == "Splash is being updated. Wait for the update to finish, then start it again.")
        #expect(problem(failure("install", message: "a/b is not installed"))
            == "Splash couldn’t prepare its files: a/b is not installed")
        #expect(problem(failure("model", message: "no supported model has this architecture (Llama)"))
            == "Splash can’t serve this model: no supported model has this architecture (Llama)")
        #expect(problem(failure("internal"))
            == "It stopped: the engine's words. The log has the details.")
        // The rest in the engine's words, as its refusal of a MAX_CONTEXT past the model's.
        for kind in ["configuration", "memory_budget", "device"] {
            #expect(problem(failure(kind)) == "the engine's words")
        }
    }

    /// A start that failed for want of memory offers the start that would fit: under a memory
    /// limit, the least limit the options offer that holds what it needs, or none where none does;
    /// with a context length set, Auto. Other failures offer none.
    @Test func aFailureForWantOfMemoryOffersTheStartThatWouldFit() {
        func failure(_ kind: String, _ bound: String?, required: Int64? = nil) -> ServeRecord.Failure {
            ServeRecord.Failure(kind: kind, message: "", bound: bound, required_bytes: required, budget_bytes: nil,
                                plan_context_tokens: nil)
        }
        func remedy(_ failure: ServeRecord.Failure, _ settings: ServeSettings) -> Server.Remedy? {
            Server.remedy(for: failure, settings: settings, memory: 36)
        }
        var settings = ServeSettings()
        settings["--max-memory"] = "8G"
        #expect(remedy(failure("memory_budget", "max_memory", required: 10_858_659_840), settings) == .memoryLimit("12G"))
        #expect(remedy(failure("memory_budget", "max_memory", required: 34_359_738_369), settings) == .memoryLimit(nil))
        #expect(remedy(failure("memory_budget", "device_memory", required: 10_858_659_840), settings) == nil)
        #expect(remedy(failure("context_budget", "max_memory"), settings) == nil)
        settings["--max-context"] = String(256 * 1024)
        #expect(remedy(failure("context_budget", "disk"), settings) == .autoContext)
        #expect(remedy(failure("host_memory", nil), settings) == nil)
        #expect([Server.Remedy.memoryLimit("12G"), .memoryLimit(nil), .autoContext].map(\.title)
            == ["Start with 12 GB", "Start Without a Limit", "Start with Auto Context"])
    }

    /// Model files on this Mac that Splash couldn't prepare or serve offer the start without the
    /// projector or draft chosen for them, where it passed either, as either may be what failed;
    /// without, taking them off the list.
    @Test func aFailureOfModelFilesOffersTheStartWithoutWhatWasChosenForThem() {
        let failure = ServeRecord.Failure(kind: "model", message: "", bound: nil, required_bytes: nil, budget_bytes: nil,
                                          plan_context_tokens: nil)
        func remedy(projector: String? = nil, draft: String? = nil, textOnly: Bool = false) -> Server.Remedy? {
            var settings = ServeSettings()
            settings.projector = projector
            settings.draft = draft
            if textOnly { settings["--language-only"] = "true" }
            return Server.remedy(for: failure, settings: settings, memory: 36, local: true)
        }
        #expect(remedy(projector: "/m/mmproj.gguf") == .without(projector: true, draft: false))
        #expect(remedy(draft: "/m/draft") == .without(projector: false, draft: true))
        #expect(remedy(projector: "/m/mmproj.gguf", draft: "/m/draft") == .without(projector: true, draft: true))
        // A projector Text only left out of the start had no part in its failure.
        #expect(remedy(projector: "/m/mmproj.gguf", textOnly: true) == .removeFromList)
        #expect(remedy() == .removeFromList)
        #expect(Server.remedy(for: failure, settings: ServeSettings(), memory: 36) == nil)
        #expect([Server.Remedy.without(projector: true, draft: false), .without(projector: false, draft: true),
                 .without(projector: true, draft: true), .removeFromList].map(\.title)
            == ["Start Without Projector", "Start Without Draft", "Start Without Projector and Draft", "Remove from List"])
    }

    @Test func keepsTheOptionsAndReadsBadOnesAsDefaults() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-options-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let defaults = UserDefaults.kept(in: folder)
        #expect(ServeSettings.current(defaults).values.isEmpty)
        let settings = ServeSettings(values: ["--port": "9000", "--max-context": "100000", "--max-memory": "20G",
                                              "--allowed-host": "mymac.local", "--allow-idle-sleep": "true"])
        settings.save(defaults)
        #expect(ServeSettings.current(defaults) == settings)
        defaults.set(["--port": "70000", "--max-context": "-1", "--max-memory": "lots", "--nope": "1",
                      "--kv-format": "bf16"], forKey: Preference.launchOptions)
        #expect(ServeSettings.current(defaults).values == ["--kv-format": "bf16"])
    }

    @Test func typedValuesTakeServesForms() throws {
        func custom(_ flag: String) throws -> LaunchOption.Value {
            let option = try #require(LaunchOption.all.first { $0.flag == flag })
            switch option.kind {
            case .choice(_, _, let custom?): return custom
            case .text(let custom, _): return custom
            default: throw CancellationError()
            }
        }
        #expect(try custom("--max-context").parse("100K") == "102400")
        #expect(try custom("--max-memory").parse("40") == "40G")
        #expect(try custom("--expert-memory").parse("2.5 g") == "2.5G")
        #expect(try custom("--expert-memory").parse("lots") == nil)
        #expect(try custom("--idle-release").parse("45M") == "45m")
        #expect(try custom("--port").parse("0") == nil)
        #expect(try custom("--served-model-name").parse("gpt-4o,  o3 ") == "gpt-4o, o3")
        // A value typed in is one the option takes, as kept and read back: every example, and the
        // menu's choices.
        for option in LaunchOption.all {
            switch option.kind {
            case .choice(_, let choices, let custom):
                #expect(choices.allSatisfy { option.accepts($0.value) }, "\(option.flag)")
                if let custom, custom.prompt != "None" {
                    #expect(custom.parse(custom.prompt).map(option.accepts) == true, "\(option.flag)")
                }
            case .text(let custom, _) where custom.prompt != "None":
                #expect(custom.parse(custom.prompt).map(option.accepts) == true, "\(option.flag)")
            default: break
            }
        }
    }

    @Test func warnsOfAServerOthersCanUse() throws {
        let host = try #require(LaunchOption.all.first { $0.flag == "--host" })
        #expect(host.warning(ServeSettings()) == nil)
        #expect(host.warning(ServeSettings(values: ["--host": "0.0.0.0"])) != nil)
        #expect(host.warning(ServeSettings(values: ["--host": "0.0.0.0"], apiKey: "k")) == nil)
        let origins = try #require(LaunchOption.all.first { $0.flag == "--allowed-origin" })
        #expect(origins.warning(ServeSettings(values: ["--allowed-origin": "a.example, *"])) != nil)
        #expect(origins.warning(ServeSettings(values: ["--allowed-origin": "a.example"])) == nil)
    }

    /// Whether this session can write the login Keychain, which is locked over SSH.
    static let keychainWritable: Bool = {
        let keychain = Keychain(service: "Splash Tests \(UUID().uuidString)")
        guard (try? keychain.write("probe")) != nil else { return false }
        try? keychain.write(nil)
        return true
    }()

    @Test(.enabled(if: keychainWritable, "the login Keychain is locked, as over SSH"))
    func keepsTheKeyInTheKeychain() throws {
        let keychain = Keychain(service: "Splash Tests \(UUID().uuidString)")
        defer { try? keychain.write(nil) }
        #expect(try keychain.read() == nil)
        try keychain.write("first")
        try keychain.write("second")
        #expect(try keychain.read() == "second")
        try keychain.write("")
        #expect(try keychain.read() == nil)
        try keychain.write(nil)
    }
}

@MainActor
struct AppStateTests {
    @Test(.enabled(if: ServeSettingsTests.keychainWritable, "the login Keychain is locked, as over SSH"))
    func keepsTheKeyTheOptionsHave() throws {
        let keychain = Keychain(service: "Splash Tests \(UUID().uuidString)")
        defer { try? keychain.write(nil) }
        let root = FileManager.default.temporaryDirectory.appending(path: "splash-state-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try keychain.write("first")
        let state = AppState(runtime: Runtime(root: root), folders: .init(in: root.appending(path: "Chats")), keychain: keychain,
                             defaults: .kept(in: root), shell: { .bare })
        #expect(state.options.apiKey == "first" && state.keyProblem == nil)
        // Kept once typed, not at each key.
        state.options.apiKey = "second"
        #expect(try keychain.read() == "first")
        state.keepKey()
        #expect(try keychain.read() == "second")
        // A key changed elsewhere is read again.
        try keychain.write("third")
        state.readKey()
        #expect(state.options.apiKey == "third")
        state.options.apiKey = nil
        state.keepKey()
        #expect(try keychain.read() == nil)
    }

    /// What model files on this Mac start with besides is kept as it is set, joins the settings
    /// their start uses, and goes with them off the list.
    @Test func keepsWhatModelFilesStartWith() throws {
        let fake = try FakeRuntime("splash-model-files", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        state.modelFiles["/m/q.gguf", default: AppState.ModelFiles()].projector = "/m/mmproj.gguf"
        state.modelFiles["/m/q.gguf", default: AppState.ModelFiles()].draft = "/m/draft"
        let again = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { again.quit() }
        #expect(again.modelFiles["/m/q.gguf"] == AppState.ModelFiles(projector: "/m/mmproj.gguf", draft: "/m/draft"))
        #expect(state.settings(for: "/m/q.gguf").projector == "/m/mmproj.gguf" && state.settings(for: "a/b").projector == nil)
        state.modelFiles["/m/q.gguf"]?.projector = nil
        #expect(state.modelFiles["/m/q.gguf"] == AppState.ModelFiles(draft: "/m/draft"))
        state.removeLocal("/m/q.gguf")
        #expect(state.modelFiles.isEmpty)
    }

    /// A check for updates says what the installer reports it found last: that the model is up
    /// to date or updated, or that it couldn't be checked or updated, whatever downloaded; nothing
    /// where the installer reported nothing.
    @Test func aCheckForUpdatesSaysWhatItFound() async throws {
        let fake = try FakeRuntime("splash-check", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        let installer = fake.root.appending(path: "install/models.py")
        try "import sys\n".write(to: installer, atomically: true, encoding: .utf8)
        state.checkForUpdates("a/b")
        #expect(state.installer.running && state.installer.work == .check)
        try await waitFor(!state.installer.running, "the check")
        #expect(state.installer.finished == nil)
        let outcomes: [(String, Installer.Outcome)] = [
            ("up_to_date", .upToDate), ("updated", .updated), ("unreachable", .unreachable), ("kept", .kept),
        ]
        for (name, outcome) in outcomes {
            try """
            print('{"event": "download", "repo": "a/b-draft", "done": 1, "total": 2}', flush=True)
            print('{"event": "prepared", "outcome": "\(name)"}', flush=True)
            """.write(to: installer, atomically: true, encoding: .utf8)
            state.checkForUpdates("a/b")
            #expect(state.installer.finished == nil)
            try await waitFor(!state.installer.running, "the check")
            #expect(state.installer.finished?.model == "a/b" && state.installer.finished?.result == .checked(outcome))
        }
        state.installer.dismissFinished()
        #expect(state.installer.finished == nil)
        // A download is no check.
        state.download("a/b")
        #expect(state.installer.work == .download)
        try await waitFor(!state.installer.running, "the download")
        #expect(state.installer.finished == nil)
    }

    /// Deleting a model runs the installer's removal, says what it freed, and lists the models
    /// again; a removal the installer refuses says why, and lists them again too, as one that
    /// failed may have removed the model all the same.
    @Test func deletingAModelRunsTheInstallersRemoval() async throws {
        let fake = try FakeRuntime("splash-delete", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        try fake.install("a/b")
        try fake.install("c/d")
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        await state.listed()
        #expect(state.isInstalled("a/b"))
        // An installer whose removal takes the model's installation away, as `remove` does.
        let installer = fake.root.appending(path: "install/models.py")
        let models = Runtime(root: fake.root).models
        try """
        import json, shutil, sys
        if sys.argv[-1] != "remove" or sys.argv[-2] != "a/b":
            sys.exit(2)
        shutil.rmtree("\(models.appending(path: "a/b").path)")
        print(json.dumps({"event": "removed", "bytes": 1}), flush=True)
        """.write(to: installer, atomically: true, encoding: .utf8)
        // Not while the launch still looks through the Hugging Face cache for models.
        if state.findingCachedModels { #expect(!state.canDelete("a/b")) }
        try await waitFor(!state.findingCachedModels, "the look through the cache")
        #expect(state.canDelete("a/b"))
        state.delete("a/b")
        #expect(state.installer.running && state.installer.work == .removal)
        try await waitFor(!state.installer.running, "the removal")
        try await waitFor(!state.isInstalled("a/b"), "the list")
        #expect(state.installer.failure == nil)
        #expect(state.installer.finished?.model == "a/b" && state.installer.finished?.result == .deleted(freed: 1))
        try """
        import json, shutil
        shutil.rmtree("\(models.appending(path: "c/d").path)")
        print(json.dumps({"event": "error", "kind": None, "message": "c/d is being served: stop it, then remove it"}), flush=True)
        raise SystemExit(1)
        """.write(to: installer, atomically: true, encoding: .utf8)
        #expect(state.isInstalled("c/d"))
        state.installer.remove("c/d")
        #expect(state.installer.finished == nil)
        try await waitFor(!state.installer.running, "the refusal")
        #expect(state.installer.failure?.message == "c/d is being served: stop it, then remove it")
        try await waitFor(!state.isInstalled("c/d"), "the list")
    }

    /// The server's status is asked less often while neither the window nor the menu bar's panel
    /// shows it.
    @Test func theStatusIsAskedLessOftenWhileNothingShowsIt() throws {
        let fake = try FakeRuntime("splash-watched", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        #expect(!state.server.unwatched)
        state.windowShows = false
        #expect(state.server.unwatched)
        state.panelShows = true
        #expect(!state.server.unwatched)
        state.panelShows = false
        state.windowShows = true
        #expect(!state.server.unwatched)
    }

    /// A model whose files are gone, as when its Hugging Face cache folder was deleted, is not on
    /// this Mac: it leaves the list, and the pick of it falls to the first model listed, as for a
    /// model taken off the list. Its files back, as installed again, it is listed and picked again.
    @Test func listsOnlyInstallationsWhoseFilesAreOnThisMac() async throws {
        let manager = FileManager.default
        let folder = manager.temporaryDirectory.appending(path: "splash-installed-\(UUID().uuidString)")
        defer { try? manager.removeItem(at: folder) }
        let runtime = Runtime(root: folder.appending(path: "Runtime"))
        let snapshot = folder.appending(path: "hub/models--owner--repo/snapshots/abc")
        let metadata = runtime.models.appending(path: ".metadata/key/config.json")
        // An installation as the installer builds it: an assembly linking the files its record
        // names, which a selection link names, its files a snapshot's and derived metadata.
        func install(_ variant: String, _ files: [String: URL]) throws {
            let assembly = runtime.models.appending(path: ".resolved/\(variant)")
            var record: [String: [String: Any]] = [:]
            for (name, file) in files {
                try manager.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data(count: 1000).write(to: file)
                let link = assembly.appending(path: name)
                try manager.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
                try manager.createSymbolicLink(at: link, withDestinationURL: file)
                record[name] = ["path": file.path, "bytes": 1000]
            }
            try JSONSerialization.data(withJSONObject: ["model": "owner/repo:\(variant)", "files": record])
                .write(to: assembly.appending(path: "model.json"))
            try manager.createDirectory(at: runtime.models.appending(path: "owner"), withIntermediateDirectories: true)
            try manager.createSymbolicLink(at: runtime.models.appending(path: "owner/repo:\(variant)"), withDestinationURL: assembly)
        }
        let q2 = snapshot.appending(path: "Q2_0/m-00001-of-00001.gguf")
        try install("IQ3_S", ["target/m-00001-of-00002.gguf": snapshot.appending(path: "IQ3_S/m-00001-of-00002.gguf"),
                              "target/m-00002-of-00002.gguf": snapshot.appending(path: "IQ3_S/m-00002-of-00002.gguf"),
                              "config.json": metadata])
        try install("Q2_0", ["target/m-00001-of-00001.gguf": q2, "config.json": metadata])
        #expect(runtime.installedModels() == [InstalledModel(id: "owner/repo:IQ3_S", bytes: 3000),
                                              InstalledModel(id: "owner/repo:Q2_0", bytes: 2000)])
        let state = AppState.forTests(in: folder, runtime: runtime)
        await state.listed()
        state.selection = "owner/repo:Q2_0"
        #expect(state.selection == "owner/repo:Q2_0")

        try manager.removeItem(at: q2.deletingLastPathComponent())
        #expect(runtime.installedModels() == [InstalledModel(id: "owner/repo:IQ3_S", bytes: 3000)])
        await state.refresh().value
        #expect(state.installed.map(\.id) == ["owner/repo:IQ3_S"])
        #expect(state.selection == "owner/repo:IQ3_S")
        // Nothing of the installation is deleted.
        #expect(manager.fileExists(atPath: runtime.models.appending(path: "owner/repo:Q2_0/model.json").path))

        try manager.createDirectory(at: q2.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: 1000).write(to: q2)
        await state.refresh().value
        #expect(state.installed.map(\.id) == ["owner/repo:IQ3_S", "owner/repo:Q2_0"])
        #expect(state.selection == "owner/repo:Q2_0")
    }

    /// Nothing starts before the app is ready or once it quits (`canStart`). The menu bar's panel
    /// checks the model that starts or runs, then the one a switch goes to, and none once it
    /// failed, whether its server ended or runs on, or while it stops; choosing another starts or
    /// switches to it, choosing the failed one starts it again, and the one checked goes on as it is.
    @Test func thePanelStartsTheModelChosen() async throws {
        let fake = try FakeRuntime("splash-menus", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let first = "a/First-GGUF:Q4", second = "a/Second-GGUF:Q4"
        try fake.install(first)
        try fake.install(second)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        func step(_ name: String) { FileManager.default.createFile(atPath: fake.root.appending(path: name).path, contents: nil) }
        #expect(!state.canStart(first))
        state.choose(first)
        #expect(!state.server.running)
        await state.launch()
        #expect(state.canStart(first) && state.checkedModel == nil)
        state.options["--port"] = String(fake.port)
        // Refused by the engine, the start fails and stops: nothing is checked, and choosing the
        // model again starts it again.
        step("refused")
        step("loaded")
        state.choose(first)
        try await waitFor(state.serverStatus.look == .failed && !state.server.running, "the refusal")
        #expect(state.checkedModel == nil)
        try FileManager.default.removeItem(at: fake.root.appending(path: "refused"))
        step("warm")
        state.choose(first)
        #expect(state.checkedModel == first && state.selection == first)
        try await waitFor(state.server.phase == .ready, "\(first) ready")
        #expect(!state.canDelete(first))
        let since = state.server.since
        state.choose(first)
        #expect(state.server.phase == .ready && state.server.since == since)
        state.choose(second)
        #expect(state.checkedModel == second && state.selection == second)
        try await waitFor(state.server.model == second && state.server.phase == .ready, "\(second) ready")
        #expect(state.server.since > since)
        // Its engine stopped, its server stops too, as it no longer starts the engine again: nothing
        // is checked, and choosing the model starts it again.
        step("broken")
        try await waitFor(state.serverStatus.look == .failed && !state.server.running, "the engine's stop")
        #expect(state.checkedModel == nil)
        try FileManager.default.removeItem(at: fake.root.appending(path: "broken"))
        let broken = state.server.since
        state.choose(second)
        try await waitFor(state.server.phase == .ready && state.server.since > broken, "\(second) ready again")
        state.stop()
        #expect(state.checkedModel == nil)
        state.quit()
        #expect(!state.canStart(first))
        state.choose(first)
        #expect(state.server.next == nil)
    }
}

struct ServerTests {
    @Test func readsTheStatus() throws {
        let golden = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../../dev/tests/engine/status_golden.json")
        let now = Date(timeIntervalSince1970: 1_000)
        let report = try #require(Server.report(try Data(contentsOf: golden), at: now))
        #expect(report == Server.Report(activity: Server.Activity(served: 2, running: 2, generated: 32, read: 4096, cached: 64,
                                                                  memory: 4_294_967_296),
                                        asleep: false,
                                        live: [LiveRequest(id: 7, time: LiveRequest.time(of: now - 1.5005), input: 4096, stage: .reading(2048)),
                                               LiveRequest(id: 9, time: LiveRequest.time(of: now - 2.25025), input: 512,
                                                           stage: .answering(40))],
                                        taken: now))
        let busy = Server.report(Data(#"{"requests": {"completed": 5}, "metrics": {"decode_tokens_per_second": 71.5, "prefill_tokens_per_second": 900, "ttft_ms": {"p50": 120}}, "weights": {"released": true}}"#.utf8))
        #expect(busy == Server.Report(activity: Server.Activity(served: 5, generation: 71.5, reading: 900, firstToken: .milliseconds(120)),
                                      asleep: true))
        // While the server starts a failed engine again.
        #expect(Server.report(Data(#"{"transport": {"ready": false, "recovering": true, "stopped": false}}"#.utf8))?.recovering == true)
        #expect(Server.report(Data("not json".utf8)) == nil)
    }

    /// What a running request does, as the engine's phase gives it, and when it came; one without
    /// the engine's id, which nothing could tell from another, counts but has no row.
    @Test func readsTheRequestsRunning() throws {
        let now = Date(timeIntervalSince1970: 1_000)
        let status = Data(#"""
            {"active_requests": [{"id": 1, "phase": "queued", "prompt_tokens": 10, "age_ms": 500},
              {"id": 2, "phase": "waiting_resources", "prompt_tokens": 20}, {"id": 3, "phase": "waiting_prefix", "prompt_tokens": 30},
              {"id": 4, "phase": "prefill", "prompt_tokens": 40, "prompt_processed": 16},
              {"id": 5, "phase": "waiting_mask", "prompt_tokens": 50, "prompt_processed": 50, "generated_tokens": 7},
              {"phase": "decode", "prompt_tokens": 60, "generated_tokens": 1}]}
            """#.utf8)
        let report = try #require(Server.report(status, at: now))
        #expect(report.live.map(\.stage) == [.waiting("Queued"), .waiting("Waiting for memory"), .waiting("Waiting for a shared prefix"),
                                             .reading(16), .answering(7)])
        #expect(report.live.map(\.id) == [1, 2, 3, 4, 5])
        #expect(report.live.map(\.input) == [10, 20, 30, 40, 50] && report.live[0].time == LiveRequest.time(of: now - 0.5))
        #expect(report.activity.running == 6)
    }

    /// A status the server answered from an earlier snapshot, as while a long prefill keeps the
    /// engine from answering, says how old it is: the requests in it were so, and came, that much
    /// earlier. One that lists no requests says nothing of them.
    @Test func readsWhenTheStatusWasTaken() throws {
        let now = Date(timeIntervalSince1970: 1_000)
        let request = #"{"id": 1, "phase": "decode", "prompt_tokens": 10, "generated_tokens": 3, "age_ms": 500}"#
        let fresh = try #require(Server.report(Data(#"""
            {"active_requests": [\#(request)], "transport": {"status_stale": false, "status_age_ms": 0}}
            """#.utf8), at: now))
        #expect(fresh.taken == now && fresh.live.map(\.time) == [LiveRequest.time(of: now - 0.5)])
        let stale = try #require(Server.report(Data(#"""
            {"active_requests": [\#(request)], "transport": {"status_stale": true, "status_age_ms": 4000}}
            """#.utf8), at: now))
        #expect(stale.taken == now - 4 && stale.live.map(\.time) == [LiveRequest.time(of: now - 4.5)])
        // As the server answers before its engine's first status.
        let none = try #require(Server.report(Data(#"{"ready": false, "transport": {"status_stale": true, "status_age_ms": 0}}"#.utf8),
                                              at: now))
        #expect(none.taken == nil && none.live.isEmpty)
        #expect(Server.report(Data(#"{"requests": {"completed": 5}}"#.utf8), at: now)?.taken == nil)
    }

    /// Each request the engine took is one row, in the order they came: its start or the status
    /// adds it, the status moves it on, its end ends it in its place, and a status from before its
    /// end leaves it ended; a line that names no request is a row of its own.
    @Test func keepsOneRowPerRequest() throws {
        func line(_ text: String) throws -> RequestEntry { try #require(RequestEntry(text)) }
        func live(_ id: Int, _ stage: LiveRequest.Stage, input: Int) -> LiveRequest {
            LiveRequest(id: id, time: "12:00:00", input: input, stage: stage)
        }
        let start = Date(timeIntervalSince1970: 1_000)
        func at(_ seconds: Double) -> Date { start + seconds }
        var rows = ServedRequest.taking(try line("12:00:01 Started · request 1 · input 881 · cached 0"), at: at(1), into: [])
        rows = ServedRequest.taking([live(1, .answering(52), input: 881), live(2, .reading(6656), input: 8193)], taken: at(2),
                                    into: rows)
        #expect(rows.map(\.id) == [1, 2] && rows.map(\.time) == ["12:00:01", "12:00:00"])
        #expect(rows.map(\.stage) == [.answering(52), .reading(6656)] && rows.allSatisfy(\.running))
        rows = ServedRequest.taking(try line("12:00:05 Done · request 1 · input 881 · cached 0 · output 65 · TTFT 1.6s · 32.7 tok/s"),
                                    at: at(5), into: rows)
        // A status from before that end and the last one, as one the server answers from an
        // earlier snapshot.
        rows = ServedRequest.taking([live(1, .answering(52), input: 881), live(2, .reading(4096), input: 8193)], taken: at(1.5),
                                    into: rows)
        #expect(rows.count == 2 && rows[0].end?.output == 65 && rows[1].running && rows[1].stage == .reading(6656))
        // Its Started line, later than the status that had it first, moves it on no further back.
        rows = ServedRequest.taking(try line("12:00:02 Started · request 2 · input 8,193 · cached 0"), at: at(6), into: rows)
        #expect(rows.map(\.time) == ["12:00:01", "12:00:00"] && rows[1].stage == .reading(6656))
        rows = ServedRequest.taking(try line("12:00:06 Error · not_found · HEAD /api/hello"), at: at(6), into: rows)
        rows = ServedRequest.taking(try line("12:00:09 Done · request 2 · input 8,193 · cached 0 · output 33"), at: at(9), into: rows)
        #expect(rows.map(\.id) == [1, 2, -1] && rows.allSatisfy { $0.end != nil })
        #expect(rows[2].end?.message == "not_found · HEAD /api/hello" && rows[2].time == "12:00:06")
        // A line that names no request: an end is a row of its own, a start none, as nothing could end it.
        let earlier = ServedRequest.taking(try line("12:01:00 Started · input 5 · cached 0"), at: at(60), into: [])
        #expect(earlier.isEmpty)
        #expect(ServedRequest.taking(try line("12:01:02 Done · input 5 · cached 0 · output 1"), at: at(62), into: earlier).map(\.id) == [-1])
    }

    /// Each status says where each running request seen before its snapshot was taken is, back
    /// too, as the engine suspends a request for memory and later reads its prompt again, and that
    /// one it no longer has has ended, its line not come; it adds the requests it has first. One
    /// from a snapshot older than a row says nothing of it, as a status asked while a line starts
    /// a request; a row ended stays ended; one that lists no requests says nothing.
    @Test func aStatusSaysWhereEachRequestSeenBeforeItIs() throws {
        func line(_ text: String) throws -> RequestEntry { try #require(RequestEntry(text)) }
        func live(_ id: Int, _ stage: LiveRequest.Stage) -> LiveRequest { LiveRequest(id: id, time: "12:00:00", input: 900, stage: stage) }
        let start = Date(timeIntervalSince1970: 1_000)
        func at(_ seconds: Double) -> Date { start + seconds }
        var rows = ServedRequest.taking([live(1, .answering(52))], taken: at(1), into: [])
        rows = ServedRequest.taking([live(1, .waiting("Waiting for memory"))], taken: at(3), into: rows)
        #expect(rows.map(\.stage) == [.waiting("Waiting for memory")])
        rows = ServedRequest.taking([live(1, .reading(300))], taken: at(5), into: rows)
        #expect(rows.map(\.stage) == [.reading(300)])
        // From a snapshot older than that: a request first seen is added, the other stays.
        rows = ServedRequest.taking([live(1, .answering(52)), live(2, .waiting("Queued"))], taken: at(2), into: rows)
        #expect(rows.map(\.id) == [1, 2] && rows.map(\.stage) == [.reading(300), .waiting("Queued")])
        // From a snapshot newer than both, as the server answers through a long prefill.
        rows = ServedRequest.taking([live(1, .reading(600)), live(2, .waiting("Queued"))], taken: at(6), into: rows)
        #expect(rows.map(\.stage) == [.reading(600), .waiting("Queued")])
        // Request 2's and request 3's starts come as a status is asked, whose snapshot is older.
        rows = ServedRequest.taking(try line("12:00:07 Started · request 2 · input 900 · cached 0"), at: at(7.1), into: rows)
        rows = ServedRequest.taking(try line("12:00:07 Started · request 3 · input 900 · cached 0"), at: at(7.2), into: rows)
        rows = ServedRequest.taking([live(2, .waiting("Queued"))], taken: at(7), into: rows)
        #expect(rows.map(\.id) == [1, 2, 3] && rows.map(\.running) == [false, true, true])
        #expect(rows[0].gone && rows[0].end == nil && rows[1].stage == .reading(0) && rows[2].stage == .reading(0))
        // Its line, come at last, ends it all the same; a status that has it runs it no more.
        rows = ServedRequest.taking(try line("12:00:08 Done · request 1 · input 900 · cached 0 · output 70"), at: at(8), into: rows)
        #expect(rows[0].end?.output == 70)
        rows = ServedRequest.taking([live(1, .answering(70)), live(2, .reading(10)), live(3, .reading(20))], taken: at(9), into: rows)
        #expect(rows.map(\.running) == [false, true, true] && rows.map(\.stage) == [.reading(600), .reading(10), .reading(20)])
        #expect(ServedRequest.taking([], taken: nil, into: rows) == rows)
    }

    /// The request lines the server writes, as the golden the server's tests check them against
    /// holds them.
    @Test func readsTheRequestLinesTheServerWrites() throws {
        let golden = try String(contentsOf: ServerTests.goldens.appending(path: "server/log_lines_golden.txt"), encoding: .utf8)
        let lines = golden.split(separator: "\n").map(String.init).filter { !$0.hasPrefix("#") }
        try #require(lines.count == 5)
        let started = try #require(RequestEntry(lines[0]))
        #expect(started.outcome == .started && started.request == 7 && (started.input, started.cached) == (4096, 1024))
        let done = try #require(RequestEntry(lines[1]))
        #expect(done.outcome == .done && done.request == 7 && (done.input, done.cached, done.output, done.firstToken, done.rate)
            == (4096, 1024, 574, 0.4, 36.8))
        let cancelled = try #require(RequestEntry(lines[2]))
        #expect(cancelled.outcome == .cancelled && cancelled.request == 8 && cancelled.output == 12)
        let failed = try #require(RequestEntry(lines[3]))
        #expect(failed.outcome == .error && failed.request == 9 && failed.message == "runtime_unavailable")
        let refused = try #require(RequestEntry(lines[4]))
        #expect(refused.outcome == .error && refused.request == nil && refused.message == "invalid_request_error")
    }

    /// The goldens the server's tests write, as the app reads them.
    static let goldens = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../../dev/tests")

    @Test func readsAStoppedServersStatus() throws {
        let golden = Self.goldens.appending(path: "server/status_transport_golden.json")
        let report = try #require(Server.report(try Data(contentsOf: golden)))
        #expect(report.context == 131072)
        #expect(!report.recovering)
    }

    @Test func readsTheServeRecord() throws {
        let golden = Self.goldens.appending(path: "server/serve_record_golden.json")
        let record = try JSONDecoder().decode(ServeRecord.self, from: Data(contentsOf: golden))
        #expect(record.pid == 4242 && record.phase == .failed)
        let failure = try #require(record.failure)
        #expect(Server.problem(failure, port: 8000)
            == "It needs 10.1 GB of memory but the memory limit is 1 GB. Raise the memory limit in Launch Options.")
        // A warm-up, with its steps done of all.
        let warming = try JSONDecoder().decode(ServeRecord.self, from: Data(contentsOf: Self.goldens.appending(
            path: "server/serve_record_warming_golden.json")))
        #expect(warming.phase == .warming && warming.warmup == ServeRecord.Warmup(steps_done: 3, steps: 9))
    }

    @Test func wordsAnExitByItsOwnRecord() throws {
        func golden(_ name: String) throws -> ServeRecord {
            try JSONDecoder().decode(ServeRecord.self, from: Data(contentsOf: Self.goldens.appending(path: "server/\(name)")))
        }
        let record = try golden("serve_record_golden.json")
        let words = "It needs 10.1 GB of memory but the memory limit is 1 GB. Raise the memory limit in Launch Options."
        func problem(_ record: ServeRecord?, pid: pid_t, holder: pid_t? = nil) -> String {
            Server.exitProblem(record: record, pid: pid, port: 8000, holder: holder)
        }
        #expect(problem(record, pid: 4242) == words)
        // A ready server that stopped restarting its engine, in the server's own words.
        let stopped = try golden("serve_record_stopped_golden.json")
        #expect(problem(stopped, pid: 4242) == stopped.failure?.message)
        // A record of another start, as one a server before this one wrote, or none.
        let unknown = "It stopped unexpectedly. The log has the details."
        #expect(problem(record, pid: 4243) == unknown)
        #expect(problem(nil, pid: 4242) == unknown)
        // A start, or a ready server, that crashed or was killed.
        for phase in ["starting", "ready"] {
            let unsaid = try JSONDecoder().decode(ServeRecord.self, from: Data(#"{"pid":4242,"phase":"\#(phase)"}"#.utf8))
            #expect(problem(unsaid, pid: 4242) == unknown)
        }
        let starting = try JSONDecoder().decode(ServeRecord.self, from: Data(#"{"pid":4242,"phase":"starting"}"#.utf8))
        // Refused the port's lock, which another server holds, with its record.
        #expect(problem(starting, pid: 4243, holder: 4242)
            == "Another Splash server is using port 8000. Stop it, or choose another port in Launch Options › Advanced.")
        // Its own lock, which an engine it started may hold a moment longer.
        #expect(problem(nil, pid: 4242, holder: 4242) == unknown)
    }

    @Test func findsTheProcessHoldingAPortsLock() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "splash-lock-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = Runtime(root: root)
        try FileManager.default.createDirectory(at: runtime.locks, withIntermediateDirectories: true)
        let lock = runtime.locks.appending(path: "serve-8123.lock")
        // What a launcher that held the lock left there.
        try Data(#"{"pid":4242,"model":"a/b","port":8123}"#.utf8).write(to: lock)
        #expect(runtime.serveLockHolder(port: 8123) == nil)
        let held = open(lock.path, O_RDONLY)
        defer { close(held) }
        #expect(flock(held, LOCK_EX | LOCK_NB) == 0)
        #expect(runtime.serveLockHolder(port: 8123) == 4242)
    }

    /// The app gives a server the engine's stop budget twice, the longest the server's own stop of
    /// its engine takes, as the golden both sides read holds it.
    @Test func appStopBudgetIsTheEngines() throws {
        let golden = try String(contentsOf: Self.goldens.appending(path: "engine/protocol_golden.txt"), encoding: .utf8)
        let line = try #require(golden.split(separator: "\n").first { $0.hasPrefix("supervision ") })
        let hex = Array(line.dropFirst("supervision ".count))
        let bytes = stride(from: 0, to: hex.count, by: 2).compactMap { UInt8(String(hex[$0...$0 + 1]), radix: 16) }
        // Little-endian u32 milliseconds.
        let budgets = stride(from: 0, to: bytes.count, by: 4).map { offset in
            bytes[offset..<offset + 4].enumerated().reduce(UInt32(0)) { $0 | UInt32($1.element) << (8 * $1.offset) }
        }
        // command timeout, loop stall, stop, probation, start retry window, mask wait.
        #expect(budgets.count == 6)
        #expect(Duration.milliseconds(budgets[2]) == Supervision.stop)
        #expect(Server.killAfter == Supervision.stop * 2)
    }

    /// A stop interrupts the server once, as Ctrl+C does, which stops its engine with its flush;
    /// a second interrupt would stop the engine at once. Only its kill budget later is the server
    /// killed.
    @MainActor @Test func aStopInterruptsOnceThenKills() async throws {
        // A checkout whose Python says its process ID, then each interrupt, and runs on.
        let root = FileManager.default.temporaryDirectory.appending(path: "splash-stop-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = Runtime(root: root)
        try FileManager.default.createDirectory(at: runtime.python.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "#!/bin/sh\necho $$\ntrap 'echo INT' INT\nwhile :; do sleep 0.1; done\n"
            .write(to: runtime.python, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: runtime.python.path)
        let log = root.appending(path: "server.log")
        let killAfter = Duration.seconds(1)
        let server = Server(runtime: runtime, log: log, defaults: .kept(in: root), killAfter: killAfter)
        server.start("a/b", settings: ServeSettings(values: ["--port": "0"]))
        // The start's own lines, after the line that names it.
        var lines: [Substring] { ((try? String(contentsOf: log, encoding: .utf8)) ?? "").split(separator: "\n").dropFirst().map { $0 } }
        try await waitFor(!lines.isEmpty, "the start")
        let pid = try #require(pid_t(lines[0]))
        defer { killpg(pid, SIGKILL) }
        let stopped = ContinuousClock.now
        server.stop()
        try await waitFor(!server.running, "the kill", seconds: 11)
        #expect(ContinuousClock.now - stopped >= killAfter)
        #expect(lines.filter { $0 == "INT" }.count == 1)
    }

    @MainActor @Test func givesTheServerTheAppsProcessID() async throws {
        // A checkout whose Python says what SPLASH_APP_PID it was given, and ends.
        let root = FileManager.default.temporaryDirectory.appending(path: "splash-server-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = Runtime(root: root)
        try FileManager.default.createDirectory(at: runtime.python.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "#!/bin/sh\necho \"SPLASH_APP_PID=$SPLASH_APP_PID\"\n".write(to: runtime.python, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: runtime.python.path)
        let log = root.appending(path: "server.log")
        let server = Server(runtime: runtime, log: log, defaults: .kept(in: root))
        // On a port nothing serves.
        server.start("a/b", settings: ServeSettings(values: ["--port": "0"]))
        try await waitFor(!server.running, "the end")
        let lines = try String(contentsOf: log, encoding: .utf8).split(separator: "\n")
        #expect(lines.count == 2 && lines[0].hasPrefix("── Starting a/b, ") && lines[1] == "SPLASH_APP_PID=\(getpid())")
    }
}

/// A program the app started, kept for the next launch: stopped then, whichever copy of the app
/// started it, and never a later process given its ID.
@MainActor
struct ChildRecordTests {
    @Test func stopsTheProgramItKeepsAndForgetsIt() async throws {
        let folder = Self.folder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let defaults = UserDefaults.kept(in: folder)
        let record = ChildRecord(defaults: defaults, key: "child", stop: SIGINT, killAfter: .seconds(20))
        let child = try ChildProcess(command: ["/bin/sleep", "60"], environment: [:], log: folder.appending(path: "child.log")) { _ in }
        #expect(child.started != nil)
        record.keep(child)
        var stopping = 0
        await record.stopLeftover(olderApps: { _ in false }) { stopping += 1 }
        #expect(processStart(child.pid) == nil)
        #expect(defaults.object(forKey: record.key) == nil)
        // The launch said it waited for the stop, once; with nothing left to stop, it doesn't.
        #expect(stopping == 1)
        await record.stopLeftover(olderApps: { _ in false }) { stopping += 1 }
        #expect(stopping == 1)
    }

    @Test func leavesAnotherProcessWithTheIDAlone() async throws {
        let folder = Self.folder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let defaults = UserDefaults.kept(in: folder)
        let record = ChildRecord(defaults: defaults, key: "child", stop: SIGINT, killAfter: .seconds(20))
        let other = try ChildProcess(command: ["/bin/sleep", "60"], environment: [:], log: folder.appending(path: "child.log")) { _ in }
        defer { other.stop(SIGKILL, killAfter: .zero) }
        // Kept with another start: an earlier process that had the ID, or a record an older app
        // kept as the ID alone.
        defaults.set([Int(other.pid), try #require(other.started) - 1], forKey: record.key)
        var stopping = false
        await record.stopLeftover(olderApps: { _ in false }) { stopping = true }
        #expect(processStart(other.pid) != nil && !stopping)
        #expect(defaults.object(forKey: record.key) == nil)
        defaults.set(Int(other.pid), forKey: record.key)
        await record.stopLeftover(olderApps: { _ in false })
        #expect(processStart(other.pid) != nil)
        #expect(defaults.object(forKey: record.key) == nil)
    }

    /// A server an app before 1.6 left running, which it kept as its ID alone beside its port, is
    /// stopped where it still holds that port's lock under the ID; a process the ID names that
    /// holds no such lock is left alone. Either way the record is forgotten.
    @Test func aServerAnOlderAppLeftIsStoppedByItsPortLock() async throws {
        let folder = Self.folder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let defaults = UserDefaults.kept(in: folder)
        let runtime = Runtime(root: folder)
        try FileManager.default.createDirectory(at: runtime.locks, withIntermediateDirectories: true)
        let port = 8123
        let lock = runtime.locks.appending(path: "serve-\(port).lock")
        // The older server: the process that holds its port's lock and records its ID there.
        let older = try ChildProcess(command: ["/usr/bin/lockf", "-k", lock.path, "/bin/sleep", "60"], environment: [:],
                                     log: folder.appending(path: "older.log")) { _ in }
        defer { older.stop(SIGKILL, killAfter: .zero) }
        try await waitFor(FileManager.default.fileExists(atPath: lock.path), "the lock")
        try Data("{\"pid\": \(older.pid)}".utf8).write(to: lock)
        try await waitFor(runtime.serveLockHolder(port: port) == older.pid, "the lock's holder")
        let other = try ChildProcess(command: ["/bin/sleep", "60"], environment: [:], log: folder.appending(path: "other.log")) { _ in }
        defer { other.stop(SIGKILL, killAfter: .zero) }
        let server = Server(runtime: runtime, log: URL(filePath: "/dev/null"), defaults: defaults)

        defaults.set(Int(other.pid), forKey: Preference.serverPID)
        defaults.set(port, forKey: Preference.olderServerPort)
        await server.stopLeftover()
        #expect(processStart(other.pid) != nil)
        #expect(processStart(older.pid) != nil)
        #expect(defaults.object(forKey: Preference.serverPID) == nil)

        defaults.set(Int(older.pid), forKey: Preference.serverPID)
        defaults.set(port, forKey: Preference.olderServerPort)
        await server.stopLeftover()
        #expect(processStart(older.pid) == nil)
        #expect(processStart(other.pid) != nil)
        #expect(defaults.object(forKey: Preference.serverPID) == nil)
        #expect(defaults.object(forKey: Preference.olderServerPort) == nil)
    }

    /// A download an app before 1.6 left running, which it kept as its ID alone, is stopped where
    /// the process the ID names still runs the installer as an app starts it; a process the ID
    /// names that runs anything else is left alone. Either way the record is forgotten.
    @Test func aDownloadAnOlderAppLeftIsStoppedByItsCommand() async throws {
        let folder = Self.folder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let defaults = UserDefaults.kept(in: folder)
        let runtime = Runtime(root: folder)
        try FileManager.default.createDirectory(at: folder.appending(path: "install"), withIntermediateDirectories: true)
        try "print('started')\nimport time\ntime.sleep(60)\n"
            .write(to: folder.appending(path: "install/models.py"), atomically: true, encoding: .utf8)
        // The older download: the runtime's installer as an app starts it, which here waits. The
        // system's Python can start another program in its place, under the same ID and arguments.
        let log = folder.appending(path: "older.log")
        let older = try ChildProcess(command: ["/usr/bin/python3"] + runtime.installCommand("a/b").dropFirst(),
                                     environment: ProcessInfo.processInfo.environment, log: log) { _ in }
        defer { older.stop(SIGKILL, killAfter: .zero) }
        try await waitFor((try? String(contentsOf: log, encoding: .utf8))?.hasSuffix("started\n") == true, "the installer")
        let other = try ChildProcess(command: ["/bin/sleep", "60"], environment: [:], log: folder.appending(path: "other.log")) { _ in }
        defer { other.stop(SIGKILL, killAfter: .zero) }
        let installer = Installer(runtime: runtime, log: URL(filePath: "/dev/null"), defaults: defaults)

        defaults.set(Int(other.pid), forKey: Preference.installerPID)
        await installer.stopLeftover()
        #expect(processStart(other.pid) != nil)
        #expect(processStart(older.pid) != nil)
        #expect(defaults.object(forKey: Preference.installerPID) == nil)

        defaults.set(Int(older.pid), forKey: Preference.installerPID)
        await installer.stopLeftover()
        #expect(processStart(older.pid) == nil)
        #expect(processStart(other.pid) != nil)
        #expect(defaults.object(forKey: Preference.installerPID) == nil)
    }

    /// A leftover server stops by itself once its app has ended, as SIGTERM stops it, and a
    /// Ctrl+C would then stop its engine at once: the next launch sends SIGTERM, which it ignores
    /// by then, and waits for it.
    @Test func aLeftoverServerAlreadyStoppingIsLeftToFinish() async throws {
        let log = try await Self.leftover("trap 'echo interrupted' INT; trap '' TERM; echo ready; sleep 1") {
            Server(runtime: Showcase.runtime, log: URL(filePath: "/dev/null"), defaults: $0).record
        }
        #expect(log == "ready\n")
    }

    /// A leftover download gets Ctrl+C, which removes its temporary files as it ends.
    @Test func aLeftoverDownloadIsInterrupted() async throws {
        let log = try await Self.leftover("trap 'echo interrupted; exit' INT; echo ready; sleep 60") {
            Installer(runtime: Showcase.runtime, log: URL(filePath: "/dev/null"), defaults: $0).record
        }
        #expect(log == "ready\ninterrupted\n")
    }

    /// A leftover that ignores its stop signal is killed, with its process group, once the record's
    /// delay has passed, and the next launch waits that and at most 10 s more. A server's record
    /// gives it as long as the server's own stop of its engine takes.
    @Test func aLeftoverIgnoringItsStopIsKilledAfterItsDelay() async throws {
        let folder = Self.folder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let defaults = UserDefaults.kept(in: folder)
        #expect(Server(runtime: Showcase.runtime, log: URL(filePath: "/dev/null"), defaults: defaults).record.killAfter == Server.killAfter)
        let killAfter = Duration.seconds(1)
        let record = ChildRecord(defaults: defaults, key: "child", stop: SIGTERM, killAfter: killAfter)
        let log = folder.appending(path: "child.log")
        // A shell that ignores SIGTERM, and a program of its group, which ignores it too, whose
        // process ID it writes.
        let child = try ChildProcess(command: ["/bin/sh", "-c", "trap '' TERM; sleep 60 & echo $!; wait"],
                                     environment: [:], log: log) { _ in }
        defer { child.stop(SIGKILL, killAfter: .zero) }
        var written: String { (try? String(contentsOf: log, encoding: .utf8)) ?? "" }
        try await waitFor(written.hasSuffix("\n"), "the start")
        let member = try #require(pid_t(written.trimmingCharacters(in: .whitespacesAndNewlines)))
        record.keep(child)
        let stopping = ContinuousClock.now
        await record.stopLeftover(olderApps: { _ in false })
        let took = ContinuousClock.now - stopping
        #expect(processStart(child.pid) == nil)
        #expect(took >= killAfter && took < killAfter + .seconds(10))
        // Once its parent is gone, launchd reaps it.
        let reaped = ContinuousClock.now + .seconds(2)
        while processStart(member) != nil, ContinuousClock.now < reaped {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(processStart(member) == nil)
    }

    /// What `script` wrote by the time `record`, made on a test's own defaults, has stopped it,
    /// once it has written `ready`.
    private static func leftover(_ script: String, record: (UserDefaults) -> ChildRecord) async throws -> String {
        let folder = Self.folder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let record = record(.kept(in: folder))
        let log = folder.appending(path: "child.log")
        let child = try ChildProcess(command: ["/bin/sh", "-c", script], environment: [:], log: log) { _ in }
        defer { child.stop(SIGKILL, killAfter: .zero) }
        try await waitFor((try? String(contentsOf: log, encoding: .utf8)) == "ready\n", "the start")
        record.keep(child)
        await record.stopLeftover(olderApps: { _ in false })
        #expect(processStart(child.pid) == nil)
        return try String(contentsOf: log, encoding: .utf8)
    }

    /// A folder of a test's own, for its record's defaults and its program's log, which
    /// `UserDefaults.kept(in:)` makes.
    private static func folder() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "splash-record-\(UUID().uuidString)")
    }
}

/// A program run to its end, as the engine's device check is.
@MainActor
struct ChildProcessRunTests {
    @Test func killsOneThatRunsPastItsTimeoutAndGivesNothing() async {
        let started = ContinuousClock.now
        #expect(await Self.run(["/bin/sh", "-c", "exec sleep 60"], timeout: .milliseconds(300)) == nil)
        #expect(ContinuousClock.now - started < .seconds(10))
    }

    @Test func killsOneItsCallerNoLongerWaitsFor() async {
        let started = ContinuousClock.now
        let run = Task { await Self.run(["/bin/sleep", "60"], timeout: .seconds(60)) }
        try? await Task.sleep(for: .milliseconds(300))
        run.cancel()
        #expect(await run.value == nil)
        #expect(ContinuousClock.now - started < .seconds(10))
    }

    private static func run(_ command: [String], timeout: Duration) async -> Int32? {
        let log = FileManager.default.temporaryDirectory.appending(path: "run-\(UUID().uuidString).log")
        defer { try? FileManager.default.removeItem(at: log) }
        return await ChildProcess.run(command, environment: [:], log: log, timeout: timeout)
    }

    @Test func theDeviceCheckGivesItsReasonOrPasses() async throws {
        // A checkout's engine, which runs the check.
        let root = FileManager.default.temporaryDirectory.appending(path: "splash-check-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = Runtime(root: root)
        try FileManager.default.createDirectory(at: runtime.engine.deletingLastPathComponent(), withIntermediateDirectories: true)
        func check(_ script: String) async throws -> String? {
            try "#!/bin/sh\n\(script)\n".write(to: runtime.engine, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: runtime.engine.path)
            return await runtime.unsupportedReason()
        }
        #expect(try await check("echo 'error: no Metal GPU' >&2; exit 1") == "no Metal GPU")
        #expect(try await check("exit 1") == nil)
        #expect(try await check("exit 0") == nil)
    }
}

/// Each installation's family, from its assembly record, and what each family makes of the launch
/// options, as the runtime reports it.
@MainActor
struct ModelFamilyTests {
    @Test func installationsHaveTheirRecordsFamily() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "splash-families-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = Runtime(root: root)
        func record(_ link: String, model: String, family: String) throws {
            let folder = runtime.models.appending(path: link)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try #"{"model": "\#(model)", "family": "\#(family)", "files": {}}"#
                .write(to: folder.appending(path: "model.json"), atomically: true, encoding: .utf8)
        }
        try record("a/Flint-GGUF:Q4", model: "a/Flint-GGUF:Q4", family: "Flint")
        try record(".selections/0123", model: "/models/27B.gguf", family: "Qwen3.8-27B")
        #expect(runtime.installedModels().map { [$0.id, $0.family] } == [["a/Flint-GGUF:Q4", "Flint"]])
        #expect(runtime.preparedFamilies() == ["/models/27B.gguf": "Qwen3.8-27B"])
    }

    @Test func theRuntimeReportsWhatEachFamilyMakesOfTheOptions() async throws {
        let fake = try FakeRuntime("splash-families", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let runtime = Runtime(root: fake.root)
        let script = runtime.root.appending(path: "install/families.py")
        // What the script says before its report is no part of it.
        try """
            print("a warning")
            print('{"Flint": {"unusable": ["--state-format"], "context_tokens": 262144, "max_context_tokens": 262144}}')
            """.write(to: script, atomically: true, encoding: .utf8)
        #expect(await runtime.families()
            == ["Flint": ModelFamily(unusable: ["--state-format"], contextTokens: 262_144, maxContextTokens: 262_144)])
        try "import sys; sys.exit(1)".write(to: script, atomically: true, encoding: .utf8)
        #expect(await runtime.families().isEmpty)
    }

    /// Model files on this Mac have the family of their preparation, which their first start
    /// writes: the list has it once that start has ended, failed too.
    @Test func modelFilesOnThisMacHaveTheFamilyTheirStartPrepared() async throws {
        let fake = try FakeRuntime("splash-families", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let runtime = Runtime(root: fake.root)
        try #"print('{"Flint": {"unusable": ["--state-format"], "context_tokens": 262144, "max_context_tokens": 262144}}')"#
            .write(to: runtime.root.appending(path: "install/families.py"), atomically: true, encoding: .utf8)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        #expect(!state.families.isEmpty)
        let file = fake.root.appending(path: "Flint-Q4_0.gguf")
        try Data(count: 1000).write(to: file)
        state.addLocal(file)
        await state.listed()
        #expect(state.installed.last == InstalledModel(id: file.path, bytes: 1000, gguf: true) && state.selection == file.path)
        #expect(state.family(of: file.path) == .unknown)
        state.options["--port"] = String(fake.port)
        state.start(file.path)
        // The start prepares the files under a selection link of its options, as `splash serve` does,
        // then the engine refuses them.
        let record = runtime.models.appending(path: ".selections/0123")
        try FileManager.default.createDirectory(at: record, withIntermediateDirectories: true)
        try #"{"model": "\#(file.path)", "family": "Flint"}"#
            .write(to: record.appending(path: "model.json"), atomically: true, encoding: .utf8)
        for step in ["refused", "loaded"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        try await waitFor(state.serverStatus.look == .failed, "failed")
        try await waitFor(state.installed.last?.family == "Flint", "the family of its preparation")
        #expect(state.installed.last?.bytes == 1000)
        #expect(!state.family(of: file.path).uses("--state-format"))
    }

    /// The card offers a model the options its family can use, and a start passes only those; model
    /// files on this Mac, whose family is not known until a start prepares them, get them all. The
    /// models are listed once launched, a model added is picked once listed, and the model that runs
    /// stays listed, with its pick and its family, while it runs, its files gone.
    @Test func aModelIsOfferedAndStartedWithTheOptionsItsFamilyCanUse() async throws {
        let fake = try FakeRuntime("splash-families", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let runtime = Runtime(root: fake.root)
        try #"print('{"Flint": {"unusable": ["--state-format"], "context_tokens": 262144, "max_context_tokens": 262144}, "Other": {"unusable": []}}')"#
            .write(to: runtime.root.appending(path: "install/families.py"), atomically: true, encoding: .utf8)
        let flint = "a/Flint-GGUF:Q4", other = "a/Other-GGUF:Q4"
        try fake.install(flint, family: "Flint")
        try fake.install(other, family: "Other")
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        #expect(state.family(of: flint) == .unknown)
        await state.launch()
        #expect(state.isInstalled(flint) && state.isInstalled(other))
        // The options the card shows the model picked.
        func shown(_ model: String) -> [String] {
            state.selection = model
            return LaunchOption.Section.allCases.flatMap { LaunchOptions.options($0, for: state.family(of: state.selection)) }.map(\.flag)
        }
        #expect(!shown(flint).contains("--state-format") && shown(flint).contains("--expert-memory"))
        #expect(shown(other).contains("--state-format"))
        // Until the list has the model added, the pick stays.
        let file = fake.root.appending(path: "Added-Q4_0.gguf")
        try Data(count: 10).write(to: file)
        state.addLocal(file)
        #expect(state.selection == other)
        await state.listed()
        #expect(state.selection == file.path)
        #expect(state.family(of: file.path) == .unknown && shown(file.path).contains("--state-format"))
        state.removeLocal(file.path)
        state.options["--port"] = String(fake.port)
        state.options["--state-format"] = "bf16"
        state.options["--expert-memory"] = "16G"
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.start(other)
        try await waitFor(state.server.phase == .ready, "\(other) ready")
        #expect(state.server.settings["--state-format"] == "bf16")
        state.start(flint)
        try await waitFor(state.server.model == flint && state.server.phase == .ready, "\(flint) ready")
        #expect(state.server.settings["--state-format"] == nil && state.server.settings["--expert-memory"] == "16G")
        #expect(!state.optionsChanged)
        // Its files gone, it stays listed while it runs, then leaves the list.
        try FileManager.default.removeItem(at: runtime.models.appending(path: "\(flint)/model.json"))
        await state.refresh().value
        #expect(state.isInstalled(flint) && state.selection == flint && !state.family(of: flint).uses("--state-format"))
        #expect(!state.optionsChanged)
        state.stop()
        try await waitFor(state.server.phase == .stopped, "stopped")
        await state.listed()
        #expect(!state.isInstalled(flint) && state.selection == other)
    }

    /// Only the models of a family the runtime serves are offered, on the Server, in the menu bar's
    /// panel and the composer, which list `installed`: a model of another, which a newer or
    /// development build installs into the models folder every Splash on the Mac shares, is not, nor
    /// picked where it was the model picked last. Model files on this Mac not prepared yet are
    /// offered. Where the runtime doesn't say which families it serves, every model is.
    @Test func onlyTheModelsOfAFamilyTheRuntimeServesAreOffered() async throws {
        let fake = try FakeRuntime("splash-families", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let script = fake.root.appending(path: "install/families.py")
        try #"print('{"Flint": {"unusable": []}}')"#.write(to: script, atomically: true, encoding: .utf8)
        let flint = "a/Flint-GGUF:Q4", speech = "a/Speech-1.7B-4bit"
        try fake.install(flint, family: "Flint")
        try fake.install(speech, family: "Speech")
        let file = fake.root.appending(path: "Added-Q4_0.gguf")
        try Data(count: 10).write(to: file)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        state.addLocal(file)
        await state.listed()
        state.selection = speech
        await state.launch()
        #expect(state.installed.map(\.id) == [flint, file.path])
        #expect(state.selection == flint)
        // The next launch, whose report fails.
        try "import sys; sys.exit(1)".write(to: script, atomically: true, encoding: .utf8)
        let next = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { next.quit() }
        await next.launch()
        #expect(next.installed.map(\.id) == [flint, speech, file.path])
        #expect(next.selection == speech)
    }
}

struct AgentTests {
    /// An agent's command line runs this runtime's `splash` on the server's port, each word quoted
    /// for the shell that types it.
    @Test func runsThisRuntimesSplashOnTheServersPort() {
        let runtime = Runtime(root: URL(filePath: "/opt/it's splash"))
        let command = Agent.claude.command(runtime: runtime, port: 8100)
        #expect(command.hasPrefix(#"'/opt/it'\''s splash/.venv/bin/python' '-I' '-B' '-u'"#))
        #expect(command.hasSuffix(#"'claude' '--port' '8100'"#))
    }

    /// An agent opens in a new terminal of the drawer, which shows, in the agents' folder.
    @MainActor @Test func opensInADrawerTerminalInTheAgentsFolder() throws {
        let fake = try FakeRuntime("splash-agent", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        let folder = fake.root.appending(path: "project")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        state.agentFolder = folder
        state.open(.claude)
        #expect(state.drawer.terminals.count == 1 && state.drawer.terminals[0].title == "project")
        #expect(state.showsDrawer && state.drawer.part == .terminal)
    }
}

struct OutputTests {
    /// The installer classes a Hub or network failure as it ends (install/hub.py failure_kind),
    /// which the app words; any other failure keeps the installer's message.
    @Test func installerErrorsAreWordedByKind() throws {
        func words(_ kind: String) throws -> String {
            let line = #"{"event": "error", "kind": "\#(kind)", "message": "cannot install a/b: refused"}"#
            return Installer.problem(try #require(Installer.Failure(line: line)), model: "a/b:Q4")
        }
        #expect(try words("not_found") == "Hugging Face has no public model named a/b. Check the name, then try again.")
        #expect(try words("offline") == "Splash couldn’t reach Hugging Face. Check the internet connection, then try again.")
        #expect(try words("auth")
            == "Hugging Face refused access to a/b. Sign in with an account that can read it (hf auth login) and, "
                + "if the model asks, accept its terms at huggingface.co/a/b, then try again.")
        let unclassed = try #require(Installer.Failure(
            line: #"{"event": "error", "kind": null, "message": "cannot install a/b: no supported model"}"#))
        #expect(Installer.problem(unclassed, model: "a/b") == "cannot install a/b: no supported model")
        #expect(Installer.Failure(line: #"{"event": "download", "repo": "a/b", "done": 1, "total": 2}"#) == nil)
        #expect(Installer.Failure(line: "error: cannot install a/b") == nil)
    }

    /// What a check for updates found and what a deletion freed are worded as the installer
    /// reports them last (install/models.py prepare --json-progress, remove).
    @Test func checkAndDeletionResultsAreWordedAsReported() throws {
        func outcome(_ name: String) -> Installer.Outcome? {
            Installer.Outcome(line: #"{"event": "prepared", "outcome": "\#(name)"}"#)
        }
        #expect(outcome("updated") == .updated && outcome("up_to_date") == .upToDate)
        #expect(outcome("unreachable") == .unreachable && outcome("kept") == .kept)
        #expect(outcome("later") == nil)
        #expect(Installer.Outcome(line: #"{"event": "removed", "bytes": 1}"#) == nil)
        #expect(Installer.freed(line: #"{"event": "removed", "bytes": 15000000000}"#) == 15_000_000_000)
        #expect(Installer.freed(line: #"{"event": "prepared", "outcome": "kept"}"#) == nil)
        func words(_ result: Installer.Result) -> String {
            let words = Installer.words(result, name: "Q")
            return "\(words.title) / \(words.detail)\(words.done ? "" : " !")"
        }
        #expect(words(.checked(.updated)) == "Q is updated / Its next start uses the new files.")
        #expect(words(.checked(.upToDate)) == "Q is up to date / Hugging Face has nothing newer.")
        #expect(words(.checked(.unreachable)) == "Splash couldn’t check Q for updates / Hugging Face was not reachable. !")
        #expect(words(.checked(.kept)) == "Splash couldn’t update Q / It kept the installed version. !")
        #expect(words(.deleted(freed: 15_000_000_000)) == "Deleted Q / Freed 15 GB.")
        #expect(words(.deleted(freed: 0)) == "Deleted Q / Its files stayed: other models or tools use them.")
    }

    @Test func readsDownloadEvents() {
        let progress = Installer.Progress(
            line: #"{"event": "download", "repo": "a/b", "done": 250, "total": 1000}"#)
        #expect(progress?.repo == "a/b")
        #expect(progress?.fraction == 0.25)
        #expect(Installer.Progress(line: #"{"event": "other", "repo": "a/b", "done": 1, "total": 2}"#) == nil)
        #expect(Installer.Progress(line: "Fetching 1 file(s), 0.93 GB") == nil)
    }

    @Test func formatsFiguresAndTimes() {
        #expect(Format.seconds(9.84) == "9.8s" && Format.seconds(24.4) == "24s" && Format.seconds(84) == "1m 24s")
        #expect(Format.latency(.milliseconds(136)) == ("136", "ms") && Format.latency(.milliseconds(1440)) == ("1.4", "s"))
        #expect(Format.parts("4.3 GB") == ("4.3", "GB") && Format.parts("12") == ("12", nil))
        #expect(Format.whole(36.8) == "37")
    }

    @Test func formatsSizes() {
        #expect(Format.size(120_000_000) == "120 MB")
        #expect(Format.size(6_730_000_000) == "6.7 GB")
        #expect(Format.size(10_980_000_000) == "11 GB")
        #expect(Format.size(21_250_000_000) == "21 GB")
        #expect(Format.size(Format.gigabytes(24)) == "24 GB")
        // Memory counts as the memory limit does: a 16 GB limit is 16 GiB.
        #expect(Format.memory(16 << 30) == "16 GB")
        #expect(Format.memory(18_467_000_000) == "17.2 GB")
        #expect(Format.memory(512 << 20) == "512 MB")
    }
}

struct ChatTests {
    @Test func readsTheEventStream() throws {
        #expect(try ChatClient.parse(": splash-keepalive") == nil)
        #expect(try ChatClient.parse("data: [DONE]") == .done)
        let delta = try ChatClient.parse(#"data: {"choices":[{"delta":{"reasoning_content":"hm","content":"Hi"}}],"metrics":{"request_latency":{"stream_tokens_per_second":71.5}}}"#)
        #expect(delta == .delta(ChatClient.Delta(text: "Hi", reasoning: "hm", tokensPerSecond: 71.5)))
        #expect(try ChatClient.parse(#"data: {"choices":[]}"#) == .delta(ChatClient.Delta()))
        #expect(try ChatClient.parse(#"data: {"choices":[{"delta":{},"finish_reason":"length"}]}"#)
            == .delta(ChatClient.Delta(cut: true)))
        // The prompt as the model reads it, and the tokens its thinking took.
        #expect(try ChatClient.parse(#"data: {"choices":[{"delta":{}}],"prompt_progress":{"total":4096,"cache":0,"processed":1024,"time_ms":80.5}}"#)
            == .delta(ChatClient.Delta(read: 0.25)))
        #expect(try ChatClient.parse(#"data: {"choices":[],"usage":{"completion_tokens":600,"completion_tokens_details":{"reasoning_tokens":412}}}"#)
            == .delta(ChatClient.Delta(reasoningTokens: 412)))
        #expect(throws: ChatClient.Failure.self) {
            try ChatClient.parse(#"data: {"error":{"message":"engine is recovering; retry shortly"}}"#)
        }
    }

    @Test func saysWhenAChatOutgrowsTheContext() {
        // As the server refuses a prompt longer than the context, before it streams anything.
        let overflow = Data(#"{"error":{"message":"prompt exceeds the context window","type":"invalid_request_error","code":"context_length_exceeded"}}"#.utf8)
        let chat = [ChatMessage(role: .user, text: "Hi"), ChatMessage(role: .assistant, text: "Hello"),
                    ChatMessage(role: .user, text: "And then?")]
        var target = ChatTarget(url: URL(string: "http://127.0.0.1:8000")!, model: "a/b", acceptsImages: false, effort: "", context: 262_144)
        #expect(ChatClient.message(of: overflow, for: chat, server: target)
            == "This chat is longer than the model’s 256K-token context. Start a new chat to continue.")
        // A first message too long has no earlier ones a new chat would leave behind.
        #expect(ChatClient.message(of: overflow, for: [chat[0]], server: target) == "This message is longer than the model’s 256K-token context.")
        target.context = 57_337
        #expect(ChatClient.message(of: overflow, for: [chat[0]], server: target) == "This message is longer than the model’s 57,337-token context.")
        target.context = nil
        #expect(ChatClient.message(of: overflow, for: chat, server: target) == "This chat is longer than the model’s context. Start a new chat to continue.")
        let other = Data(#"{"error":{"message":"engine is recovering; retry shortly","code":"engine_recovering"}}"#.utf8)
        #expect(ChatClient.message(of: other, for: chat, server: target) == "engine is recovering; retry shortly")
        #expect(ChatClient.message(of: Data("Bad Gateway".utf8), for: chat, server: target) == nil)
    }

    @Test func asksWithImagesEarlierThoughtsAndTheEffort() throws {
        let png = Showcase.picture
        let messages = [ChatMessage(role: .user, text: "What is this?", attachments: [Showcase.pictureAttachment]),
                        ChatMessage(role: .assistant, text: "A cat.", reasoning: "Whiskers."),
                        ChatMessage(role: .user, text: "Sure?"),
                        ChatMessage(role: .assistant, text: "", error: "The reply was interrupted."),
                        ChatMessage(role: .user, text: "Well?")]
        let files = Showcase.attachments
        let body = try JSONSerialization.jsonObject(with: ChatClient.body(messages, effort: "low", vision: true, files: files)) as! [String: Any]
        #expect(body["reasoning_effort"] as? String == "low")
        #expect(body["stream"] as? Bool == true)
        #expect(body["return_progress"] as? Bool == true)
        let wire = body["messages"] as! [[String: Any]]
        let parts = wire[0]["content"] as! [[String: Any]]
        #expect((parts[0]["image_url"] as! [String: String])["url"] == "data:image/png;base64,\(png.base64EncodedString())")
        #expect(parts[1]["text"] as? String == "What is this?")
        #expect(wire[1]["reasoning_content"] as? String == "Whiskers.")
        // The interrupted answer, which holds nothing, is left out.
        #expect(wire.map { $0["role"] as? String } == ["user", "assistant", "user", "user"])
        let auto = try JSONSerialization.jsonObject(with: ChatClient.body(messages, effort: "", vision: true, files: files)) as! [String: Any]
        #expect(auto["reasoning_effort"] == nil)
        // A model that cannot see images gets a note in their place.
        let text = try JSONSerialization.jsonObject(with: ChatClient.body(messages, effort: "", vision: false, files: files)) as! [String: Any]
        #expect((text["messages"] as! [[String: Any]])[0]["content"] as? String == "[An image this model can’t see]\n\nWhat is this?")
    }

    @Test func asksWithPDFsAsFiles() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-pdf-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let data = Data("%PDF-1.4 plan".utf8)
        let pdf = Splash.Attachment(kind: .pdf, type: "application/pdf", bytes: data.count, name: "Plan.pdf", pages: 1)
        try data.write(to: folder.appending(path: pdf.file))
        try Showcase.picture.write(to: folder.appending(path: Showcase.pictureAttachment.file))
        let messages = [ChatMessage(role: .user, text: "Compare", attachments: [Showcase.pictureAttachment, pdf])]
        let body = try JSONSerialization.jsonObject(with: ChatClient.body(messages, effort: "", vision: true, files: folder)) as! [String: Any]
        let parts = (body["messages"] as! [[String: Any]])[0]["content"] as! [[String: Any]]
        #expect(parts.map { $0["type"] as? String } == ["image_url", "file", "text"])
        #expect(parts[1]["file"] as? [String: String] == ["filename": "Plan.pdf", "file_data": "data:application/pdf;base64,\(data.base64EncodedString())"])
        // A model that takes neither gets a note for each kind.
        let text = try JSONSerialization.jsonObject(with: ChatClient.body(messages, effort: "", vision: false, files: folder)) as! [String: Any]
        #expect((text["messages"] as! [[String: Any]])[0]["content"] as? String
            == "[An image this model can’t see] [A PDF this model can’t read]\n\nCompare")
    }

    @MainActor @Test func aChatOfAttachmentsAloneIsNamedForItsPDF() {
        let pdf = Splash.Attachment(kind: .pdf, type: "application/pdf", bytes: 1, name: "Plan.pdf", pages: 1)
        #expect(ChatStore.firstLine("Read this\nplease", attachments: [pdf]) == "Read this")
        #expect(ChatStore.firstLine("", attachments: [Showcase.pictureAttachment, pdf]) == "Plan.pdf")
        #expect(ChatStore.firstLine("", attachments: [Showcase.pictureAttachment]) == "Image")
    }

    /// What the model can't take stays in the draft, which says why: a kind it doesn't take, a PDF
    /// over its bytes or pages, PDFs over their bytes together, an image over its bytes, more
    /// images and pages than a request takes, or more bytes than it takes once encoded.
    @Test func attachmentsTheModelCannotTakeSayWhy() {
        let limits = Server.Limits(request_bytes: 800, images: 4, image_bytes: 300, pdf_bytes: 400, pdf_pages: 3)
        func pdf(_ name: String, bytes: Int = 100, pages: Int? = 1) -> Splash.Attachment {
            Splash.Attachment(kind: .pdf, type: "application/pdf", bytes: bytes, name: name, pages: pages)
        }
        let image = Splash.Attachment(kind: .image, type: "image/png", bytes: 100)
        func problem(_ attachments: [Splash.Attachment], vision: Bool = true, limits: Server.Limits? = limits) -> String? {
            AppState.attachmentProblem(attachments, vision: vision, limits: limits)
        }
        #expect(problem([]) == nil && problem([], vision: false) == nil)
        #expect(problem([image], vision: false) == "This model can’t see images")
        #expect(problem([image, pdf("a.pdf")], vision: false) == "This model can’t read PDFs")
        // A server that doesn't say its limits says why itself, as it refuses them.
        #expect(problem([pdf("a.pdf", bytes: 10_000)], limits: nil) == nil)
        #expect(problem([image, pdf("a.pdf", pages: 3)]) == nil)
        // One PDFKit can't count goes, for the server to say why.
        #expect(problem([pdf("a.pdf", pages: nil)]) == nil)
        #expect(problem([pdf("a.pdf", bytes: 401)]) == "a.pdf is larger than the model’s \(Format.size(400)) limit")
        #expect(problem([pdf("a.pdf", pages: 4)]) == "a.pdf has more than the model’s 3 pages")
        #expect(problem([pdf("a.pdf", bytes: 250), pdf("b.pdf", bytes: 250)])
            == "These PDFs are larger together than the model’s \(Format.size(400)) limit")
        #expect(problem([Splash.Attachment(kind: .image, type: "image/png", bytes: 301)]) == "An image is larger than the model’s \(Format.size(300)) limit")
        #expect(problem([image, image, pdf("a.pdf", pages: 3)]) == "The images and PDF pages are more than the model’s 4")
        // Sent as base64, a third larger: 700 bytes are 936.
        #expect(problem([image, image, image, pdf("a.pdf", bytes: 400)]) == "The attachments are larger than one message to the model takes")
    }

    /// A conversation keeps its attachments as files of their own: once the app adopts the files
    /// (`adoptFiles`), the images an earlier app kept inside its messages move into files, in their
    /// order, the conversation written again without them, and a file no conversation names goes,
    /// unless a conversation that can't be read may name it; and a conversation deleted takes its
    /// files.
    @MainActor @Test func keepsAttachmentsAsFilesOfTheirOwn() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder.appending(path: "Attachments"), withIntermediateDirectories: true)
        let png = Showcase.picture, jpeg = Data([0xFF, 0xD8, 0xFF, 0xE0, 1, 2])
        let id = UUID()
        let old = #"{"id": "\#(id)", "title": "Old", "updated": 0, "messages": [{"id": "\#(UUID())", "role": "user", "#
            + #""text": "Look", "reasoning": "", "images": ["\#(png.base64EncodedString())", "\#(jpeg.base64EncodedString())"]}]}"#
        try Data(old.utf8).write(to: folder.appending(path: "\(id).json"))
        let stray = folder.appending(path: "Attachments/stray.png")
        try Data([1]).write(to: stray)
        let store = ChatStore(folder: folder)
        // Nothing changes until the app adopts the files, once it is the one Splash that runs: a
        // copy that gives way leaves them as the app that goes on running reads them.
        store.flush()
        let file = folder.appending(path: "\(id).json")
        #expect(try String(contentsOf: file, encoding: .utf8).contains(png.base64EncodedString()))
        #expect(FileManager.default.fileExists(atPath: stray.path))
        store.adoptFiles()
        store.flush()
        let message = try #require(store.conversations.first?.messages.first)
        let attachments = try #require(message.attachments)
        #expect(attachments.map(\.type) == ["image/png", "image/jpeg"] && message.images.isEmpty)
        #expect(try attachments.map { try Data(contentsOf: store.attachments.appending(path: $0.file)) } == [png, jpeg])
        let written = try String(contentsOf: file, encoding: .utf8)
        #expect(written.contains("attachments") && !written.contains(png.base64EncodedString()))
        #expect(!FileManager.default.fileExists(atPath: stray.path))
        // A conversation that can't be read keeps every file: it may name them.
        try Data([1]).write(to: stray)
        let unreadable = folder.appending(path: "\(UUID()).json")
        try Data("{".utf8).write(to: unreadable)
        let other = ChatStore(folder: folder)
        other.adoptFiles()
        other.flush()
        #expect(FileManager.default.fileExists(atPath: stray.path))
        try FileManager.default.removeItem(at: unreadable)
        // Deleted, a conversation takes its files.
        store.delete([id])
        store.flush()
        #expect(attachments.allSatisfy { !FileManager.default.fileExists(atPath: store.attachments.appending(path: $0.file).path) })
    }

    /// A launch that can't go on, as on a Mac Splash can't run on, leaves the chats as the app that
    /// goes on running reads them: only the Splash that stays moves their images into files.
    @MainActor @Test func aBlockedLaunchLeavesTheChatsAsTheyAre() async throws {
        let fake = try FakeRuntime("splash-blocked", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let engine = Runtime(root: fake.root).engine
        try FileManager.default.createDirectory(at: engine.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "#!/bin/sh\necho 'error: no Metal GPU' >&2\nexit 1\n".write(to: engine, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: engine.path)
        try FileManager.default.createDirectory(at: fake.chats, withIntermediateDirectories: true)
        let id = UUID()
        let old = #"{"id": "\#(id)", "title": "Old", "updated": 0, "messages": [{"id": "\#(UUID())", "role": "user", "#
            + #""text": "Look", "reasoning": "", "images": ["\#(Showcase.picture.base64EncodedString())"]}]}"#
        let file = fake.chats.appending(path: "\(id).json")
        try Data(old.utf8).write(to: file)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        #expect(state.blocker != nil)
        state.chats.flush()
        #expect(try String(contentsOf: file, encoding: .utf8) == old)
    }

    @MainActor @Test func keepsConversationsInTheirFolder() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let store = ChatStore(folder: folder)
        #expect(store.conversations.isEmpty)
        // Sending to no server fails, and the conversation keeps the question and why.
        let nowhere = ChatTarget(url: URL(string: "http://127.0.0.1:9")!, model: "a/b", acceptsImages: false, effort: "")
        store.send("Hello\nthere", attachments: [], to: nowhere)
        #expect(store.current?.title == "Hello")
        #expect(store.streaming != nil)
        while store.streaming != nil { try await Task.sleep(for: .milliseconds(20)) }
        let answer = try #require(store.current?.messages.last)
        #expect(answer.role == .assistant && answer.model == "a/b" && answer.error != nil)
        store.rename(store.conversations[0].id, to: "Greeting")
        // Written off the main thread, on disk once flushed, as the app does as it quits.
        store.flush()
        let reloaded = ChatStore(folder: folder)
        #expect(reloaded.conversations.map(\.title) == ["Greeting"])
        #expect(reloaded.conversations[0].messages == store.conversations[0].messages)
        store.delete([store.conversations[0].id])
        #expect(store.conversations.isEmpty && store.selection == nil)
        store.flush()
        #expect(ChatStore(folder: folder).conversations.isEmpty)
    }

    /// The list's groups: the pinned chats, whenever they changed, the last pinned first; then today,
    /// yesterday, the week before, the month before and older, by the day each changed last, each
    /// in its chats' order; a group with no chat is left out.
    @Test func theChatListGroupsThePinnedChatsThenTheOthersByDay() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        func chat(_ title: String, _ daysAgo: Double, pinnedDaysAgo: Double? = nil) -> Conversation {
            Conversation(title: title, updated: now.addingTimeInterval(-daysAgo * 86_400),
                         pinned: pinnedDaysAgo.map { now.addingTimeInterval(-$0 * 86_400) })
        }
        let chats = [chat("a", 0), chat("p", 0, pinnedDaysAgo: 5), chat("b", 1), chat("c", 3), chat("d", 20), chat("e", 60),
                     chat("f", 60, pinnedDaysAgo: 1), chat("g", 0)]
        let sections = ChatSection.of(chats, now: now, calendar: calendar)
        #expect(sections.map(\.group) == [.pinned, .today, .yesterday, .week, .month, .older])
        #expect(sections.map { $0.chats.map(\.title) } == [["f", "p"], ["a", "g"], ["b"], ["c"], ["d"], ["e"]])
        #expect(ChatSection.of([chat("b", 1)], now: now, calendar: calendar).map(\.group) == [.yesterday])
    }

    /// A pinned chat stays pinned on disk, and one unpinned stays so; a file from before pinning
    /// reads as not pinned.
    @MainActor @Test func aChatPinnedStaysPinned() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let old = Conversation(title: "Old")
        try JSONEncoder().encode(old).write(to: folder.appending(path: "\(old.id.uuidString).json"))
        #expect(try !String(contentsOf: folder.appending(path: "\(old.id.uuidString).json"), encoding: .utf8).contains("pinned"))
        let store = ChatStore(folder: folder)
        #expect(store.conversations.map(\.pinned) == [nil])
        store.pin(old.id, true)
        store.flush()
        #expect(ChatStore(folder: folder).conversations.first?.pinned != nil)
        store.pin(old.id, false)
        store.flush()
        #expect(ChatStore(folder: folder).conversations.first?.pinned == nil)
    }

    /// The list's search: the chats whose title or a message holds the query, whatever its case;
    /// none for a query nothing holds; every chat, in the list's order, for an empty query.
    @MainActor @Test func theChatListSearchesTitlesAndMessages() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for conversation in [Conversation(title: "Plan the Lisbon trip", messages: [ChatMessage(role: .user, text: "Four days in May.")]),
                             Conversation(title: "Grocery list", messages: [ChatMessage(role: .user, text: "Eggs, milk and bread.")])] {
            try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        }
        let store = ChatStore(folder: folder)
        #expect(store.matching("lisbon").map(\.title) == ["Plan the Lisbon trip"])
        #expect(store.matching("MILK").map(\.title) == ["Grocery list"])
        #expect(store.matching("Porto").isEmpty)
        #expect(store.matching("").map(\.id) == store.conversations.map(\.id))
    }

    /// A title is the first line of the model's reply that holds any, without a Title: label,
    /// quotes, Markdown's marks or a closing stop, in at most 80 characters; a reply with none
    /// gives no title.
    @Test func aTitleIsTheReplysFirstLineTidied() {
        #expect(ChatClient.title(in: "Planning a Lisbon Trip") == "Planning a Lisbon Trip")
        #expect(ChatClient.title(in: "Title: \"Swift Concurrency Basics.\"") == "Swift Concurrency Basics")
        #expect(ChatClient.title(in: "**KV Cache Paging**\nIt covers…") == "KV Cache Paging")
        #expect(ChatClient.title(in: "\n  “法语学习计划。”  ") == "法语学习计划")
        #expect(ChatClient.title(in: "What Is Speculative Decoding?") == "What Is Speculative Decoding?")
        #expect(ChatClient.title(in: "## Learning C#") == "Learning C#")
        #expect(ChatClient.title(in: "**Title:** Trip Plans") == "Trip Plans")
        #expect(ChatClient.title(in: "# Title: Trip Plans") == "Trip Plans")
        #expect(ChatClient.title(in: "**Trip Plans**.") == "Trip Plans")
        #expect(ChatClient.title(in: "**\nTrip Plans") == "Trip Plans")
        #expect(ChatClient.title(in: String(repeating: "a", count: 100))?.count == 80)
        #expect(ChatClient.title(in: " \n ") == nil)
        #expect(ChatClient.title(in: "\"\"") == nil)
    }

    /// A title is asked of the start of the first exchange, in the background, without thinking, in a
    /// few tokens and not streamed.
    @Test func aTitleIsAskedOfTheFirstExchangeWithoutThinking() throws {
        let body = try JSONSerialization.jsonObject(with: ChatClient.titleBody(question: "Hi", answer: String(repeating: "x", count: 3000))) as! [String: Any]
        #expect(body["reasoning_effort"] as? String == "none" && body["thinking_token_budget"] as? Int == 0)
        #expect(body["priority"] as? String == "background")
        #expect(body["max_tokens"] as? Int == 32)
        #expect(body["stream"] == nil)
        let content = try #require(((body["messages"] as? [[String: Any]])?.first?["content"]) as? String)
        #expect(content.contains("User: Hi\n\nAssistant: " + String(repeating: "x", count: 1000)) && !content.contains(String(repeating: "x", count: 1001)))
    }

    /// Its first exchange answered, a chat takes the title the model sums it up in, tidied; until
    /// then it has its first line.
    @MainActor @Test func aChatTakesTheTitleTheModelSumsItUpIn() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let server = Process()
        server.executableURL = URL(filePath: "/usr/bin/python3")
        server.arguments = [try #require(Bundle.module.path(forResource: "streaming_server", ofType: "py", inDirectory: "Fakes")),
                            "--title", "“Greeting the user.”"]
        let output = Pipe()
        server.standardOutput = output
        try server.run()
        defer { server.terminate() }
        let port = try #require(String(data: output.fileHandleForReading.availableData, encoding: .utf8)
            .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) })
        let store = ChatStore(folder: folder)
        store.send("Hello\nthere", attachments: [], to: ChatTarget(url: URL(string: "http://127.0.0.1:\(port)")!, model: "a/b", acceptsImages: false, effort: ""))
        #expect(store.current?.title == "Hello")
        try await waitFor(store.current?.title == "Greeting the user", "the model's title")
        store.flush()
        #expect(ChatStore(folder: folder).conversations.first?.title == "Greeting the user")
    }

    /// A chat renamed before its first reply ends keeps its name: no title is asked for.
    @MainActor @Test func aChatRenamedKeepsItsName() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let server = Process()
        server.executableURL = URL(filePath: "/usr/bin/python3")
        server.arguments = [try #require(Bundle.module.path(forResource: "streaming_server", ofType: "py", inDirectory: "Fakes")),
                            "--title", "Not Mine"]
        let output = Pipe()
        server.standardOutput = output
        try server.run()
        defer { server.terminate() }
        let port = try #require(String(data: output.fileHandleForReading.availableData, encoding: .utf8)
            .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) })
        let store = ChatStore(folder: folder)
        store.send("Hello", attachments: [], to: ChatTarget(url: URL(string: "http://127.0.0.1:\(port)")!, model: "a/b", acceptsImages: false, effort: ""))
        let id = try #require(store.current?.id)
        store.rename(id, to: "Mine")
        try await waitFor(store.streaming == nil, "the reply to end")
        try await Task.sleep(for: .milliseconds(500))
        #expect(store.current?.title == "Mine")
    }

    /// The question to delete chats names one by its title, and else how many.
    @MainActor @Test func theQuestionToDeleteChatsNamesOneByItsTitleElseHowMany() {
        let one = ChatDeletion([Conversation(title: "Greeting")])
        #expect(one.title == "Delete this chat?" && one.message == "“Greeting” will be deleted from this Mac.")
        let three = ChatDeletion([Conversation(), Conversation(), Conversation()])
        #expect(three.ids.count == 3 && three.title == "Delete 3 chats?" && three.message == "They will be deleted from this Mac.")
    }

    /// Deleting chats deletes each with its file and stops the reply that streams in one, and the
    /// open one deleted, a new chat opens; a file in the folder that is no chat stays, even under
    /// the name of an id asked for.
    @MainActor @Test func deletingChatsDeletesThemWithTheirFiles() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let kept = Conversation(title: "Kept"), gone = Conversation(title: "Gone")
        for conversation in [kept, gone] {
            try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        }
        let unread = UUID()
        let stray = folder.appending(path: "\(unread.uuidString).json")
        try Data("not a chat".utf8).write(to: stray)
        // A reply without end: only the stop ends it.
        let server = Process()
        server.executableURL = URL(filePath: "/usr/bin/python3")
        server.arguments = [try #require(Bundle.module.path(forResource: "streaming_server", ofType: "py", inDirectory: "Fakes")), "--endless"]
        let output = Pipe()
        server.standardOutput = output
        try server.run()
        defer { server.terminate() }
        let port = try #require(String(data: output.fileHandleForReading.availableData, encoding: .utf8)
            .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) })
        let store = ChatStore(folder: folder)
        store.send("Hi", attachments: [], to: ChatTarget(url: URL(string: "http://127.0.0.1:\(port)")!, model: "a/b", acceptsImages: false, effort: ""))
        let streaming = try #require(store.streaming)
        #expect(store.selection == streaming)
        store.delete([streaming, gone.id, unread])
        #expect(store.conversations.map(\.id) == [kept.id] && store.selection == nil)
        try await waitFor(store.streaming == nil, "the reply to stop")
        #expect(store.conversations.map(\.id) == [kept.id])
        store.flush()
        #expect(ChatStore(folder: folder).conversations.map(\.id) == [kept.id])
        #expect(FileManager.default.fileExists(atPath: stray.path))
    }

    @MainActor @Test func aReplyStreamsApartFromTheConversationsUntilItEnds() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let server = Process()
        server.executableURL = URL(filePath: "/usr/bin/python3")
        server.arguments = [try #require(Bundle.module.path(forResource: "streaming_server", ofType: "py", inDirectory: "Fakes"))]
        let output = Pipe()
        server.standardOutput = output
        try server.run()
        defer { server.terminate() }
        let port = try #require(String(data: output.fileHandleForReading.availableData, encoding: .utf8)
            .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) })
        let store = ChatStore(folder: folder)
        store.send("Hi", attachments: [], to: ChatTarget(url: URL(string: "http://127.0.0.1:\(port)")!, model: "a/b", acceptsImages: false, effort: ""))
        // While it streams, the reply holds what came, and the conversation, which the list and
        // the composer read, holds it empty and changes no more.
        try await waitFor(store.reply?.message.text.isEmpty == false)
        let first = store.reply?.message.text ?? ""
        #expect(store.current?.messages.last?.text == "")
        nonisolated(unsafe) var changed = false
        withObservationTracking { _ = store.conversations } onChange: { changed = true }
        try await waitFor(store.reply.map { $0.message.text != first } ?? true)
        #expect(store.reply?.message.text.hasPrefix(first) == true)
        #expect(!changed && store.current?.messages.last?.text == "")
        // Once it ends, it goes into its conversation, which is written.
        try await waitFor(store.streaming == nil)
        #expect(changed && store.reply == nil && store.current?.messages.last?.text == "One two three")
        store.flush()
        #expect(ChatStore(folder: folder).conversations.first?.messages.last?.text == "One two three")
    }

    /// What changed last is on disk by when the app has ended, though writes go to a queue. Large,
    /// so that the queue is still writing it when the app goes on to end.
    @MainActor @Test func theAppEndsOnceTheChatsAreWritten() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let conversation = Conversation(title: "Before", messages: [ChatMessage(role: .user, text: "Hi", images: [Data(count: 8_000_000)])])
        try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        let state = AppState.forTests(in: folder)
        let delegate = AppDelegate(state: state)
        state.chats.rename(conversation.id, to: "After")
        delegate.applicationWillTerminate(Notification(name: NSApplication.willTerminateNotification))
        #expect(ChatStore(folder: folder).conversations.first?.title == "After")
    }

    @MainActor @Test func anAnswerTheAppQuitDuringSaysSo() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let conversation = Conversation(title: "Hi", messages: [ChatMessage(role: .user, text: "Hi"),
                                                               ChatMessage(role: .assistant, text: "")])
        try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        #expect(ChatStore(folder: folder).conversations[0].messages[1].error == "The reply was interrupted.")
    }
}
