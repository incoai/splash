import Foundation

/// The models Splash suggests: `install/suggested-models.tsv`, which shell completion reads too.
/// Each line is `MODEL<TAB>NAME<TAB>MEMORY<TAB>SIZE<TAB>ABOUT`: the ID `splash serve --model`
/// takes, the model's name, the smallest Mac memory in GB the variant is meant for, its download
/// in GB, and a few words on the model. Best first.
struct Catalog {
    struct Variant: Hashable {
        let id: String
        let name: String
        let memory: Int
        /// The download, in GB.
        let size: Int
        let about: String
    }

    let variants: [Variant]

    init(contentsOf url: URL) {
        self.init(text: (try? String(contentsOf: url, encoding: .utf8)) ?? "")
    }

    init(text: String) {
        variants = text.split(separator: "\n").compactMap { line in
            let fields = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard !line.hasPrefix("#"), fields.count == 5, let memory = Int(fields[2]), let size = Int(fields[3])
            else { return nil }
            return Variant(id: fields[0], name: fields[1], memory: memory, size: size, about: fields[4])
        }
    }

    /// For each model, best first, the variant for a Mac with `memory` GB: the one meant for the
    /// most memory that Mac has, the first listed among equals. Models it has too little memory for
    /// are left out.
    func suggestions(memory: Int) -> [Variant] {
        var chosen: [String: Variant] = [:]
        for variant in variants where variant.memory <= memory {
            if variant.memory > chosen[variant.name]?.memory ?? -1 { chosen[variant.name] = variant }
        }
        var names: [String] = []
        for variant in variants where !names.contains(variant.name) { names.append(variant.name) }
        return names.compactMap { chosen[$0] }
    }

    /// The least memory any suggested model is meant for.
    var minimumMemory: Int? { variants.map(\.memory).min() }

    /// The name to show for model `id`: its suggested name, or else its repository's, without the
    /// variant and the format the repository's name ends in.
    func name(of id: String) -> String {
        if let name = variants.first(where: { $0.id == id })?.name { return name }
        // Model files on this Mac: the file's or folder's name.
        if id.hasPrefix("/") { return URL(filePath: id).lastPathComponent.replacing(/(?i)\.gguf$/, with: "") }
        let repository = id.split(separator: ":")[0].split(separator: "/").last.map(String.init) ?? id
        // Plain string checks: views ask for names on every render.
        let lower = repository.lowercased()
        return lower.hasSuffix("-gguf") ? String(repository.dropLast(5)) : lower.hasSuffix("-mlx") ? String(repository.dropLast(4)) : repository
    }

    /// What tells variants of a model apart: the folder of model files on this Mac, the GGUF variant
    /// `id` names, or MLX for a model in MLX's format, or else its repository.
    static func variant(of id: String) -> String {
        if id.hasPrefix("/") { return URL(filePath: id).deletingLastPathComponent().lastPathComponent }
        if let variant = id.split(separator: ":", maxSplits: 1).dropFirst().first { return String(variant) }
        return id.localizedCaseInsensitiveContains("mlx") ? "MLX" : String(id.split(separator: "/").last ?? Substring(id))
    }

    /// The few words on model `id`, if it is a suggested one.
    func about(of id: String) -> String? {
        variants.first { $0.id == id }?.about
    }

    /// This Mac's unified memory in GB, as Apple sells it (24, 36, 48, ...).
    static var memory: Int {
        Int((Double(ProcessInfo.processInfo.physicalMemory) / Double(1 << 30)).rounded())
    }
}
