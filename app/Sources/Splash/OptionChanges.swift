import Foundation

extension LaunchOption {
    /// One effective difference between the running server and the next start. These are display
    /// values only; the API key never becomes part of a change's text.
    struct Change: Identifiable, Equatable {
        let id: String
        let title: String
        let before: String
        let after: String

        var detail: String { before == after ? "Changed" : "\(before) → \(after)" }
    }

    fileprivate func displayValue(in settings: ServeSettings) -> String {
        guard applies(settings) else { return "Not used" }
        if flag == "--port" { return String(settings.port) }
        let value = settings[flag]
        switch kind {
        case .toggle(let on):
            return ((value == "true") == on) ? "On" : "Off"
        case .choice(let standard, let choices, let custom):
            guard let value else { return standard }
            return choices.first { $0.value == value }?.title ?? custom?.title(value) ?? value
        case .text(_, list: true):
            return value.map { Self.items($0).joined(separator: ", ") } ?? "Default"
        case .text, .folder:
            return value ?? "Default"
        case .secret:
            return settings.apiKey == nil ? "Not required" : "Required"
        }
    }
}

extension ServeSettings {
    /// In the launch table's order, comparing only what each start actually passes to the server.
    /// Inactive options and alternate comma/space formatting of lists do not create false changes.
    func changes(from running: ServeSettings) -> [LaunchOption.Change] {
        LaunchOption.all.compactMap { option in
            let different: Bool
            if option.flag == "--port" {
                different = port != running.port
            } else if case .secret = option.kind {
                different = apiKey != running.apiKey
            } else {
                let before = option.applies(running) ? option.arguments(running[option.flag]) : []
                let after = option.applies(self) ? option.arguments(self[option.flag]) : []
                different = before != after
            }
            guard different else { return nil }
            let title = option.section == .main ? option.title : "\(option.section.rawValue): \(option.title)"
            // Form flattens review and editable rows into the same section. Their IDs must differ.
            return LaunchOption.Change(id: "pending:\(option.flag)", title: title,
                                       before: option.displayValue(in: running), after: option.displayValue(in: self))
        }
    }
}
