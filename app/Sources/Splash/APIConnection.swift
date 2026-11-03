import Foundation
import SwiftUI

/// Connection details for one running model, using its launch settings rather than pending edits.
struct APIConnection: Equatable {
    let settings: ServeSettings
    let model: String

    var baseURL: String { settings.url.appending(path: "v1").absoluteString }

    /// The name the API reports, including local files and an announced alias.
    var modelID: String {
        if settings["--announce-served-name"] == "true",
           let names = settings["--served-model-name"], let first = LaunchOption.items(names).first {
            return first
        }
        return model.hasPrefix("/") ? ServeSettings.localID(model) : model
    }

    /// A short request, with JSON and shell quoting kept separate. The key stays in the user's
    /// environment: neither the preview nor its copy contains the key kept in the Keychain.
    func curl() throws -> String {
        struct Message: Encodable { let role = "user"; let content = "Say hello in one sentence." }
        struct Body: Encodable {
            let model: String
            let messages = [Message()]
            let max_tokens = 128
            let reasoning_effort = "none"
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let body = String(decoding: try encoder.encode(Body(model: modelID)), as: UTF8.self)
        var lines = ["curl \(settings.url.appending(path: "v1/chat/completions").absoluteString)",
                     "  -H 'Content-Type: application/json'"]
        if settings.apiKey != nil { lines.append(#"  -H "Authorization: Bearer ${SPLASH_API_KEY}""#) }
        lines.append("  -d '" + body.replacing("'", with: #"'\''"#) + "'")
        return lines.joined(separator: " \\\n")
    }
}

/// The settings an external app needs, with a request folded away until wanted.
struct APIConnectionSection: View {
    let connection: APIConnection
    @State var example = false

    var body: some View {
        Section {
            value("Base URL", connection.baseURL)
            value("Model ID", connection.modelID)
            if let request = try? connection.curl() {
                DisclosureRow(title: "Example Request", open: $example)
                if example {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("cURL").font(.callout).foregroundStyle(.secondary)
                            Spacer()
                            CopyButton(copy: { NSPasteboard.put(request) }) { copied in
                                Label(copied ? "Copied" : "Copy Request", systemImage: copied ? "checkmark" : "doc.on.doc")
                            }
                            .controlSize(.small)
                        }
                        ScrollView(.horizontal) {
                            Text(verbatim: request)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .fixedSize(horizontal: true, vertical: false)
                                .padding(.bottom, 4)
                        }
                    }
                    .padding(10)
                    .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 8))
                }
            }
        } header: {
            SectionTitle(text: "Connect to the API")
        } footer: {
            VStack(alignment: .leading, spacing: 4) {
                Text("Use these settings in an OpenAI-compatible app on this Mac.")
                if connection.settings.apiKey != nil {
                    Text("API key required. Set SPLASH_API_KEY to your key before running the example.")
                } else {
                    Text("No API key required.")
                }
            }
        }
    }

    private func value(_ title: String, _ text: String) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.callout).foregroundStyle(.secondary)
                Text(verbatim: text)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .lineLimit(2)
                    .help(text)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            CopyButton(copy: { NSPasteboard.put(text) }) { copied in
                Text(copied ? "Copied" : "Copy")
            }
            .controlSize(.small)
            .accessibilityLabel("Copy \(title)")
            .help("Copy \(title)")
        }
        .padding(.vertical, 2)
    }
}
