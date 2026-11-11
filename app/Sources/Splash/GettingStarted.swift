import SwiftUI

/// The first conversation needs three steps; preparing a download or a model stays on its step.
enum FirstUseStage: Equatable, CaseIterable {
    case download, downloading, start, starting, chat

    var step: Int {
        switch self {
        case .download, .downloading: 0
        case .start, .starting: 1
        case .chat: 2
        }
    }

    var title: String {
        switch self {
        case .download: "Welcome to Splash"
        case .downloading: "Downloading your model"
        case .start: "Your model is ready to start"
        case .starting: "Getting your model ready"
        case .chat: "You’re ready to chat"
        }
    }

    var message: String {
        switch self {
        case .download: "An AI model is the part that writes replies. Choose one below to chat privately on this Mac."
        case .downloading: "Keep Splash open while it downloads. When it finishes, choose Start Model."
        case .start: "Choose Start Model below, then open Chat. You can use the default settings to get started."
        case .starting: "Preparing your model so it can answer. This may take a little longer the first time."
        case .chat: "Open Chat to ask a question, write something, or explore an idea."
        }
    }
}

struct GettingStartedGuide: View {
    let stage: FirstUseStage?
    private let steps = ["Download a model", "Start Model", "Open Chat"]

    var body: some View {
        // Each stage keeps the same space, so a start changes the guide's
        // words without moving or animating the model rows below it.
        ZStack(alignment: .topLeading) {
            ForEach(FirstUseStage.allCases, id: \.self) { candidate in
                content(for: candidate)
                    .opacity(candidate == stage ? 1 : 0)
                    .accessibilityHidden(candidate != stage)
            }
            if stage == nil {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Set up your first chat").font(.title2.weight(.semibold)).foregroundStyle(.primary)
                    Text("Use the details below to get your model ready, then open Chat.")
                        .font(.callout).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .transaction { $0.animation = nil }
    }

    private func content(for stage: FirstUseStage) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(stage.title).font(.title2.weight(.semibold)).foregroundStyle(.primary)
            Text(stage.message).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 12) {
                ForEach(steps.indices, id: \.self) { index in
                    Label(steps[index], systemImage: index < stage.step ? "checkmark.circle.fill" : "\(index + 1).circle")
                        .font(.callout.weight(index == stage.step ? .semibold : .regular))
                        .foregroundStyle(index <= stage.step ? .primary : .secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel("Step \(index + 1): \(steps[index])")
                        .accessibilityValue(index < stage.step ? "Complete" : index == stage.step ? "Current step" : "Next step")
                }
            }
            .padding(.top, 4)
        }
        .textCase(nil)
    }
}

/// Concrete examples to edit in a new chat; picking one does not start a reply.
enum ChatExample: String, CaseIterable, Identifiable {
    case explain = "Explain something", write = "Write a message", plan = "Plan a weekend"

    var id: Self { self }

    var symbol: String {
        switch self {
        case .explain: "lightbulb"
        case .write: "square.and.pencil"
        case .plan: "calendar"
        }
    }

    var text: String {
        switch self {
        case .explain: "Explain how rainbows form in simple terms."
        case .write: "Help me write a friendly thank-you message."
        case .plan: "Help me plan a relaxing weekend at home."
        }
    }
}
