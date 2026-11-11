import SwiftUI

/// How a model starts: the `splash serve` options of `LaunchOption.all` the model picked can use,
/// in one card, the main ones first and the others under Advanced, which opens them below it in the
/// same card, each part under a band of its own. Each is on Auto or the server's own default until
/// changed; a change applies the next time a model starts.
struct LaunchOptions: View {
    @Environment(AppState.self) private var state
    @State var advanced = false
    /// The Advanced row, which opening it scrolls to the top, the options under it in view.
    @State private var row = ScrollTarget()

    var body: some View {
        let family = state.family(of: state.selection)
        GroupedSection {
            ForEach(Self.options(.main, for: family)) { OptionRow(option: $0, family: family) }
            DisclosureRow(title: "Advanced", open: $advanced)
                .background(ViewAnchor { row.view = $0 })
                .onChange(of: advanced) { if advanced { row.scrollToTop() } }
            if advanced {
                ForEach(LaunchOption.Section.allCases.dropFirst(), id: \.self) { section in
                    let options = Self.options(section, for: family)
                    if !options.isEmpty {
                        // A band across the card, as a list groups its rows, so that each part
                        // reads as one.
                        Text(section.rawValue)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(.quaternary.opacity(0.6), in: .rect(cornerRadius: 6))
                        ForEach(options) { OptionRow(option: $0, family: family) }
                    }
                }
            }
        } header: {
            HStack(alignment: .firstTextBaseline) {
                SectionTitle(text: LaunchOption.Section.main.rawValue)
                Spacer()
                // For the options the card shows. Laid out while hidden, so that the header keeps
                // its height: the button is taller than the title, and as it came and went the card
                // under it moved. It shows within `settle` and goes at once.
                let defaults = state.options.defaults(for: family)
                let changed = state.options != defaults
                Button("Restore Defaults") { state.options = defaults }
                    .buttonStyle(.hover)
                    .foregroundStyle(.tint)
                    .font(.callout)
                    .padding(.trailing, -6)
                    .opacity(changed ? 1 : 0)
                    .animation(changed ? Motion.settle : nil, value: changed)
                    .disabled(!changed)
                    .accessibilityHidden(!changed)
            }
        }
    }

    /// The options of `section` the models of `family` can use.
    nonisolated static func options(_ section: LaunchOption.Section, for family: ModelFamily) -> [LaunchOption] {
        LaunchOption.all.filter { $0.section == section && family.uses($0.flag) }
    }
}

extension EnvironmentValues {
    /// Whether the pointer is over the launch option's row the view is in (OptionRow).
    @Entry var optionRowHovered = false
    /// What the launch option of the row the view is in lacks to apply, while it doesn't
    /// (`LaunchOption.lacks`).
    @Entry var optionRowNeeds: String?
}

/// An option's name, with an info button that shows what it does, and what it needs while it is
/// dimmed: the one place its explanation shows. The button appears while the pointer is over the
/// option's row, its explanation is open or it has the keyboard's focus, so the card stays as plain
/// as a list of names; VoiceOver and the keyboard reach it always.
private struct OptionLabel: View {
    let title: String
    let help: String
    @Environment(\.optionRowHovered) private var hovered
    @Environment(\.optionRowNeeds) private var needs
    @State private var shown = false
    @FocusState private var focused: Bool

    var body: some View {
        let visible = hovered || shown || focused
        HStack(spacing: 4) {
            Text(title)
            Button { shown.toggle() } label: {
                Image(systemName: "info.circle").foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .focused($focused)
            .opacity(visible ? 1 : 0)
            .animation(Motion.quick, value: visible)
            // A dimmed option still says what it does, and what it needs to apply.
            .environment(\.isEnabled, true)
            .accessibilityLabel("About \(title)")
            .popover(isPresented: $shown, arrowEdge: .bottom) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(help)
                    if let needs { Text(needs).foregroundStyle(.secondary) }
                }
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 280, alignment: .leading)
                .padding(12)
            }
        }
    }
}

/// A row of a value control labeled with its option's name and info button (OptionLabel).
private struct OptionLabeled<Content: View>: View {
    let title: String
    let help: String
    @ViewBuilder let content: Content

    var body: some View {
        LabeledContent {
            content
        } label: {
            OptionLabel(title: title, help: help)
        }
    }
}

/// One launch option, with the control its kind takes; dimmed where the others leave it no effect,
/// a switch then off, and with what to beware of under it: why the model picked doesn't take the
/// value kept, too, or that YaRN stretches its window to the value.
private struct OptionRow: View {
    @Environment(AppState.self) private var state
    let option: LaunchOption
    /// The family of the model the options are for.
    let family: ModelFamily
    @State private var hovered = false

    var body: some View {
        @Bindable var state = state
        let value = Binding(get: { state.options[option.flag] }, set: { state.options[option.flag] = $0 })
        let applies = option.applies(state.options)
        let takes = { family.takes(option.flag, $0) }
        VStack(alignment: .leading, spacing: 6) {
            Group {
                switch option.kind {
                case .toggle(let on):
                    Toggle(isOn: Binding(get: { applies && (value.wrappedValue == "true") == on },
                                         set: { value.wrappedValue = $0 == on ? "true" : nil })) {
                        OptionLabel(title: option.title, help: option.help)
                    }
                case .choice(let standard, _, let custom):
                    ChoicePicker(title: option.title, help: option.help, standard: standard, choices: option.choices(for: family),
                                 custom: custom, family: family, takes: takes, value: value)
                case .text(let custom, _):
                    TextRow(title: option.title, help: option.help, custom: custom, takes: takes, value: value)
                case .folder:
                    FolderRow(title: option.title, help: option.help, value: value)
                case .secret:
                    TextRow(title: option.title, help: option.help, custom: LaunchOption.Value(prompt: "None", message: "", parse: { $0.isEmpty ? nil : $0 }),
                            takes: takes, secure: true, value: $state.options.apiKey, left: state.keepKey)
                        .disabled(state.keyProblem != nil)
                }
            }
            .disabled(!applies)
            .environment(\.optionRowHovered, hovered)
            .environment(\.optionRowNeeds, option.lacks(state.options))
            .contentShape(.rect)
            .onHover { hovered = $0 }
            // Its value changes at once, as a control's does: animated with the page, the value
            // before and the value chosen cross-faded in the field, one over the other. The control
            // moves with the page as one (`geometryGroup`), laid out in it at once.
            .transaction { $0.animation = nil }
            .geometryGroup()
            if let warning = option.warning(state.options)
                ?? value.wrappedValue.flatMap({ family.refusal(option.flag, $0) ?? family.caution(option.flag, $0) }) {
                // As a row of the page comes and goes, a warning in place of another too: it faded
                // in over the rows moving down to make room for it, and out under those moving up
                // into its place.
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .symbolRenderingMode(.multicolor)
                    .id(warning)
                    .transition(Motion.comingAndGoing)
            }
        }
    }
}

/// A menu of the server's default, the choices, the value set, dimmed where the model picked doesn't
/// take it, and Custom… to type another, which asks for it in the words for a model of `family`
/// and takes only one it `takes`.
private struct ChoicePicker: View {
    let title: String
    let help: String
    let standard: String
    let choices: [(value: String, title: String)]
    let custom: LaunchOption.Value?
    let family: ModelFamily
    let takes: (String) -> Bool
    @Binding var value: String?
    @State private var typing = false

    /// The items' IDs of the default and of Custom…, which are no values.
    private static let none = "\u{0}default", other = "\u{0}custom"

    var body: some View {
        // A value typed in before, which the choices don't hold.
        let typed = value.flatMap { value in
            choices.contains { $0.value == value } ? nil : custom.map { PopupItem(id: value, title: $0.title(value)) }
        }
        let items = [PopupItem(id: Self.none, title: standard, badge: "Default")]
            + choices.map { PopupItem(id: $0.value, title: $0.title) }
            + [typed].compactMap { $0 }
            + (custom == nil ? [] : [PopupItem(id: Self.other, title: "Custom…", separated: true)])
        OptionLabeled(title: title, help: help) {
            // Its row names it.
            PopupMenu(items: items, chosen: value ?? Self.none) { choice in
                if choice == Self.other { typing = true } else { value = choice == Self.none ? nil : choice }
            } label: {
                HStack(spacing: 5) {
                    Text(items.first { $0.id == (value ?? Self.none) }?.title ?? standard)
                        .foregroundStyle(value.map(takes) == false ? .tertiary : .primary)
                    MenuChevron()
                }
                .optionField()
            }
            .accessibilityLabel(title)
            .accessibilityValue(items.first { $0.id == (value ?? Self.none) }?.title ?? standard)
        }
        .sheet(isPresented: $typing) {
            if let custom {
                CustomValueSheet(title: title, custom: custom, message: custom.message(family), takes: takes,
                                 initial: value.map(custom.text) ?? "") { value = $0 }
            }
        }
    }
}

/// A value typed in a field, hidden for a secret, taken as it is typed once it is one the option
/// and the model picked take, so that Start uses it at once; empty for the default. Text they don't
/// take is ringed, and gives way to the value taken when the field is left.
private struct TextRow: View {
    let title: String
    let help: String
    let custom: LaunchOption.Value
    let takes: (String) -> Bool
    var secure = false
    @Binding var value: String?
    /// What the field's leaving does besides, as an API key's reaching the Keychain.
    var left: () -> Void = {}
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        OptionLabeled(title: title, help: help) {
            Group {
                if secure {
                    SecureField(title, text: $text, prompt: Text(custom.prompt))
                } else {
                    TextField(title, text: $text, prompt: Text(custom.prompt))
                }
            }
            .labelsHidden()
            .textFieldStyle(.plain)
            .multilineTextAlignment(.trailing)
            .focused($focused)
            .onSubmit {
                show()
                left()
            }
            .optionField(focused: focused, invalid: taken(text) == nil)
            .frame(maxWidth: 220)
        }
        .onAppear(perform: show)
        .onChange(of: text) { if let taken = taken(text) { value = taken } }
        // Changed elsewhere, as by Restore Defaults.
        .onChange(of: value) { if taken(text) != .some(value) { show() } }
        .onChange(of: focused) {
            if !focused {
                show()
                left()
            }
        }
    }

    /// The value `text` sets: none for empty text, the value parsed, or nil for text the option or
    /// the model picked doesn't take, which sets nothing.
    private func taken(_ text: String) -> String?? {
        text.trimmingCharacters(in: .whitespaces).isEmpty ? .some(nil) : custom.parse(text).flatMap { takes($0) ? $0 : nil }.map(Optional.some)
    }

    /// Shows the value taken.
    private func show() {
        text = value.map(custom.text) ?? ""
    }
}

/// A folder picked, or the server's default.
private struct FolderRow: View {
    let title: String
    let help: String
    @Binding var value: String?

    var body: some View {
        OptionLabeled(title: title, help: help) {
            HStack {
                Spacer(minLength: 0)
                Text(value.map { URL(filePath: $0).lastPathComponent } ?? "Default")
                    .foregroundStyle(.secondary)
                    .truncationMode(.middle)
                    .layoutPriority(1)
                    .help(value ?? "")
                if value != nil {
                    Button { value = nil } label: { Text("Use Default").optionField() }.buttonStyle(.hoverFlush)
                }
                Button {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true
                    panel.canChooseFiles = false
                    panel.canCreateDirectories = true
                    if panel.runModal() == .OK, let url = panel.url { value = url.path }
                } label: {
                    Text("Choose…").optionField()
                }
                .buttonStyle(.hoverFlush)
            }
        }
    }
}

/// Asks for a value the menu doesn't offer, one the model picked `takes`.
private struct CustomValueSheet: View {
    @Environment(\.dismiss) private var dismiss
    let title: String
    let custom: LaunchOption.Value
    let message: String
    let takes: (String) -> Bool
    let initial: String
    let set: (String) -> Void
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        Form {
            Section {
                TextField(title, text: $text, prompt: Text(custom.prompt))
                    .labelsHidden()
                    .focused($focused)
                    .onSubmit(done)
            } header: {
                Text(title).font(.headline)
            } footer: {
                Text(message)
            }
        }
        .formStyle(.grouped)
        .frame(width: 400)
        .onAppear {
            text = initial
            focused = true
        }
        .toolbar {
            ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            ToolbarItem(placement: .confirmationAction) {
                Button("Set", action: done).disabled(value == nil)
            }
        }
    }

    /// The value the text typed sets.
    private var value: String? {
        custom.parse(text).flatMap { takes($0) ? $0 : nil }
    }

    private func done() {
        guard let value else { return }
        set(value)
        dismiss()
    }
}

extension View {
    /// The look the launch options' fields and menus share: a light rounded fill, deeper under the
    /// pointer as `HoverStyle` adds its highlight, and a ring in the accent color while a field is
    /// typed in, in red while it holds what the option doesn't take.
    func optionField(focused: Bool = false, invalid: Bool = false) -> some View {
        padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.quaternary.opacity(0.6), in: .rect(cornerRadius: 7))
            .overlay {
                RoundedRectangle(cornerRadius: 7)
                    .strokeBorder(invalid ? AnyShapeStyle(.red.opacity(0.8)) : AnyShapeStyle(.tint.opacity(focused ? 0.8 : 0)), lineWidth: 1.5)
            }
            .animation(Motion.quick, value: focused)
            .animation(Motion.quick, value: invalid)
    }
}
