import AppKit
import SwiftUI

/// The app's motion, in three meanings, and the one place SwiftUI's durations and curves are named
/// (their twins keep the same values: AppKit's, in a popup's fade and the scroll to options opened,
/// and the transcript page's `--quick`, `--settle` and glide to the latest message):
/// - `quick`, touch and feedback: a field's ring, a check, the mark's color, a reply's actions
///   coming into view; a SwiftUI hover highlight comes at once, the page's within `quick`;
/// - `settle`, a change of state or shape: a disclosure, rows and sections coming and going, the
///   server's stage across the window, the switch's pill, figures that change, the sidebar and the
///   drawer sliding away and back, the composer moving between a new chat and a conversation;
/// - `glide`, travel: a scroll to the latest message or to options opened, the mark taking its
///   pose as a model becomes ready.
///
/// What appears fades in within `settle`, a section or a row of a page once what makes room for it
/// has moved, and what goes from a page goes at once (`comingAndGoing`), as does what the composer
/// holds over its text, which comes at once where nothing moves (`overText`); a row's words change
/// in place, and one check shows at a time. Nothing slides in but the switch's pill, the sidebar and
/// the drawer. While a reply streams, the page lets its text out over the frames that follow it
/// and eases toward the bottom after it, so that its lines slide in (page.js, `paced` and
/// `follower`); nothing else loops then, as drawing shares the GPU with the model: the reply's
/// counters change once a second without rolling, and the engine's figures, every 2 s, change in
/// place. The swap of the Server and the Chat, the composer's height, a popup's opening and the
/// Server's control bar, whose words and buttons change at once, don't move. With Reduce Motion
/// changes of state and travel happen at once, streaming text shows as it comes and the page
/// follows it at once, and the mark keeps its pose.
@MainActor
enum Motion {
    static var reduced: Bool { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }
    static let quick = Animation.easeOut(duration: 0.12)
    static var settle: Animation? { reduced ? nil : .spring(settleSpring) }
    static var glide: Animation? { reduced ? nil : .smooth(duration: 0.4) }
    static let settleSeconds = 0.24
    /// `settle`'s curve, which a page's scroll follows in step with its content (`AppKitScroll`).
    static let settleSpring = Spring.smooth(duration: settleSeconds)

    /// How a section or a row of a page comes and goes as the state changes: one that goes, goes
    /// at once, and one that comes fades in within `settle` once what makes room for it has moved,
    /// so that nothing fades over what moves; in a switch of models a row faded out over the rows
    /// moving into its place, and one faded in over those still moving away. One that goes is
    /// hidden at once, unanimated: left as it was (`identity`), it stayed drawn where it was until
    /// the change's animations ended, as the suggestion downloaded first on the first run did,
    /// over the rows moving into its place and the download's row fading in there.
    static var comingAndGoing: AnyTransition {
        .asymmetric(insertion: .opacity.animation(settle?.delay(settleSeconds)), removal: .opacity.animation(nil))
    }

    /// How what the composer holds over its text comes and goes, the line that says why and what
    /// is attached: as a section of a page does (`comingAndGoing`) where the change moves anything,
    /// and at once where it moves nothing.
    /// - One that goes is hidden at once, whatever moves with the change that takes it. As the
    ///   server took a new chat's first message, they faded within `settle` where the composer had
    ///   been, over the transcript, while the composer travelled to the foot: in 4 frames drawn in
    ///   its first 80 ms, the picture attached was still there.
    /// - One that comes with a change that moves the composer fades in once the composer has
    ///   moved. As a new chat opened again from another chat, the line that says why its message
    ///   was refused showed at its place in the middle 50 ms on, the composer coming up under it
    ///   still 240 points short.
    /// - One that comes with nothing moving comes at once, as a refusal's reason and a picture
    ///   attached do: faded in after `settle`, as a page's row is, the line that says why came
    ///   into a row the composer had left empty that long.
    static var overText: some Transition { OverText(later: settle?.delay(settleSeconds)) }
}

/// `Motion.overText`.
private struct OverText: Transition {
    /// `settle`, once what moves has moved.
    let later: Animation?

    func body(content: Content, phase: TransitionPhase) -> some View {
        content.transaction { transaction in
            if phase == .didDisappear {
                transaction.animation = nil
            } else if transaction.animation != nil {
                transaction.animation = later
            }
        } body: {
            $0.opacity(phase.isIdentity ? 1 : 0)
        }
    }
}
