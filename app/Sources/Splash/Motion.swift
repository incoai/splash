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
/// What appears fades in within `settle`; nothing slides in but the switch's pill, the sidebar and
/// the drawer. While a reply streams, the page lets its text out over the frames that follow it
/// and eases toward the bottom after it, so that its lines slide in (page.js, `paced` and
/// `follower`); nothing else loops then, as drawing shares the GPU with the model: the reply's
/// counters change once a second without rolling, and the engine's figures, every 2 s, change in
/// place. The swap of the Server and the Chat, the composer's height and a popup's opening don't
/// move. With Reduce Motion changes of state and travel happen at once, streaming text shows as it
/// comes and the page follows it at once, and the mark keeps its pose.
@MainActor
enum Motion {
    static var reduced: Bool { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }
    static let quick = Animation.easeOut(duration: 0.12)
    static var settle: Animation? { reduced ? nil : .smooth(duration: 0.24) }
    static var glide: Animation? { reduced ? nil : .smooth(duration: 0.4) }
}
