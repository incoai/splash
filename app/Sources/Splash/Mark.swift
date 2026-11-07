import AppKit
import SwiftUI

/// Splash's mark, as the app icon draws it (app/AppIcon.icon/Assets/mark.svg): three arcs of water
/// springing from one point of a rippled surface, two thrown to the right and one to the left, each
/// throwing a drop. `spread` is how far the splash has risen, from 0, the surface alone, to 1, the
/// icon's pose; the mark keeps its place and size at every spread.
struct SplashMark: Shape {
    var spread = 1.0

    var animatableData: Double {
        get { spread }
        set { spread = newValue }
    }

    // On the icon's 1024-point canvas, drawn in strokes 76 points wide with round ends.
    private static let surface = Path { path in
        path.move(to: CGPoint(x: 200, y: 717.2))
        path.addCurve(to: CGPoint(x: 398, y: 717.2), control1: CGPoint(x: 270, y: 699.2), control2: CGPoint(x: 340, y: 699.2))
        path.addCurve(to: CGPoint(x: 606, y: 717.2), control1: CGPoint(x: 444, y: 733.2), control2: CGPoint(x: 560, y: 733.2))
        path.addCurve(to: CGPoint(x: 824, y: 717.2), control1: CGPoint(x: 670, y: 699.2), control2: CGPoint(x: 744, y: 699.2))
    }

    /// Each arc, from its root in the surface, and the drop it throws.
    private static let arcs: [(arc: Path, drop: CGPoint, radius: CGFloat)] = [
        (arc(CGPoint(x: 502, y: 711.2), CGPoint(x: 500, y: 593.2), CGPoint(x: 534, y: 471.2), CGPoint(x: 604, y: 399.2)),
         CGPoint(x: 684.3, y: 312.8), 36),
        (arc(CGPoint(x: 516, y: 711.2), CGPoint(x: 544, y: 633.2), CGPoint(x: 606, y: 567.2), CGPoint(x: 700, y: 547.2)),
         CGPoint(x: 805.2, y: 522.6), 30),
        (arc(CGPoint(x: 490, y: 711.2), CGPoint(x: 466, y: 653.2), CGPoint(x: 424, y: 621.2), CGPoint(x: 364, y: 609.2)),
         CGPoint(x: 267.2, y: 588.4), 25),
    ]

    /// The icon's pose, by which the mark is placed.
    private static let bounds = CGRect(x: 162, y: 276.8, width: 700, height: 490.4)

    private static func arc(_ root: CGPoint, _ control1: CGPoint, _ control2: CGPoint, _ end: CGPoint) -> Path {
        Path { path in
            path.move(to: root)
            path.addCurve(to: end, control1: control1, control2: control2)
        }
    }

    func path(in rect: CGRect) -> Path {
        let stroke = StrokeStyle(lineWidth: 76, lineCap: .round)
        var mark = Self.surface.strokedPath(stroke)
        // The drops come as the arcs reach their full height.
        let drop = min(max((spread - 0.75) / 0.25, 0), 1)
        for (arc, center, radius) in Self.arcs {
            mark = mark.union(arc.trimmedPath(from: 0, to: spread).strokedPath(stroke))
            if drop > 0 {
                let r = radius * drop
                mark = mark.union(Path(ellipseIn: CGRect(x: center.x - r, y: center.y - r, width: 2 * r, height: 2 * r)))
            }
        }
        // Placed as the icon's pose fits rect, centred.
        let bounds = Self.bounds
        let scale = min(rect.width / bounds.width, rect.height / bounds.height)
        return mark.applying(CGAffineTransform(translationX: -bounds.midX, y: -bounds.midY)
            .concatenating(CGAffineTransform(scaleX: scale, y: scale))
            .concatenating(CGAffineTransform(translationX: rect.midX, y: rect.midY)))
    }

    /// The mark as the menu bar shows the app: in the menu bar's own color, faint until a model
    /// serves.
    static func menuBarImage(serving: Bool) -> NSImage {
        let image = NSImage(size: NSSize(width: 20, height: 16), flipped: true) { rect in
            NSColor.black.withAlphaComponent(serving ? 1 : 0.45).setFill()
            NSBezierPath(cgPath: SplashMark().path(in: rect.insetBy(dx: 1, dy: 1)).cgPath).fill()
            return true
        }
        image.isTemplate = true
        return image
    }
}

extension ShapeStyle where Self == Color {
    /// The mark's color: the icon's aqua, deepened on a light ground so that it stands out there.
    static var splash: Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
                ? NSColor(displayP3Red: 0.345, green: 0.902, blue: 0.878, alpha: 1)
                : NSColor(displayP3Red: 0.055, green: 0.604, blue: 0.588, alpha: 1)
        })
    }
}

/// What the server does, as the mark shows it (`StatusMark`).
enum MarkLook: Equatable {
    case off, loading, on, asleep, failed
}

/// The server's state in the mark, in a slot of its own size: the surface alone, grey, while nothing
/// runs; the splash rising and falling back while a model starts, or while the engine starts again
/// after failing; risen, in the icon's color, while one serves; faint while it sleeps; a warning
/// sign when it failed. It moves only then, never while the model answers. The words beside it say
/// what it shows.
struct StatusMark: View {
    let look: MarkLook
    /// When the model began to start, where the slide starts lined up.
    let since: Date
    var size: CGFloat = 14

    var body: some View {
        Group {
            if look == .failed {
                Image(systemName: "exclamationmark.triangle.fill")
                    .symbolRenderingMode(.multicolor)
                    .font(.system(size: size - 1))
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 30, paused: look != .loading || Motion.reduced)) { context in
                    let spread = spread(at: context.date)
                    ZStack {
                        SplashMark(spread: spread)
                            .fill(look == .off ? HierarchicalShapeStyle.tertiary : .secondary)
                            .opacity(look == .on || look == .asleep ? 0 : 1)
                        SplashMark(spread: spread)
                            .fill(.splash)
                            .opacity(look == .on ? 1 : look == .asleep ? 0.4 : 0)
                    }
                    .frame(width: size, height: size)
                }
            }
        }
        .frame(width: size + 2, height: size + 2)
        .animation(Motion.glide ?? Motion.quick, value: look)
        // Its look changes its pose and color, not its place, which moves with the views beside
        // it: SwiftUI otherwise animates where each part of it draws with its look.
        .geometryGroup()
        .accessibilityHidden(true)
    }

    /// How far the splash has risen at `date`: up and back every 1.6 s while loading, risen while
    /// serving, the surface alone otherwise; risen throughout with Reduce Motion.
    private func spread(at date: Date) -> Double {
        if Motion.reduced { return 1 }
        switch look {
        case .loading:
            let wave = sin(Double.pi * date.timeIntervalSince(since) / 1.6)
            return wave * wave
        case .on, .asleep: return 1
        case .off, .failed: return 0
        }
    }
}
