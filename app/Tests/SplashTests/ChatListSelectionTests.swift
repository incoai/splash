import Foundation
import Testing

@testable import Splash

struct ChatListSelectionTests {
    @Test func clicksPickARangeOrTheChatsThemselves() {
        let first = UUID(), second = UUID(), third = UUID(), fourth = UUID()
        let order = [first, second, third, fourth]
        var selection = ChatListSelection()
        selection.follow(first)
        #expect(selection.ids == [first])

        // A plain click opens the chat and picks it alone.
        let opened = selection.click(third, command: false, shift: false, in: order)
        #expect(opened)
        #expect(selection.ids == [third])

        // Command-click picks and unpicks one; the next Shift-click extends from it.
        let toggled = selection.click(first, command: true, shift: false, in: order)
        #expect(!toggled)
        #expect(selection.ids == [first, third])
        let extended = selection.click(fourth, command: false, shift: true, in: order)
        #expect(!extended)
        #expect(selection.ids == [first, second, third, fourth])
        let shortened = selection.click(second, command: false, shift: true, in: order)
        #expect(!shortened)
        #expect(selection.ids == [first, second])

        // A Shift-click in the other direction picks the rows between.
        selection.follow(fourth)
        let reversed = selection.click(second, command: false, shift: true, in: order)
        #expect(!reversed)
        #expect(selection.ids == [second, third, fourth])

        // Without an anchor, a Shift-click is a plain one.
        selection.clear()
        let withoutAnchor = selection.click(third, command: false, shift: true, in: order)
        #expect(withoutAnchor)
        #expect(selection.ids == [third])
    }

    /// A delete takes only the rows in view: a search can hide the open chat, which stays picked.
    @Test func picksAmongTheRowsShown() {
        let open = UUID(), shown = UUID(), other = UUID()
        var selection = ChatListSelection()
        selection.follow(open)
        selection.click(shown, command: true, shift: false, in: [shown, other])
        #expect(selection.ids == [open, shown])
        #expect(selection.picked(in: [shown, other]) == [shown])
        #expect(selection.picked(in: [open, shown, other]) == [open, shown])
    }

    /// The row a Shift-click extends from goes with its chat, and the next one is a plain click.
    @Test func aDeletedAnchorLeavesNoneToExtendFrom() {
        let first = UUID(), second = UUID(), third = UUID()
        let order = [first, second, third]
        var selection = ChatListSelection()
        selection.follow(first)
        selection.click(second, command: true, shift: false, in: order)
        selection.click(third, command: true, shift: false, in: order)
        selection.click(third, command: true, shift: false, in: order)
        #expect(selection.anchor == third && selection.ids == [first, second])
        selection.prune(to: [first, second])
        #expect(selection.anchor == nil && selection.ids == [first, second])
        let opened = selection.click(second, command: false, shift: true, in: [first, second])
        #expect(opened)
        #expect(selection.ids == [second])
    }

    @Test func followsAndPrunesChats() {
        let first = UUID(), second = UUID(), third = UUID()
        var selection = ChatListSelection()
        selection.follow(first)
        selection.click(second, command: true, shift: false, in: [first, second, third])
        #expect(selection.ids == [first, second])
        selection.prune(to: [first, third])
        #expect(selection.ids == [first])
        selection.follow(third)
        #expect(selection.ids == [third])
        selection.clear()
        #expect(selection.ids.isEmpty)
    }
}
