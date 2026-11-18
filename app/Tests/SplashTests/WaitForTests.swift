import Foundation
import Testing

@MainActor
struct WaitForTests {
    /// A main-actor callback queued by the last poll gets a turn before the deadline fails.
    /// UI snapshot work can hold that actor beyond the wait's deadline.
    @Test func aQueuedCallbackGetsTheFinalPoll() async throws {
        var ready = false
        var queued = false
        func condition() -> Bool {
            if !queued {
                queued = true
                Task { @MainActor in ready = true }
            }
            return ready
        }
        try await waitFor(condition(), "the queued callback", seconds: 0)
        #expect(ready)
    }

    @Test func anAlreadyTrueConditionDoesNotWait() async throws {
        try await waitFor(true, seconds: 0)
    }

    @Test func aConditionThatStaysFalseStillTimesOut() async {
        await withKnownIssue("The false condition must fail its bounded wait") {
            try await waitFor(false, "the deliberately false condition", seconds: 0)
        }
    }
}
