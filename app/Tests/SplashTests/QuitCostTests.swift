import Testing

@testable import Splash

struct QuitCostTests {
    /// Quitting asks while a model runs, sleeps, starts or restarts, and says which.
    @MainActor @Test func quittingCostsWhatItStops() {
        func cost(_ phase: Server.Phase, asleep: Bool = false, next: String? = nil, downloading: String? = nil,
                  terminals: Int = 0) -> AppState.QuitCost? {
            AppState.quitCost(phase: phase, asleep: asleep, model: "27B", next: next, downloading: downloading,
                              terminals: terminals)
        }
        let running = "27B is running. Quitting stops it and interrupts any connected clients."
        let download = "Flash is still downloading. If you quit, the download starts over next time."
        let sleeping = "27B is sleeping. Quitting stops it, and connected clients can’t wake it."
        #expect(cost(.ready) == .init(text: running, button: "Stop Model and Quit"))
        #expect(cost(.ready, asleep: true) == .init(text: sleeping, button: "Stop Model and Quit"))
        #expect(cost(.starting)?.text == "27B is still starting. Quitting cancels the startup and the in-progress model load.")
        #expect(cost(.stopping, next: "27B")?.text == "27B is restarting. Quitting cancels the restart and stops the server.")
        #expect(cost(.stopping, next: "Flash")?.text == "Splash is switching to Flash. Quitting cancels the switch and stops the server.")
        // Quits at once.
        #expect(cost(.stopping) == nil)
        #expect(cost(.stopped) == nil)
        #expect(cost(.failed("No")) == nil)
        // A download counts on its own, and beside a model.
        #expect(cost(.stopped, downloading: "Flash") == .init(text: download, button: "Quit"))
        #expect(cost(.ready, downloading: "Flash") == .init(text: "\(running)\n\n\(download)", button: "Stop Model and Quit"))
        // Terminals running a program count on their own, and beside a model.
        #expect(cost(.stopped, terminals: 1) == .init(text: "A terminal is running a program. Quitting ends it.", button: "Quit"))
        #expect(cost(.stopped, terminals: 2)?.text == "2 terminals are running programs. Quitting ends them.")
        #expect(cost(.ready, terminals: 1)?.text == "\(running)\n\nA terminal is running a program. Quitting ends it.")
        #expect(cost(.ready, terminals: 1)?.button == "Stop Model and Quit")
    }
}
