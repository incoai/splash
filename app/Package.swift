// swift-tools-version: 6.2
// The Splash desktop app: a window and a menu bar item around `splash serve`.
// `make app` builds it and assembles Splash.app around the packaged runtime.

import PackageDescription

let package = Package(
    name: "Splash",
    platforms: [.macOS("26.4")],
    dependencies: [
        // Updates the app from the appcast of the GitHub releases (docs/package.md, Updates).
        // Pinned to one release: its tools sign the appcast the app checks, so a new one is taken
        // on purpose.
        .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.10.1"),
        // The drawer's terminal: a native view over a pseudo-terminal, MIT licensed, pinned as
        // Sparkle is.
        .package(url: "https://github.com/migueldeicaza/SwiftTerm", exact: "1.20.0"),
    ],
    targets: [
        .executableTarget(
            name: "Splash",
            dependencies: [.product(name: "Sparkle", package: "Sparkle"), .product(name: "SwiftTerm", package: "SwiftTerm")],
            path: "Sources/Splash",
            // The Chat's transcript is a web page, in Transcript/.
            resources: [.copy("Transcript")],
            // Where the executable finds Sparkle.framework: in Contents/Frameworks of Splash.app, and
            // beside it in .build, for a build run from there.
            linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks",
                                           "-Xlinker", "-rpath", "-Xlinker", "@executable_path"])]
        ),
        // The fake servers the tests run, Python scripts in Fakes/.
        .testTarget(name: "SplashTests", dependencies: ["Splash"], path: "Tests/SplashTests", resources: [.copy("Fakes")]),
    ]
)
