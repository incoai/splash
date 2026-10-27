// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SplashDesktop",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "SplashDesktop", targets: ["SplashDesktop"])],
    targets: [
        .target(name: "SplashChat"),
        .executableTarget(name: "SplashDesktop", dependencies: ["SplashChat"]),
        .testTarget(name: "SplashChatTests", dependencies: ["SplashChat"]),
    ]
)
