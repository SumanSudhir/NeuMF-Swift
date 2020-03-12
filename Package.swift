// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
import PackageDescription

let package = Package(
    name: "NeuMF",
    // platforms: [.macOS(SupportedPlatform.MacOSVersion.v10_13)],
    products: [
        .executable(name: "NeuMF", targets: ["NeuMF"]),
    ],
    dependencies: [
        .package(url: "https://github.com/tensorflow/swift-models", .branch("master"))
    ],
    targets: [
        .target(
            name: "NeuMF",
            dependencies: [.product(name :"Datasets", package: "swift-models")])
    ]
)
