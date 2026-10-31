import CoreGraphics
import Foundation
import ImageIO

final class MarkParser: NSObject, XMLParserDelegate {
    var polygons: [[CGPoint]] = []

    func parser(_ parser: XMLParser, didStartElement elementName: String,
                namespaceURI: String?, qualifiedName qName: String?,
                attributes attributeDict: [String: String]) {
        guard elementName == "polygon", let points = attributeDict["points"] else { return }
        let tokens = points.split { $0.isWhitespace || $0 == "," }
        let values = tokens.compactMap(Double.init)
        guard values.count == tokens.count, values.count >= 6, values.count.isMultiple(of: 2) else {
            return
        }
        let polygon = stride(from: 0, to: values.count, by: 2).map {
            CGPoint(x: values[$0], y: values[$0 + 1])
        }
        polygons.append(polygon)
    }
}

guard CommandLine.arguments.count == 2 else {
    fatalError("Usage: swift generate_brand_assets.swift <desktop-directory>")
}

let desktop = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true).standardizedFileURL
let source = desktop.appendingPathComponent("branding/splash-mark.svg")
guard let parser = XMLParser(contentsOf: source) else { fatalError("Cannot open the Splash mark") }
let mark = MarkParser()
parser.delegate = mark
guard parser.parse(), mark.polygons.count == 3, mark.polygons.allSatisfy({ $0.count >= 3 }) else {
    fatalError("Invalid Splash mark: \(parser.parserError?.localizedDescription ?? "unknown error"), \(mark.polygons.map(\.count))")
}

let assets = desktop.appendingPathComponent("Splash/Assets.xcassets")
let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
let brand = CGColor(colorSpace: colorSpace, components: [227.0 / 255.0, 131.0 / 255.0,
                                                         101.0 / 255.0, 1])!
let surface = CGColor(colorSpace: colorSpace, components: [248.0 / 255.0, 250.0 / 255.0,
                                                           249.0 / 255.0, 1])!
let markWidth = 36.1615
let markHeight = 35.7978

func render(size: Int, icon: Bool, destination: URL) throws {
    guard let context = CGContext(data: nil, width: size, height: size,
                                  bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        fatalError("Cannot create icon bitmap")
    }
    context.setShouldAntialias(true)
    context.clear(CGRect(x: 0, y: 0, width: size, height: size))

    if icon {
        context.setFillColor(surface)
        let radius = CGFloat(size) * 0.22
        context.addPath(CGPath(roundedRect: CGRect(x: 0, y: 0, width: size, height: size),
                               cornerWidth: radius, cornerHeight: radius, transform: nil))
        context.fillPath()
    }

    let width = CGFloat(size) * (icon ? 0.64 : 1.0)
    let height = width * markHeight / markWidth
    let bounds = CGRect(x: (CGFloat(size) - width) / 2,
                        y: (CGFloat(size) - height) / 2, width: width, height: height)
    context.setFillColor(brand)
    for polygon in mark.polygons {
        let path = CGMutablePath()
        for (index, point) in polygon.enumerated() {
            let mapped = CGPoint(x: bounds.minX + point.x / markWidth * bounds.width,
                                 y: bounds.maxY - point.y / markHeight * bounds.height)
            if index == 0 { path.move(to: mapped) }
            else { path.addLine(to: mapped) }
        }
        path.closeSubpath()
        context.addPath(path)
        context.fillPath()
    }
    guard let image = context.makeImage(),
          let output = CGImageDestinationCreateWithURL(destination as CFURL, "public.png" as CFString,
                                                       1, nil) else {
        fatalError("Cannot encode icon bitmap")
    }
    CGImageDestinationAddImage(output, image, nil)
    guard CGImageDestinationFinalize(output) else { fatalError("Cannot write icon bitmap") }
}

let markFolder = assets.appendingPathComponent("SplashMark.imageset")
try render(size: 64, icon: false, destination: markFolder.appendingPathComponent("splash-mark.png"))
try render(size: 128, icon: false, destination: markFolder.appendingPathComponent("splash-mark@2x.png"))

let menuFolder = assets.appendingPathComponent("SplashMenuMark.imageset")
try render(size: 16, icon: false, destination: menuFolder.appendingPathComponent("splash-menu-mark.png"))
try render(size: 32, icon: false, destination: menuFolder.appendingPathComponent("splash-menu-mark@2x.png"))

let iconFolder = assets.appendingPathComponent("AppIcon.appiconset")
for (name, pixels) in [("16x16", 16), ("16x16@2x", 32), ("32x32", 32),
                       ("32x32@2x", 64), ("128x128", 128), ("128x128@2x", 256),
                       ("256x256", 256), ("256x256@2x", 512), ("512x512", 512),
                       ("512x512@2x", 1024)] {
    try render(size: pixels, icon: true,
               destination: iconFolder.appendingPathComponent("icon_\(name).png"))
}
