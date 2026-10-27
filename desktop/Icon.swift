// Build-time icon drawing; the app itself has no image or third-party dependencies.
import AppKit

let destination = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = size * scale
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
                                      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                      isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        let factor = CGFloat(pixels) / 1024
        let transform = NSAffineTransform()
        transform.scale(by: factor)
        transform.concat()
        NSColor(srgbRed: 0.75, green: 0.36, blue: 0.21, alpha: 1).setFill()
        NSBezierPath(roundedRect: NSRect(x: 80, y: 80, width: 864, height: 864), xRadius: 195, yRadius: 195).fill()
        let bolt = NSBezierPath()
        bolt.move(to: NSPoint(x: 563, y: 811))
        bolt.line(to: NSPoint(x: 313, y: 459))
        bolt.line(to: NSPoint(x: 494, y: 459))
        bolt.line(to: NSPoint(x: 466, y: 213))
        bolt.line(to: NSPoint(x: 716, y: 565))
        bolt.line(to: NSPoint(x: 535, y: 565))
        bolt.close()
        NSColor(srgbRed: 1, green: 0.97, blue: 0.93, alpha: 1).setFill()
        bolt.fill()
        NSGraphicsContext.restoreGraphicsState()
        let suffix = scale == 2 ? "@2x" : ""
        try bitmap.representation(using: .png, properties: [:])!
            .write(to: destination.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
    }
}
