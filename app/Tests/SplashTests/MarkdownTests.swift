import AppKit
import Testing
import WebKit

@testable import Splash

/// How the transcript's page draws a reply's Markdown (markdown.js, markdown.css): each drawn into a
/// box in the page, as a reply is, whole or as it streams.
extension TranscriptTests {
    /// What `script`, run in the page, gives: numbers, or strings joined.
    func drawn<Value>(_ script: String, as type: Value.Type = [Int].self) async throws -> Value? {
        let (_, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        return try await webView.callAsyncJavaScript(script, contentWorld: .page) as? Value
    }

    /// A table's column marked right or center aligned (`--:`, `:-:`) shows so. markdown-it puts
    /// `style="text-align:right"` on the cells, which the page's Content-Security-Policy
    /// (`style-src 'self'`) refuses, so every column showed left aligned; markdown.js marks them
    /// by a class.
    @Test func aTablesAlignedColumnsShowAligned() async throws {
        let aligns = try await drawn("""
            const box = document.createElement('div')
            document.body.append(box)
            markdown.draw(box, '| Model | Memory | Speed | Notes |\\n|:------|-------:|:-----:|-------|\\n| Small | 16 GB | 95 tok/s | quick |')
            const aligns = [...box.querySelectorAll('td')].map(c => getComputedStyle(c).textAlign).join(',')
            box.remove()
            return aligns
            """, as: String.self)
        #expect(aligns == "left,right,center,left")
    }

    /// Bold beside Chinese text shows bold: asterisks whose inner side is punctuation, a full-width
    /// bracket, and whose outer side is a Chinese character kept as asterisks by CommonMark's rule,
    /// which wants a space or punctuation there. Emphasis in English, and asterisks that mark
    /// nothing, stay as they were.
    @Test func boldBesideChineseTextShowsBold() async throws {
        let shown = try await drawn("""
            const box = document.createElement('div')
            markdown.draw(box, '信号量允许**指定数量（N）**的线程同时访问，而**“互斥锁”**只允许一个。\\n\\nA **bold** word, 2 * 3 * 4, and **（括号）**开头。')
            return [[...box.querySelectorAll('strong')].map(e => e.textContent).join('|'), box.textContent]
            """, as: [String].self)
        #expect(shown?.first == "指定数量（N）|“互斥锁”|bold|（括号）")
        #expect(shown?.last.map { !$0.contains("**") && $0.contains("2 * 3 * 4") } == true, "\(shown ?? [])")
    }
}
