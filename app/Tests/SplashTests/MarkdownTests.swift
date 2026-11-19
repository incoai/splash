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

    /// A table of a narrow first column and two wide ones cut the first's header, Feature, after
    /// its sixth letter, as the page wraps anywhere. No column is narrower than its longest word:
    /// the header is on one line, as the one-letter cells under it are.
    @Test func aTablesColumnIsNeverNarrowerThanItsLongestWord() async throws {
        let heights = try await drawn("""
            const box = document.createElement('div')
            // As narrow as the chat's column at the window's least size, where the two wide
            // columns take the room (the property, as the page's policy refuses a style attribute).
            box.style.width = '520px'
            document.body.append(box)
            const wide = "Annotations are evaluated lazily, which speeds a module's import. ".repeat(3)
            markdown.draw(box, `| Feature | Python 3.14 | Python 3.15 |\\n|---|---|---|\\n| 1 | ${wide} | ${wide} |\\n| 2 | ${wide} | ${wide} |`)
            const heights = [box.querySelector('th > div').getBoundingClientRect().height,
                             box.querySelector('td > div').getBoundingClientRect().height]
            box.remove()
            return heights
            """, as: [Double].self) ?? []
        #expect(heights.count == 2 && heights[0] > 0 && heights[0] == heights[1], "\(heights)")
    }

    /// A table that streams shows as a table from its first frame: its header line, alone or with
    /// part of the line of dashes under it, is a paragraph to the parser, and drew as text with
    /// its bars before the table took its place. It waits until it is one; what is no table after
    /// all draws as the reply ends.
    @Test func aTableThatStreamsNeverShowsItsBars() async throws {
        let counts = try await drawn("""
            const box = document.createElement('div')
            // Of each text as it streams: how many tables it draws, whether a bar shows, and how
            // many blocks; then the last text, whole.
            const stream = (text, live = true) => {
              markdown.draw(box, text, live)
              return [box.querySelectorAll('table').length, box.textContent.includes('|') ? 1 : 0, box.children.length]
            }
            const texts = ['Here they are:\\n\\n| File | Si', 'Here they are:\\n\\n| File | Size |\\n',
                           'Here they are:\\n\\n| File | Size |\\n|--', 'Here they are:\\n| File | Size |\\n|:--|',
                           'Here they are:\\n\\n| File | Size |\\n|---|---|\\n| a.py | 1 K',
                           'Then:\\n\\n|x| is its size']
            return [...texts.flatMap(text => stream(text)), ...stream(texts[5], false)]
            """) ?? []
        // The header alone, with its line's end, and with part of the dashes: the sentence, no bar.
        #expect(Array(counts.prefix(9)) == [0, 0, 1, 0, 0, 1, 0, 0, 1], "\(counts)")
        // Under a sentence with no empty line between: the sentence still, no bar.
        #expect(Array(counts.dropFirst(9).prefix(3)) == [0, 0, 1], "\(counts)")
        // The table once the dashes have a cell for each of the header's, its row as it comes.
        #expect(Array(counts.dropFirst(12).prefix(3)) == [1, 0, 2], "\(counts)")
        // A line of bars that is no table waits while it streams, and draws as the reply ends.
        #expect(Array(counts.dropFirst(15)) == [0, 0, 1, 0, 1, 2], "\(counts)")
    }

    /// A formula shows as the model wrote it, in code's type: \( \), \[ \] and $$ $$ keep their
    /// backslashes, and their * and _ are no emphasis; a price's $ and a tier's $$$ stay text. A
    /// block that streams is one from its first line, never a paragraph first; a line of prose
    /// that only starts with $$ takes its paragraph, not the rest of the reply.
    @Test func mathShowsAsTheModelWroteIt() async throws {
        let checks = try await drawn(#"""
            const box = document.createElement('div')
            const same = (selector, text) => [...box.querySelectorAll(selector)].map(element => element.textContent)
              .join('|') === text ? 1 : 0
            markdown.draw(box, 'So \\(a*b*c_d_e\\) and $$x_1 * y_1$$, for $5 and $10.\n\n\\[\n\\sum_i x_i * y_i\n\\]')
            const whole = [same('code.math', '\\(a*b*c_d_e\\)|$$x_1 * y_1$$'), same('pre.math', '\\[\n\\sum_i x_i * y_i\n\\]'),
                           box.querySelectorAll('em').length, box.textContent.includes('for $5 and $10.') ? 1 : 0]
            const frames = ['Then:\n\n$$', 'Then:\n\n$$\n\\frac{a*b}{c', 'Then:\n\n$$\n\\frac{a*b}{c}\n$$'].map(text => {
              markdown.draw(box, text, true)
              return same('pre.math', text.slice('Then:\n\n'.length)) + box.querySelectorAll('em').length * 10
            })
            markdown.draw(box, '$$ is the shell process ID.\n\nAnother *paragraph*.')
            const prose = [box.querySelectorAll('pre.math').length, box.querySelectorAll('em').length]
            markdown.draw(box, "Joe's ($$) and Luigi's ($$$)")
            const tiers = [box.querySelectorAll('.math').length, box.textContent.includes("Joe's ($$) and Luigi's ($$$)") ? 1 : 0]
            return [...whole, ...frames, ...prose, ...tiers]
            """#) ?? []
        #expect(checks == [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1], "\(checks)")
    }

    /// A code block's copy button, clicked while its reply streams, keeps its check for its time:
    /// the block drawn again as the text grows is the same block, the check no part of the text.
    @Test func aCopiedCheckStaysAsTheReplyStreams() async throws {
        let checks = try await drawn("""
            const box = document.createElement('div')
            document.body.append(box)
            markdown.draw(box, 'Run:\\n\\n```sh\\nmake', true)
            const button = box.querySelector('.copy-code')
            // As its click does (transcript.js).
            page.copy(button, 'make')
            markdown.draw(box, 'Run:\\n\\n```sh\\nmake test', true)
            markdown.draw(box, 'Run:\\n\\n```sh\\nmake test\\n```', true)
            markdown.draw(box, 'Run:\\n\\n```sh\\nmake test\\n```', false)
            const kept = box.querySelector('.copy-code')
            box.remove()
            return [kept === button ? 1 : 0, kept.classList.contains('copied') ? 1 : 0]
            """) ?? []
        #expect(checks == [1, 1], "\(checks)")
    }
}
