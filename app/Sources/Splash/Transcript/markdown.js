// The model's Markdown as the transcript draws it, with the runtime's markdown-it, highlight.js and
// the web chat's link rules (markdown-links.js), which the app adds to the page before its scripts:
// no raw HTML, and its line breaks kept.
'use strict';

const markdown = (() => {
  const escape = text => text.replace(/[&<>"']/g, character => `&#${character.charCodeAt(0)};`);
  // A copy button's symbols, which turn into a check (page.css).
  const copyGlyphs = '<span class="glyph copy"></span><span class="glyph check"></span>';
  const bullets = ['•', '◦', '▪'];

  // None without the link rules too: markdown-it alone draws images, which load by themselves.
  const parser = window.markdownit && window.markdownLinks
    ? configured(window.markdownit({html: false, breaks: true}).use(window.markdownLinks)) : null;

  // What highlight.js made of each closed code block, by its language and code, and the HTML of
  // each text drawn whole, by the text, the latest of each up to its bound: a reply that streams
  // draws its earlier code again on every frame, and a conversation opened again draws its replies
  // again, which then take what was made before.
  const highlights = bounded(400);
  const drawn = bounded(400);

  function bounded(size) {
    const entries = new Map();
    return {
      get(key) {
        const value = entries.get(key);
        if (value !== undefined) {
          entries.delete(key);
          entries.set(key, value);
        }
        return value;
      },
      set(key, value) {
        entries.delete(key);
        entries.set(key, value);
        if (entries.size > size) entries.delete(entries.keys().next().value);
      },
    };
  }

  function configured(parser) {
    parser.core.ruler.push('splash', state => {
      let cell = false;
      for (const token of state.tokens) {
        // A list item's text is a paragraph, in a tight list or a loose one.
        if (token.type === 'paragraph_open' || token.type === 'paragraph_close') token.hidden = false;
        if (token.type === 'th_open' || token.type === 'td_open') cell = true;
        if (token.type === 'th_close' || token.type === 'td_close') cell = false;
        if (cell && token.type === 'inline') token.children = lineBreaks(token.children, state);
      }
    });
    const rules = parser.renderer.rules;
    // A fence the text leaves open, `env.open`, streams still: its code shows as it is until it
    // closes, and is highlighted once.
    rules.fence = (tokens, index, options, env) =>
      code(tokens[index].content, parser.utils.unescapeAll(tokens[index].info).trim().split(/\s+/)[0], index === env.open);
    rules.code_block = (tokens, index) => code(tokens[index].content, '');
    // Markers in a column of their own, as wide as the widest, so that the items' text lines up;
    // numbered from the list's start, bullets by how deep the list is.
    rules.bullet_list_open = (tokens, index, options, env) => {
      env.lists.push({ordered: false});
      return '<ul>';
    };
    rules.ordered_list_open = (tokens, index, options, env) => {
      env.lists.push({ordered: true, next: Number(tokens[index].attrGet('start') ?? 1)});
      return '<ol>';
    };
    rules.bullet_list_close = rules.ordered_list_close = (tokens, index, options, env) =>
      env.lists.pop().ordered ? '</ol>' : '</ul>';
    rules.list_item_open = (tokens, index, options, env) => {
      const list = env.lists.at(-1);
      const marker = list.ordered ? `${list.next++}.` : bullets[(env.lists.length - 1) % bullets.length];
      return `<li><span class="marker">${marker}</span><div class="item">`;
    };
    rules.list_item_close = () => '</div></li>';
    // A table scrolls sideways when it is wider than the reply; its cells wrap at a width of their
    // own and line up at their start.
    rules.table_open = (tokens, index, options, env, self) => '<div class="table">' + self.renderToken(tokens, index, options);
    rules.table_close = (tokens, index, options, env, self) => self.renderToken(tokens, index, options) + '</div>';
    for (const cell of ['th', 'td']) {
      rules[`${cell}_open`] = (tokens, index, options, env, self) => {
        tokens[index].attrs = null;
        return self.renderToken(tokens, index, options) + '<div>';
      };
      rules[`${cell}_close`] = (tokens, index, options, env, self) => '</div>' + self.renderToken(tokens, index, options);
    }
    return parser;
  }

  // A table cell's <br>s as line breaks, as GitHub shows them.
  function lineBreaks(children, state) {
    return children.flatMap(token => token.type !== 'text' ? [token] : token.content.split(/<br\s*\/?>/i).flatMap((part, index) => [
      ...(index > 0 ? [new state.Token('hardbreak', 'br', 0)] : []),
      Object.assign(new state.Token('text', '', 0), {content: part}),
    ]));
  }

  // Code, outlined, under its language and Copy; highlighted when highlight.js knows the language,
  // but while it streams, `open`.
  function code(source, language, open) {
    const text = source.replace(/\n$/, '');
    return `<figure class="code"><figcaption><span>${escape(language)}</span>` +
      `<button type="button" class="hover copy-code" title="Copy" aria-label="Copy Code">${copyGlyphs}</button></figcaption>` +
      `<pre><code>${open ? escape(text) : highlighted(text, language)}</code></pre></figure>`;
  }

  function highlighted(text, language) {
    if (!language || !window.hljs?.getLanguage(language)) return escape(text);
    const key = `${language}\n${text}`;
    let html = highlights.get(key);
    if (html === undefined) {
      html = window.hljs.highlight(text, {language, ignoreIllegals: true}).value;
      highlights.set(key, html);
    }
    return html;
  }

  // The HTML of `text`; with `open`, its last fence left open, as it streams.
  function render(text, open = false) {
    try {
      const env = {lists: [], open: -1};
      const tokens = parser.parse(text, env);
      if (open) env.open = tokens.findLastIndex(token => token.type === 'fence');
      return parser.renderer.render(tokens, parser.options, env);
    } catch {
      // Never nothing: the text as it came, should Markdown be missing or fail.
      return `<p>${escape(text)}</p>`;
    }
  }

  // Where `text` can be cut, from `start`, which is such a place, so that what comes before it
  // draws alone as it does in the whole: the start of the last block the parser finds at the top
  // level, after a blank line and at the column's start, and no list item, as a list goes on past
  // a blank line. A line that might still become an item, "1" or "-", counts as one. Also whether
  // the text ends in a fence that is still open: one that reaches its end with no closing line,
  // all of its lines but the first being its code.
  function scan(text, start) {
    const rest = text.slice(start);
    let tokens = [];
    // None where Markdown is missing or fails, as `render` then shows the text as it came.
    try {
      tokens = parser.parse(rest, {});
    } catch {}
    const blocks = new Set(tokens.filter(token => token.level === 0 && token.map).map(token => token.map[0]));
    let cut = start;
    let at = start;
    let blank = false;
    let number = 0;
    // The lines as the text breaks them, which the parser numbers alike, and an empty one after them.
    for (const [line] of rest.matchAll(/[^\r\n]*(?:\r\n?|\n|$)/g)) {
      if (blank && blocks.has(number) && /^\S/.test(line) && !/^([-+*]|\d{1,9}[.)]?)(\s|$)/.test(line)) cut = at;
      blank = line.trim() === '';
      at += line.length;
      number++;
    }
    // The fence's code lines: its content ends each with a line break, but the last one where the
    // text ends mid-line. It reaches the end of the text as the parser counts its lines, which
    // leaves out a last line of only spaces, such as the indent of a line of code just begun. It
    // counts them once each \r\n or \r is a \n, as `parse` makes it.
    const fence = tokens.findLast(token => token.type === 'fence');
    const code = fence?.content === '' ? 0 : fence?.content.replace(/\n$/, '').split('\n').length;
    const lines = fence ? new parser.block.State(rest.replace(/\r\n?/g, '\n'), parser, {}, []).lineMax : 0;
    return {cut, open: fence?.map[1] === lines && fence.map[1] - fence.map[0] === code + 1};
  }

  // Draws `text` into `container` as Markdown. Whole, it changes only the nodes that differ from
  // what the container shows, so that a reply drawn again keeps its selection and its place. While
  // it streams, `live`, what it has finished, up to where `scan` cuts it, stays as it was drawn,
  // and only the rest is drawn again: each frame draws the end of a reply, never all of it.
  function draw(container, text, live = false) {
    container.wanted = undefined;
    const last = container.markdown;
    if (last?.text === text && last.live === live) return;
    if (!live) {
      let html = drawn.get(text);
      if (html === undefined) {
        html = render(text);
        drawn.set(text, html);
      }
      replace(container, 0, nodes(html));
      container.markdown = {text, live, frozen: '', count: 0};
      return;
    }
    const kept = last?.live && text.startsWith(last.frozen) ? last : {frozen: '', count: 0};
    const {cut, open} = scan(text, kept.frozen.length);
    const finished = cut > kept.frozen.length ? nodes(render(text.slice(kept.frozen.length, cut))) : [];
    replace(container, kept.count, [...finished, ...nodes(render(text.slice(cut), open))]);
    container.markdown = {text, live, frozen: text.slice(0, cut), count: kept.count + finished.length};
  }

  // Draws `text` as `draw` does, whole, but later where it has not been drawn before, as the
  // replies of a conversation that opens that are above its first screen: in slices of a few
  // milliseconds after the page has shown, the latest asked first.
  const later = (() => {
    const waiting = [];
    function work() {
      const until = performance.now() + 8;
      while (waiting.length > 0 && performance.now() < until) {
        const container = waiting.pop();
        if (container.isConnected && container.wanted !== undefined) draw(container, container.wanted);
      }
      if (waiting.length > 0) setTimeout(work);
    }
    return (container, text) => {
      if (drawn.get(text) !== undefined) return draw(container, text);
      container.wanted = text;
      waiting.push(container);
      if (waiting.length === 1) setTimeout(work);
    };
  })();

  // The blocks of `html`, without the line breaks between them.
  function nodes(html) {
    const template = document.createElement('template');
    template.innerHTML = html;
    return [...template.content.children];
  }

  // `fresh` in place of the nodes of `parent` from `from` on, each node that is there already
  // changed only where it differs: a code block that streams keeps its button, under the
  // pointer or not, and its text node, which only grows.
  function replace(parent, from, fresh) {
    const old = parent.childNodes;
    fresh.forEach((node, index) => {
      const there = old[from + index];
      if (there) morph(there, node);
      else parent.append(node);
    });
    while (old.length > from + fresh.length) parent.lastChild.remove();
  }

  function morph(node, fresh) {
    if (node.nodeType !== fresh.nodeType || node.nodeName !== fresh.nodeName) return node.replaceWith(fresh);
    if (node.nodeType !== Node.ELEMENT_NODE) {
      if (node.data !== fresh.data) node.data = fresh.data;
      return;
    }
    for (const {name} of [...node.attributes]) if (!fresh.hasAttribute(name)) node.removeAttribute(name);
    for (const {name, value} of fresh.attributes) if (node.getAttribute(name) !== value) node.setAttribute(name, value);
    replace(node, 0, [...fresh.childNodes]);
  }

  return {draw, later, copyGlyphs};
})();
