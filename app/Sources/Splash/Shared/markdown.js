// The model's Markdown as the app's pages draw it, with the runtime's markdown-it and highlight.js,
// which the app adds to each page before its scripts: no raw HTML, its line breaks kept, only web
// and mail links, and an image as a link to it, so that what the model writes never loads anything
// by itself.
'use strict';

const markdown = (() => {
  const escape = text => text.replace(/[&<>"']/g, character => `&#${character.charCodeAt(0)};`);
  const web = url => /^(https?|mailto):/i.test(url ?? '');
  // A copy button's symbols, which turn into a check (page.css).
  const copyGlyphs = '<span class="glyph copy"></span><span class="glyph check"></span>';
  const bullets = ['•', '◦', '▪'];

  const parser = window.markdownit ? configured(window.markdownit({html: false, breaks: true})) : null;

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
    // Every link reaches the rule below, which keeps web and mail links and the text of others.
    parser.validateLink = () => true;
    parser.core.ruler.push('splash', state => {
      let cell = false;
      for (const token of state.tokens) {
        // A list item's text is a paragraph, in a tight list or a loose one.
        if (token.type === 'paragraph_open' || token.type === 'paragraph_close') token.hidden = false;
        if (token.type === 'th_open' || token.type === 'td_open') cell = true;
        if (token.type === 'th_close' || token.type === 'td_close') cell = false;
        if (token.type !== 'inline') continue;
        token.children = links(token.children, state);
        if (cell) token.children = lineBreaks(token.children, state);
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

  // The inline tokens with only web and mail links left links, and images as links to them.
  function links(children, state) {
    const kept = [];
    const dropped = [];
    for (const token of children) {
      if (token.type === 'link_open') {
        dropped.push(!web(token.attrGet('href')));
        if (!dropped.at(-1)) kept.push(token);
      } else if (token.type === 'link_close') {
        if (!dropped.pop()) kept.push(token);
      } else if (token.type === 'image') {
        const text = Object.assign(new state.Token('text', '', 0), {content: state.md.renderer.renderInlineAsText(token.children, {}, {})});
        // An image in a link is that link's text.
        if (web(token.attrGet('src')) && !dropped.includes(false)) {
          const open = new state.Token('link_open', 'a', 1);
          open.attrs = [['href', token.attrGet('src')]];
          kept.push(open, text, new state.Token('link_close', 'a', -1));
        } else {
          kept.push(text);
        }
      } else {
        kept.push(token);
      }
    }
    return kept;
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
  // draws alone as it does in the whole: the start of the last line after a blank line that
  // starts at the column's start, outside any fence, and no list item, as a list goes on past a
  // blank line and its items' lines are indented. A line that might still become an item, "1" or
  // "-", counts as one. Also whether the text ends in a fence that is still open.
  function scan(text, start) {
    let fence = null;
    let blank = false;
    let cut = start;
    for (let at = start; at < text.length;) {
      const end = text.indexOf('\n', at);
      const line = text.slice(at, end === -1 ? text.length : end);
      const marker = /^\s*(`{3,}|~{3,})(.*)$/.exec(line);
      if (fence) {
        if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length && marker[2].trim() === '') fence = null;
      } else if (/^\s*$/.test(line)) {
        blank = true;
      } else {
        if (blank && /^\S/.test(line) && !/^([-+*]|\d{1,9}[.)]?)(\s|$)/.test(line)) cut = at;
        blank = false;
        // A backtick fence's info has no backtick.
        if (marker && !(marker[1][0] === '`' && marker[2].includes('`'))) fence = marker[1];
      }
      if (end === -1) break;
      at = end + 1;
    }
    return {cut, open: fence !== null};
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
