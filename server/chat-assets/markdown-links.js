// The links a reply's Markdown keeps, in the web chat and the app's transcript alike: web and mail
// links stay links, any other keeps only its text, and an image is a link to it, so that what a
// model writes never loads anything, nor opens another app, by itself. A markdown-it plugin:
// `markdownit(options).use(markdownLinks)`.
'use strict';

function markdownLinks(parser) {
  const web = url => /^(https?|mailto):/i.test(url ?? '');
  // Every link reaches the rule below, which keeps web and mail links and the text of others.
  parser.validateLink = () => true;
  parser.core.ruler.push('links', state => {
    for (const token of state.tokens) {
      if (token.type === 'inline') token.children = links(token.children, state);
    }
  });

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
}
