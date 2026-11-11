// The Chat's transcript: the open conversation as the app sends it (Transcript.swift). The app
// calls page.setup once the page has loaded, show with the whole conversation when it changes, and
// update with its last message alone, a reply as it streams or once it has ended; the page sends
// back what to copy, and Ask Again. Markdown comes from markdown.js, and the rest of what it does
// apart from the conversation from page.js.
'use strict';

const transcript = (() => {
  const messages = document.getElementById('messages');
  const follow = page.follower(messages, document.getElementById('latest'));
  // The characters of the replies a conversation that opens draws at once, its latest: about a
  // window's worth. The Markdown of those before comes after it has shown (markdown.later).
  const firstScreen = 6000;
  // How the page lays out a message's text, for the room one not drawn yet keeps (page.estimator):
  // the user's words in their bubble, 22 px a row, which leaves the column's first 80 px clear and
  // pads them 16 px at either side, under their attachments (page.css, transcript.css); a reply's
  // Markdown across the column, 24 px a paragraph's row, its blocks 14 px apart less the 5 px their
  // rows leave over and under them (markdown.css), over its actions and under its line and why it
  // stopped short (transcript.css).
  const estimate = page.estimator(messages);
  const userText = attached => ({inset: 112, row: 22, gap: null, around: 17 + (attached ? 128 : 0)});
  const replyText = (line, error) => ({inset: 0, row: 24, gap: 9, around: 26 + (line ? 31 : 0) + (error ? 25 : 0)});

  // The messages drawn, by id: their element, the message as last given, and of a reply, its
  // texts as they show, which let out what streams at an even pace (page.paced).
  const views = new Map();
  // The reader's choice to open or close a reply's thinking, by id.
  const expanded = new Map();
  let conversation = null;
  let askAgain = {enabled: false, help: ''};

  const replyMarkup =
    '<div class="line"><button type="button" class="hover toggle"><span class="glyph chevron"></span>' +
    '<span class="verb"></span><span class="facts"></span></button>' +
    '<div class="thought"><div class="fold"><div class="reasoning"><div class="text"><div></div></div></div></div></div></div>' +
    '<div class="markdown"></div>' +
    '<p class="error"><span class="glyph warning"></span><span class="message"></span></p>' +
    '<div class="actions"><button type="button" class="hover copy-reply" title="Copy" aria-label="Copy Reply">' + markdown.copyGlyphs +
    '</button><button type="button" class="hover again" aria-label="Ask Again"><span class="glyph again"></span></button>' +
    '<span class="details"></span></div>';

  // Draws `message`; a reply's Markdown `later` where it is above a conversation's first screen.
  function draw(message, later = false) {
    let view = views.get(message.id);
    if (!view) {
      const element = document.createElement('article');
      element.className = message.role === 'user' ? 'user' : 'reply';
      element.dataset.id = message.id;
      element.innerHTML = message.role === 'user' ? '<div class="attachments"></div><p class="bubble"></p>' : replyMarkup;
      follow.keep(element);
      view = {element, message: null, shown: null};
      if (message.role !== 'user') {
        page.measure(element.querySelector('.text > div'));
        view.pace = page.paced(shown => drawTexts(view, shown));
      }
      views.set(message.id, view);
    }
    if (!same(view.message, message)) {
      const previous = view.message;
      view.message = message;
      if (message.role === 'user') drawUser(view.element, message, previous);
      else drawReply(view, later);
    }
    return view.element;
  }

  function same(a, b) {
    return a !== null && a.text === b.text && a.reasoning === b.reasoning && a.error === b.error && a.live === b.live &&
      a.details === b.details && a.line?.verb === b.line?.verb && a.line?.facts === b.line?.facts &&
      a.line?.since === b.line?.since && sameAttachments(a, b);
  }

  const sameAttachments = (a, b) => a.images.length === b.images.length && a.images.every((image, index) => image === b.images[index]) &&
    a.documents.length === b.documents.length &&
    a.documents.every((pdf, index) => pdf.name === b.documents[index].name && pdf.details === b.documents[index].details);

  // A PDF the user attached: its glyph, name, and its pages and size.
  function documentChip(pdf) {
    const chip = document.createElement('div');
    chip.className = 'pdf';
    chip.innerHTML = '<span class="glyph document"></span><span class="pdf-about"><span class="pdf-name"></span><span class="pdf-details"></span></span>';
    chip.querySelector('.pdf-name').textContent = pdf.name;
    chip.querySelector('.pdf-details').textContent = pdf.details;
    return chip;
  }

  // The user's images and PDFs over their text, at the column's end.
  function drawUser(element, message, previous) {
    const attachments = element.querySelector('.attachments');
    if (!previous || !sameAttachments(previous, message)) {
      attachments.replaceChildren(...message.images.map(url => Object.assign(document.createElement('img'), {src: url, alt: ''})),
        ...message.documents.map(documentChip));
    }
    const attached = message.images.length > 0 || message.documents.length > 0;
    attachments.hidden = !attached;
    const bubble = element.querySelector('.bubble');
    bubble.textContent = message.text;
    bubble.hidden = message.text === '';
    estimate(element, message.text, userText(attached));
  }

  // A reply: its line, its Markdown, why it stopped short, and its actions, which keep their place
  // and show under the pointer and on the last reply, but never while it streams. Its texts show
  // as they stream, at an even pace, and whole once it has ended.
  function drawReply(view, later) {
    const {element, message} = view;
    drawLine(view);
    const error = element.querySelector('.error');
    error.querySelector('.message').textContent = message.error ?? '';
    error.hidden = message.error === null;
    element.classList.toggle('live', message.live);
    element.querySelector('.details').textContent = message.details;
    element.querySelector('.actions').setAttribute('aria-hidden', message.live);
    if (!message.live) estimate(element, message.text, replyText(message.line !== null, message.error !== null));
    view.later = later;
    view.pace.to({text: message.text, reasoning: message.reasoning}, message.live);
  }

  // What shows of a reply's texts, as page.paced lets them out: its Markdown, which the page
  // follows easing while it streams, and its thinking.
  function drawTexts(view, shown) {
    const {element, message} = view;
    const body = element.querySelector('.markdown');
    if (view.later) {
      view.later = false;
      markdown.later(body, shown.text);
    } else {
      if (message.live && element.isConnected && body.markdown?.text !== shown.text) follow.ease();
      markdown.draw(body, shown.text, message.live);
    }
    body.hidden = shown.text === '';
    view.shown = shown;
    drawThought(view);
  }

  // The line over a reply: its verb while it waits, which shows half a second into the wait;
  // what the thinking took after. It opens to the thinking, of which it shows the latest lines
  // while the model thinks, unless the reader opened it: said while it is open alone, so that a
  // thought that folds, as when the answer starts, folds from what it showed.
  function drawLine(view) {
    const {element, message} = view;
    const line = element.querySelector('.line');
    line.hidden = message.line === null;
    if (message.line === null) return page.count(line, null);
    const reasoned = message.reasoning !== '';
    const thinking = message.live && message.text === '' && reasoned;
    const open = reasoned && (expanded.get(message.id) ?? thinking);
    line.classList.toggle('waiting', message.live && !reasoned);
    line.classList.toggle('reasoned', reasoned);
    line.classList.toggle('open', open);
    if (open) line.classList.toggle('latest', thinking && !expanded.has(message.id));
    const toggle = line.querySelector('.toggle');
    if (reasoned) toggle.setAttribute('aria-expanded', open);
    else toggle.removeAttribute('aria-expanded');
    line.querySelector('.verb').textContent = message.line.verb;
    if (message.line.since === null) facts(line, message.line.facts);
    page.count(line, message.line.since, words => facts(line, words));
    drawThought(view);
  }

  // The thinking a line opens to, drawn while it shows: its latest lines or all of it, as the line
  // says. A thought folded keeps what it showed as it folds.
  function drawThought(view) {
    const line = view.element.querySelector('.line');
    if (view.shown === null || !line.classList.contains('open')) return;
    const reasoning = view.shown.reasoning;
    page.write(line.querySelector('.text > div'), line.classList.contains('latest') ? page.latest(reasoning) : reasoning);
  }

  function facts(line, words) {
    const span = line.querySelector('.facts');
    span.textContent = words ? `· ${words}` : '';
    span.hidden = !words;
  }

  // The last message has Ask Again, as the app allows it, and its actions showing.
  function markLast() {
    for (const element of messages.querySelectorAll('.last')) element.classList.remove('last');
    const last = messages.lastElementChild;
    last?.classList.add('last');
    const again = last?.querySelector('.again');
    if (again) {
      again.disabled = !askAgain.enabled;
      again.title = askAgain.help;
    }
  }

  // The whole conversation: another opens at its latest message, its first screen drawn at once,
  // and a message sent scrolls there.
  function show(shown) {
    const opened = shown.id !== conversation;
    if (opened) {
      for (const view of views.values()) view.pace?.stop();
      views.clear();
      expanded.clear();
    }
    const added = !opened && shown.messages.at(-1)?.id !== messages.lastElementChild?.dataset.id;
    conversation = shown.id;
    askAgain = shown.askAgain;
    // The first message of the first screen, from the latest back.
    let first = shown.messages.length;
    for (let left = opened ? firstScreen : Infinity; first > 0 && left > 0;) left -= shown.messages[--first].text.length;
    const elements = shown.messages.map((message, index) => draw(message, index < first));
    const ids = new Set(shown.messages.map(message => message.id));
    for (const [id, view] of views) {
      if (ids.has(id)) continue;
      view.pace?.stop();
      views.delete(id);
    }
    if (elements.length !== messages.children.length || elements.some((element, index) => messages.children[index] !== element)) {
      messages.replaceChildren(...elements);
    }
    markLast();
    if (opened || added) follow.toLatest();
  }

  // The last message alone: a reply as it streams or once it has ended.
  function update(message) {
    if (views.has(message.id)) draw(message);
  }

  messages.addEventListener('click', event => {
    const button = event.target.closest('button');
    const view = button && views.get(button.closest('article').dataset.id);
    if (!view) return;
    if (button.classList.contains('copy-reply')) page.copy(button, view.message.text);
    else if (button.classList.contains('copy-code')) page.copy(button, button.closest('figure').querySelector('code').textContent);
    else if (button.classList.contains('again')) page.post({askAgain: true});
    else if (button.classList.contains('toggle')) {
      expanded.set(view.message.id, !button.closest('.line').classList.contains('open'));
      drawLine(view);
    }
  });

  return {show, update};
})();
