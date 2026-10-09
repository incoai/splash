import shutil
import subprocess
import unittest
from pathlib import Path

from dev.tests.server_fixtures import FakeRuntime, HarnessTestCase

NODE = shutil.which("node")
CHAT = Path(__file__).resolve().parents[3] / "server" / "chat.html"

# Execute the complete production script with a minimal DOM and controlled I/O.
# Tests drive the registered UI handlers; no copied production handler or network.
HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(process.argv[1], 'utf8');
const scripts = [...html.matchAll(/<script(?: src="([^"]+)")?>([\s\S]*?)<\/script>/g)]
  .map(([, src, script]) => src
    ? fs.readFileSync(path.join(path.dirname(process.argv[1]), 'chat-assets', path.basename(src)), 'utf8')
    : script);

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.handlers = {};
    this.value = '';
    this.style = {};
    this.options = [];
    this.children = [];
    this.queries = {};
    this.scrollHeight = 40;
    const classes = new Set();
    this.classList = {
      add: name => classes.add(name),
      remove: name => classes.delete(name),
      contains: name => classes.has(name),
      toggle(name, force = !classes.has(name)) {
        if (force) classes.add(name); else classes.delete(name);
        return force;
      },
    };
  }
  addEventListener(name, callback) { this.handlers[name] = callback; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  remove() {} focus() {} setAttribute() {} removeAttribute() {}
  querySelector(selector) {
    const descendants = children => children.flatMap(child => [child, ...descendants(child.children)]);
    const child = descendants(this.children).find(child => selector.startsWith('.')
      ? (child.className || '').split(' ').includes(selector.slice(1))
      : child.tagName.toLowerCase() === selector);
    return child || (this.queries[selector] ||= new Element());
  }
}

function createChat(storage = new Map(), writable = true, models = null,
                    crypto = require('node:crypto').webcrypto) {
  const elements = {};
  const requests = [];
  const modelRequests = [];
  const reads = [];
  const copied = [];
  const timers = [];
  const dialog = {accept: true, messages: []};
  const clipboard = {writeText: async text => { copied.push(text); }};
  let scrolls = 0;
  let scrollBehavior;
  for (const id of ['chat', 'form', 'input', 'attachments', 'image-input',
                   'attach', 'effort', 'api-key', 'send', 'recents', 'new-chat',
                   'mobile-new', 'menu', 'scrim', 'scroll-bottom', 'sidebar',
                   'sidebar-close', 'sidebar-open']) elements[id] = new Element();
  elements.effort.options = ['', 'xhigh', 'medium', 'low', 'none'].map(value => ({value}));
  class FileReader {
    readAsDataURL(file) {
      this.result = `data:image/png;base64,${file.name}`;
      reads.push(this);
    }
  }
  const handlers = {};
  const viewport = {scrollHeight: 3000, clientHeight: 800};
  const context = vm.createContext({
    document: {
      querySelector: selector => elements[selector.slice(1)] || null,
      createElement: tag => new Element(tag),
      body: new Element(),
      documentElement: viewport,
      scrollingElement: viewport,
    },
    localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem(key, value) {
        if (!writable) throw new Error('Storage is full');
        storage.set(key, value);
      },
    },
    scrollY: 2200, innerHeight: 800,
    matchMedia: () => ({matches: false}),
    addEventListener(name, callback) { handlers[name] = callback; },
    scrollTo(options) {
      scrolls += 1;
      scrollBehavior = options.behavior;
      context.scrollY = Math.min(options.top, viewport.scrollHeight - 800);
    },
    AbortController, TextDecoder, Uint8Array, console, FileReader,
    navigator: {clipboard},
    confirm(message) { dialog.messages.push(message); return dialog.accept; },
    setTimeout: callback => { timers.push(callback); },
    atob: value => Buffer.from(value, 'base64').toString('binary'),
    crypto,
    fetch(url, options) {
      if (url === '/v1/models') {
        modelRequests.push(options.headers);
        // models answers from the request headers; without it the server is offline.
        return models ? Promise.resolve(models(options.headers))
          : Promise.reject(new Error('offline'));
      }
      return new Promise((resolve, reject) => {
        requests.push({url, headers: options.headers, body: JSON.parse(options.body), resolve, reject});
        options.signal.addEventListener('abort', () => {
          reject(Object.assign(new Error('Stopped'), {name: 'AbortError'}));
        });
      });
    },
  });
  context.window = context;
  scripts.forEach(script => vm.runInContext(script, context));
  return {elements, requests, reads, modelRequests, clipboard, copied, timers, dialog, scrolls: () => scrolls,
    scrollBehavior: () => scrollBehavior,
    scrollTo(y) { context.scrollY = y; handlers.scroll?.(); },
    scrollEvent() { handlers.scroll?.(); },
    grow() { viewport.scrollHeight += 200; },
  };
}

const flush = () => new Promise(resolve => setImmediate(resolve));
const submit = chat => chat.elements.form.handlers.submit({preventDefault() {}});
const enter = chat => chat.elements.input.handlers.keydown({
  key: 'Enter', keyCode: 13, shiftKey: false, isComposing: false,
  preventDefault() {},
});
function setText(chat, value) {
  chat.elements.input.value = value;
  chat.elements.input.handlers.input();
}
function pasteImages(chat, ...names) {
  const start = chat.reads.length;
  chat.elements.input.handlers.paste({
    clipboardData: {
      items: names.map(name => ({type: 'image/png', getAsFile: () => ({type: 'image/png', name})})),
      getData: () => '',
    },
    preventDefault() {},
  });
  return chat.reads.slice(start);
}
const pasteImage = (chat, name) => pasteImages(chat, name)[0];
function selectImages(chat, ...names) {
  const start = chat.reads.length;
  chat.elements['image-input'].files = names.map(name => ({type: 'image/png', name}));
  chat.elements['image-input'].handlers.change();
  return chat.reads.slice(start);
}
const imagePart = name => ({type: 'image_url', image_url: {url: `data:image/png;base64,${name}`}});
const draftImages = chat => chat.elements.attachments.children.flatMap(
  item => item.children.filter(child => child.tagName === 'IMG').map(image => image.src)
);
function removeImage(chat, index) {
  const item = chat.elements.attachments.children[index];
  item.children.find(child => child.tagName === 'BUTTON').handlers.click();
}

function respond(request, events) {
  let sent = false;
  request.resolve({ok: true, body: {getReader: () => ({
    async read() {
      if (sent) return {done: true};
      sent = true;
      return {done: false, value: Buffer.from(events)};
    },
  })}});
}
const succeed = request => respond(
  request, 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'
);
"""


@unittest.skipUnless(NODE, "Node.js is required to execute the chat UI tests")
class ChatTest(unittest.TestCase):
    def run_chat(self, program):
        result = subprocess.run(
            [NODE, "-e", HARNESS + program, str(CHAT)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_api_key_is_sent_only_as_a_header_and_not_persisted(self):
        self.run_chat(r"""
const storage = new Map();
const chat = createChat(storage);
chat.elements['api-key'].value = 'test-server-key';
setText(chat, 'hello');
submit(chat);
assert.equal(chat.requests[0].headers.Authorization, 'Bearer test-server-key');
assert.ok(!JSON.stringify(chat.requests[0].body).includes('test-server-key'));
assert.ok(!JSON.stringify([...storage]).includes('test-server-key'));
const fresh = createChat(storage);
assert.equal(fresh.elements['api-key'].value, '');
setText(fresh, 'hello');
submit(fresh);
assert.equal(fresh.requests[0].headers.Authorization, undefined);
""")

    def test_image_attachment_follows_the_served_input_modalities(self):
        self.run_chat(r"""
(async () => {
  const served = modalities => ({ok: true, json: async () => ({data: [{input_modalities: modalities}]})});
  for (const [modalities, accepted] of [
    [['text'], false], [['text', 'image', 'pdf'], true],
  ]) {
    const chat = createChat(new Map(), true, () => served(modalities));
    await flush();
    assert.equal(Boolean(chat.elements.attach.hidden), !accepted, `${modalities}`);
    assert.equal(pasteImages(chat, 'pasted').length, Number(accepted));
    assert.equal(selectImages(chat, 'selected').length, Number(accepted));
  }
  // An offline server leaves the modalities unknown and keeps the button.
  const offline = createChat();
  await flush();
  assert.ok(!offline.elements.attach.hidden, 'offline hid the button');
  // A server that needs a key rejects the first request, which also keeps the
  // button. Entering the key asks again with it; a text-only answer hides the
  // button and drops the image attached meanwhile.
  const chat = createChat(new Map(), true, headers => headers.Authorization ? served(['text'])
    : {ok: false, json: async () => ({error: {type: 'authentication_error'}})});
  await flush();
  assert.ok(!chat.elements.attach.hidden, 'unauthorized hid the button');
  selectImages(chat, 'early');
  assert.equal(chat.elements.attachments.children.length, 1, 'early image not attached');
  chat.elements['api-key'].value = 'key';
  chat.elements['api-key'].handlers.change();
  await flush();
  assert.equal(chat.modelRequests.at(-1).Authorization, 'Bearer key');
  assert.ok(chat.elements.attach.hidden, 'text-only model kept the button');
  assert.equal(chat.elements.attachments.children.length, 0, 'early image kept');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_restores_saved_chats_and_effort_with_read_only_storage(self):
        self.run_chat(r"""
const saved = JSON.stringify([{id: 'saved', title: 'Saved chat', updated: 1,
  messages: [{role: 'user', content: 'remembered message'}]}]);
for (const writable of [true, false]) {
  const storage = new Map([
    ['splash-chats', saved], ['splash-thinking-effort', 'low'],
  ]);
  const chat = createChat(storage, writable);
  assert.equal(chat.elements.effort.value, 'low');
  assert.equal(chat.elements.recents.children.length, 1);
  chat.elements.recents.children[0].querySelector('.recent-open').handlers.click();
  setText(chat, 'continue');
  submit(chat);
  assert.equal(chat.requests[0].body.messages[0].content, 'remembered message');
  assert.equal(storage.get('splash-thinking-effort'), 'low');
}
""")

    def test_default_effort_omits_reasoning_effort(self):
        # The server's --default-reasoning-effort, or the template's default,
        # applies until the user picks an effort.
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  assert.equal(chat.elements.effort.value, '');
  setText(chat, 'hello');
  submit(chat);
  assert.ok(!('reasoning_effort' in chat.requests[0].body));
  succeed(chat.requests[0]);
  await flush();
  chat.elements.effort.value = 'low';
  chat.elements.effort.handlers.change();
  setText(chat, 'again');
  submit(chat);
  assert.equal(chat.requests[1].body.reasoning_effort, 'low');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_saves_chats_without_crypto_random_uuid(self):
        # Browsers omit crypto.randomUUID outside secure contexts, such as a
        # LAN address over plain HTTP (#142).
        self.run_chat(r"""
(async () => {
  const {webcrypto} = require('node:crypto');
  const storage = new Map();
  const chat = createChat(storage, true, null,
    {getRandomValues: array => webcrypto.getRandomValues(array)});
  for (const prompt of ['first chat', 'second chat']) {
    setText(chat, prompt);
    submit(chat);
    succeed(chat.requests.at(-1));
    await flush();
    chat.elements['new-chat'].handlers.click();
  }
  const saved = JSON.parse(storage.get('splash-chats'));
  assert.deepEqual(saved.map(item => item.title).sort(), ['first chat', 'second chat']);
  assert.equal(new Set(saved.map(item => item.id)).size, 2);
  for (const {id} of saved) assert.match(id, /^[0-9a-f]{32}$/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_saves_chats_that_exceed_the_storage_quota(self):
        # Browser storage throws once the site holds more than a few MB, which
        # one photo's data URL can fill.
        self.run_chat(r"""
(async () => {
  class Storage extends Map {
    set(key, value) {
      let used = value.length;
      for (const [name, stored] of this) if (name !== key) used += stored.length;
      if (used > 10000) throw Object.assign(new Error('Quota exceeded'), {name: 'QuotaExceededError'});
      return super.set(key, value);
    }
  }
  const storage = new Storage();
  const chat = createChat(storage);
  async function send(text, ...images) {
    chat.elements['new-chat'].handlers.click();
    setText(chat, text);
    pasteImages(chat, ...images).forEach(reading => reading.onload());
    await flush();
    submit(chat);
    succeed(chat.requests.at(-1));
    await flush();
    // The next chat must be newer, since saved chats are ordered by time.
    const last = Date.now();
    while (Date.now() === last) await new Promise(resolve => setTimeout(resolve, 1));
  }
  const saved = () => JSON.parse(storage.get('splash-chats'));
  const notSaved = {type: 'text', text: '[Image not saved]'};
  const screenshot = 'S'.repeat(2000);
  await send('screenshot', screenshot);
  await send('huge photo', 'H'.repeat(12000));
  // A photo too large to store even alone does not cost older chats their images.
  assert.deepEqual(saved().map(conversation => conversation.messages[0].content), [
    [{type: 'text', text: 'huge photo'}, notSaved],
    [{type: 'text', text: 'screenshot'}, imagePart(screenshot)],
  ]);
  const [photoA, photoB] = ['A', 'B'].map(letter => letter.repeat(6000));
  await send('old photo', photoA);
  await send('new photo', photoB);
  // The two photos do not fit together, so the older chat loses its image.
  assert.deepEqual(saved().map(conversation => conversation.messages[0].content), [
    [{type: 'text', text: 'new photo'}, imagePart(photoB)],
    [{type: 'text', text: 'old photo'}, notSaved],
    [{type: 'text', text: 'huge photo'}, notSaved],
    [{type: 'text', text: 'screenshot'}, imagePart(screenshot)],
  ]);
  // Chats that do not fit even without images leave out the oldest chats.
  for (const name of ['one', 'two', 'three']) await send(`${name} ${'x'.repeat(3500)}`);
  assert.deepEqual(saved().map(conversation => conversation.title.split(' ')[0]), ['three', 'two']);
  // The open page keeps every chat with its images.
  assert.equal(chat.elements.recents.children.length, 7);
  chat.elements.recents.children[4].querySelector('.recent-open').handlers.click();
  setText(chat, 'again');
  submit(chat);
  assert.deepEqual(chat.requests.at(-1).body.messages[0].content,
    [{type: 'text', text: 'old photo'}, imagePart(photoA)]);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_enter_preserves_composition_and_shift_but_sends_normal_input(self):
        self.run_chat(r"""
for (const [name, overrides, shouldSend] of [
  ['composition', {isComposing: true}, false],
  ['Safari composition end', {isComposing: false, keyCode: 229}, false],
  ['Shift+Enter', {shiftKey: true}, false],
  ['other key', {key: 'a'}, false],
  ['Enter', {}, true],
]) {
  const {elements, requests} = createChat();
  elements.input.value = 'hello';
  let prevented = false;
  elements.input.handlers.keydown({
    key: 'Enter', keyCode: 13, shiftKey: false, isComposing: false,
    ...overrides,
    preventDefault() { prevented = true; },
  });
  assert.equal(requests.length, Number(shouldSend), name);
  assert.equal(prevented, shouldSend, name);
  assert.equal(elements.input.value, shouldSend ? '' : 'hello', name);
  if (shouldSend) {
    assert.equal(requests[0].url, '/v1/chat/completions');
    assert.deepEqual(requests[0].body.messages, [{role: 'user', content: 'hello'}]);
  }
}
""")

    def test_completion_preserves_the_next_draft_and_restores_failed_attachments(self):
        self.run_chat(r"""
(async () => {
  for (const outcome of ['success', 'stop', 'error']) {
    for (const draft of ['empty', 'ready image', 'loading image']) {
      const chat = createChat();
      const {elements, requests} = chat;
      pasteImage(chat, 'original').onload();
      await flush();
      elements.input.value = 'first prompt';
      submit(chat);
      const original = [{type: 'text', text: 'first prompt'}, imagePart('original')];
      assert.deepEqual(requests[0].body.messages, [{role: 'user', content: original}]);
      assert.deepEqual(draftImages(chat), []);
      let reading;
      if (draft !== 'empty') {
        elements.input.value = 'next prompt';
        reading = pasteImage(chat, 'next');
        if (draft === 'ready image') {
          reading.onload();
          await flush();
          assert.deepEqual(draftImages(chat), [imagePart('next').image_url.url]);
        }
      }
      assert.equal(elements.send.disabled, false, 'Stop stays available while an image loads');
      if (outcome === 'success') succeed(requests[0]);
      else if (outcome === 'stop') submit(chat);
      else requests[0].resolve({ok: false, json: async () => ({error: {message: 'Failed'}})});
      await flush();
      assert.equal(elements.attach.disabled, false, `${outcome}: request must finish`);
      if (draft === 'loading image') {
        assert.equal(elements.send.disabled, true, `${outcome}: unfinished next image blocks Send`);
        submit(chat);
        enter(chat);
        assert.equal(requests.length, 1, `${outcome}: cannot submit an unfinished next image`);
        reading.onload();
        await flush();
        assert.equal(requests.length, 1, 'finishing a read must not queue a request');
      }
      const restored = outcome === 'success' ? [] : [imagePart('original')];
      const expectedImages = [...restored, ...(draft === 'empty' ? [] : [imagePart('next')])];
      const expectedText = draft !== 'empty' ? 'next prompt' : outcome === 'success' ? '' : 'first prompt';
      assert.equal(elements.input.value, expectedText, `${outcome}, ${draft}`);
      assert.deepEqual(draftImages(chat), expectedImages.map(part => part.image_url.url), `${outcome}, ${draft}`);
      // The prior serialized request must stay unchanged as the next draft grows.
      assert.deepEqual(requests[0].body.messages, [{role: 'user', content: original}]);
      if (expectedText || expectedImages.length) {
        submit(chat);
        const expected = [...(expectedText ? [{type: 'text', text: expectedText}] : []), ...expectedImages];
        assert.deepEqual(requests[1].body.messages.at(-1), {role: 'user', content: expected});
        // Successful history remains; stopped/failed requests are removed.
        assert.equal(requests[1].body.messages.length, outcome === 'success' ? 3 : 1);
        succeed(requests[1]);
        await flush();
      }
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_chunks_without_text_neither_render_nor_scroll(self):
        self.run_chat(r"""
(async () => {
  const scrolls = [];
  for (const count of [0, 3]) {
    const chat = createChat();
    const {elements, requests} = chat;
    elements.input.value = 'prompt';
    submit(chat);
    const empty = 'data: {"choices":[{"delta":{}}]}\n\n'.repeat(count);
    respond(requests[0], 'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n' +
      empty + 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n' + empty + 'data: [DONE]\n\n');
    await flush();
    scrolls.push(chat.scrolls());
    elements.input.value = 'next';
    submit(chat);
    assert.equal(requests[1].body.messages[1].content, 'answer');
  }
  assert.equal(scrolls[1], scrolls[0], 'chunks without text must not scroll the page');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_markdown_renders_in_streams_and_saved_chats_without_changing_requests(
        self,
    ):
        self.run_chat(r"""
(async () => {
  const storage = new Map();
  const chat = createChat(storage);
  const source = '# Title\n\n**bold** and `inline`\n\n- one\n- two\n\n' +
    '| A | B |\n| --- | --- |\n| 1 | 2 |\n\n' +
    '```python\ndef greet():\n    return "你好"\n```';
  setText(chat, 'show code');
  submit(chat);
  const event = 'data: ' + JSON.stringify({choices: [{delta: {content: source}}]}) + '\n\n';
  respond(chat.requests[0], event + 'data: [DONE]\n\n');
  await flush();
  const rendered = chat.elements.chat.children.at(-1).querySelector('.content').innerHTML;
  assert.match(rendered, /<h1>Title<\/h1>/);
  assert.match(rendered, /<strong>bold<\/strong>/);
  assert.match(rendered, /<code>inline<\/code>/);
  assert.match(rendered, /<li>one<\/li>/);
  assert.match(rendered, /<table>/);
  assert.match(rendered, /hljs-keyword/);
  assert.match(rendered, /你好/);
  chat.elements.recents.children[0].querySelector('.recent-open').handlers.click();
  assert.equal(chat.elements.chat.children.at(-1).querySelector('.content').innerHTML, rendered);
  setText(chat, 'next');
  submit(chat);
  assert.equal(chat.requests[1].body.messages[1].content, source);
  succeed(chat.requests[1]);
  await flush();
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_markdown_handles_partial_fences_unknown_languages_and_unsafe_html(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  setText(chat, 'show code');
  submit(chat);
  let deliver;
  chat.requests[0].resolve({ok: true, body: {getReader: () => ({
    read: () => new Promise(resolve => { deliver = resolve; }),
  })}});
  await flush();
  async function chunk(delta) {
    deliver({done: false, value: Buffer.from('data: ' + JSON.stringify({choices: [{delta}]}) + '\n\n')});
    await flush();
  }
  await chunk({content: '```python\ndef greet():\n    return "你好"'});
  let content = chat.elements.chat.children.at(-1).querySelector('.content');
  assert.match(content.innerHTML, /hljs-keyword/);
  await chunk({content: '\n```\n\n```unknown-lang\n<b>literal</b>\n```\n\n' +
    '<img src=x onerror=alert(1)>\n\n[bad](javascript:alert(1))\n\n[good](https://example.com)\n\n' +
    '![x](https://example.com/a.png)'});
  assert.match(content.innerHTML, /&lt;b&gt;literal&lt;\/b&gt;/);
  assert.doesNotMatch(content.innerHTML, /href="javascript:/);
  assert.doesNotMatch(content.innerHTML, /<img/, 'a reply must not load images');
  assert.match(content.innerHTML, /href="https:\/\/example.com" target="_blank" rel="noopener noreferrer"/,
    'a link must open in a new tab');
  const reasoning = '**thinking**\n\n```python\ndef think():\n    return 1\n```';
  await chunk({reasoning_content: reasoning});
  assert.equal(chat.elements.chat.children.at(-1).querySelector('details')
    .querySelector('.reasoning').textContent, reasoning);
  deliver({done: false, value: Buffer.from('data: [DONE]\n\n')});
  await flush();
  deliver({done: true});
  await flush();
  chat.elements.recents.children[0].querySelector('.recent-open').handlers.click();
  assert.equal(chat.elements.chat.children.at(-1).querySelector('details')
    .querySelector('.reasoning').textContent, reasoning);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_stream_follows_at_bottom_and_pauses_while_reading_earlier_messages(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  setText(chat, 'prompt');
  submit(chat);
  let deliver;
  chat.requests[0].resolve({ok: true, body: {getReader: () => ({
    read: () => new Promise(resolve => { deliver = resolve; }),
  })}});
  await flush();
  async function chunk(delta) {
    chat.grow();
    deliver({done: false, value: Buffer.from('data: ' + JSON.stringify({choices: [{delta}]}) + '\n\n')});
    await flush();
  }
  let before = chat.scrolls();
  await chunk({content: 'first'});
  assert.equal(chat.scrolls(), before + 1, 'output follows the response after sending');
  assert.equal(chat.elements['scroll-bottom'].hidden, true);
  chat.scrollEvent();
  before = chat.scrolls();
  await chunk({content: ' second'});
  assert.equal(chat.scrolls(), before + 1, 'programmatic scroll events must not disable following');
  chat.scrollTo(400);
  before = chat.scrolls();
  await chunk({content: ' more'});
  const reasoning = chat.elements.chat.children.at(-1).querySelector('details').querySelector('.reasoning');
  Object.assign(reasoning, {scrollHeight: 900, clientHeight: 300, scrollTop: 600});
  await chunk({reasoning_content: 'reasoning'});
  assert.equal(reasoning.scrollTop, 900, 'Thinking at its bottom must follow new text');
  reasoning.scrollTop = 100;
  await chunk({reasoning_content: ' more'});
  assert.equal(reasoning.scrollTop, 100, 'Thinking must not pull a reader down from earlier text');
  assert.equal(chat.scrolls(), before, 'content and reasoning must not pull a reader down');
  assert.equal(chat.elements['scroll-bottom'].hidden, false);
  before = chat.scrolls();
  chat.elements['scroll-bottom'].handlers.click();
  assert.equal(chat.scrollBehavior(), 'instant', 'a jump during a reply must not animate');
  reasoning.scrollTop = 30;
  // The chunk arrives before the jump's scroll event.
  await chunk({content: ' latest'});
  assert.equal(reasoning.scrollTop, 30, 'reply text must not scroll the Thinking area');
  assert.equal(chat.scrolls(), before + 2, 'a jump to the latest reply resumes following');
  chat.scrollEvent();
  assert.equal(chat.elements['scroll-bottom'].hidden, true);
  chat.scrollTo(400);
  chat.scrollTo(100000);
  before = chat.scrolls();
  await chunk({content: ' tail'});
  assert.equal(chat.scrolls(), before + 1, 'manual scrolling back to the bottom resumes following');
  before = chat.scrolls();
  chat.elements.chat.handlers.load();
  assert.equal(chat.scrolls(), before + 1, 'image loading follows again after returning to the bottom');
  chat.scrollTo(400);
  before = chat.scrolls();
  deliver({done: false, value: Buffer.from('data: [DONE]\n\n')});
  await flush();
  deliver({done: true});
  await flush();
  assert.equal(chat.scrolls(), before, 'finishing must also preserve the reading position');
  setText(chat, 'next prompt');
  before = chat.scrolls();
  submit(chat);
  assert.equal(chat.scrolls(), before + 1, 'every new send must jump to the bottom');
  chat.requests[1].resolve({ok: true, body: {getReader: () => ({
    read: () => new Promise(resolve => { deliver = resolve; }),
  })}});
  await flush();
  before = chat.scrolls();
  await chunk({content: 'next answer'});
  assert.equal(chat.scrolls(), before + 1, 'the next response resumes following');
  chat.scrollEvent();
  before = chat.scrolls();
  chat.grow();
  chat.elements.chat.handlers.load();
  assert.equal(chat.scrolls(), before + 1, 'image loading must follow while enabled');
  chat.scrollTo(400);
  before = chat.scrolls();
  await chunk({content: ' later'});
  assert.equal(chat.scrolls(), before, 'leaving the bottom stops following again');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_reply_copy_uses_latest_raw_markdown_and_survives_history_reload(self):
        self.run_chat(r"""
(async () => {
  const storage = new Map();
  const chat = createChat(storage);
  setText(chat, 'show code');
  submit(chat);
  const box = chat.elements.chat.children.at(-1);
  const copy = box.querySelector('.copy-markdown');
  assert.equal(typeof copy.handlers.click, 'function', 'reply needs a Markdown copy button');
  assert.equal(copy.hidden, true, 'an empty reply cannot be copied');
  let deliver;
  chat.requests[0].resolve({ok: true, body: {getReader: () => ({
    read: () => new Promise(resolve => { deliver = resolve; }),
  })}});
  await flush();
  async function chunk(delta) {
    deliver({done: false, value: Buffer.from('data: ' + JSON.stringify({choices: [{delta}]}) + '\n\n')});
    await flush();
  }
  await chunk({reasoning_content: 'Private thinking', content: '\n\n# Title\n\n'});
  assert.equal(copy.hidden, false);
  await copy.handlers.click();
  assert.equal(chat.copied.at(-1), '\n\n# Title\n\n');
  await chunk({content: '```python\nreturn "<value>"\n```\n\n'});
  await copy.handlers.click();
  const source = '\n\n# Title\n\n```python\nreturn "<value>"\n```\n\n';
  assert.equal(chat.copied.at(-1), source, 'copy must preserve fences and surrounding whitespace');
  deliver({done: false, value: Buffer.from('data: [DONE]\n\n')});
  await flush();
  deliver({done: true});
  await flush();
  const fresh = createChat(storage);
  fresh.elements.recents.children[0].querySelector('.recent-open').handlers.click();
  await fresh.elements.chat.children.at(-1).querySelector('.copy-markdown').handlers.click();
  assert.equal(fresh.copied.at(-1), source, 'history must copy Markdown, excluding Thinking');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_copy_failure_is_reported_without_breaking_the_reply(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  setText(chat, 'prompt');
  submit(chat);
  succeed(chat.requests[0]);
  await flush();
  const box = chat.elements.chat.children.at(-1);
  const copy = box.querySelector('.copy-markdown');
  assert.equal(typeof copy.handlers.click, 'function', 'reply needs a Markdown copy button');
  chat.clipboard.writeText = async () => { throw new Error('Denied'); };
  await copy.handlers.click();
  assert.match(copy.title, /failed/i);
  assert.match(box.querySelector('.content').innerHTML, /answer/);
  chat.timers.forEach(callback => callback());
  assert.match(copy.title, /copy/i);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_deleting_chats_preserves_other_history_and_resets_the_active_chat(self):
        self.run_chat(r"""
const storage = new Map([['splash-chats', JSON.stringify([
  {id:'first', title:'First', updated:2, messages:[{role:'user', content:'first prompt'}]},
  {id:'second', title:'Second', updated:1, messages:[{role:'user', content:'second prompt'}]},
])]]);
const chat = createChat(storage);
const first = chat.elements.recents.children[0];
const remove = first.querySelector('.recent-delete');
assert.equal(typeof remove.handlers.click, 'function', 'recent chat needs a delete button');
first.querySelector('.recent-open').handlers.click();
const content = chat.elements.chat.children[0].querySelector('.user-text').textContent;
const saved = storage.get('splash-chats');
chat.dialog.accept = false;
remove.handlers.click();
assert.equal(storage.get('splash-chats'), saved, 'cancelling deletion must preserve saved history');
assert.equal(chat.elements.recents.children.length, 2);
assert.equal(chat.elements.chat.children[0].querySelector('.user-text').textContent, content,
  'cancelling deletion must preserve the active conversation');
assert.match(chat.dialog.messages.at(-1), /First/, 'confirmation must identify the selected chat');
chat.dialog.accept = true;
chat.elements.recents.children[1].querySelector('.recent-delete').handlers.click();
assert.match(chat.dialog.messages.at(-1), /Second/);
assert.equal(chat.elements.chat.children[0].querySelector('.user-text').textContent, content,
  'deleting another chat must preserve the active conversation');
assert.deepEqual(JSON.parse(storage.get('splash-chats')).map(chat => chat.id), ['first']);
chat.elements.recents.children[0].querySelector('.recent-delete').handlers.click();
assert.deepEqual(JSON.parse(storage.get('splash-chats')), []);
assert.equal(chat.elements.recents.children.length, 0);
assert.equal(chat.elements.chat.children[0].id, 'empty');
setText(chat, 'new prompt');
submit(chat);
assert.deepEqual(chat.requests[0].body.messages, [{role:'user', content:'new prompt'}]);
""")

    def test_loading_image_blocks_send_and_enter_until_ready(self):
        self.run_chat(r"""
(async () => {
  for (const text of ['describe this image', '']) {
    const chat = createChat();
    const {elements, requests} = chat;
    setText(chat, text);
    const reading = pasteImage(chat, 'before-submit');
    // Dispatching submit directly also exercises the guard behind the disabled button.
    submit(chat);
    enter(chat);
    assert.equal(requests.length, 0, 'neither submit nor Enter may send partial content');
    assert.equal(elements.attachments.children.length, 1, 'selection is visible immediately');
    assert.deepEqual(draftImages(chat), [], 'unfinished image is a placeholder');
    assert.equal(elements.send.disabled, true, 'Send is disabled while reading');
    assert.equal(elements.input.value, text);
    reading.onload();
    await flush();
    assert.equal(requests.length, 0, 'completion does not automatically submit');
    assert.equal(elements.send.disabled, false, 'image-only requests become sendable too');
    assert.deepEqual(draftImages(chat), [imagePart('before-submit').image_url.url]);
    if (text) submit(chat);
    else enter(chat);
    assert.deepEqual(requests[0].body.messages, [{role: 'user', content: [
      ...(text ? [{type: 'text', text}] : []), imagePart('before-submit'),
    ]}]);
    succeed(requests[0]);
    await flush();
    assert.deepEqual(draftImages(chat), []);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_separate_selections_keep_order_when_reads_finish_out_of_order(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  const [first, second] = selectImages(chat, 'first', 'second');
  const [third] = selectImages(chat, 'third');
  assert.equal(chat.elements.attachments.children.length, 3);
  third.onload();
  second.onload();
  await flush();
  assert.deepEqual(draftImages(chat), ['second', 'third'].map(name => imagePart(name).image_url.url));
  assert.equal(chat.elements.send.disabled, true, 'one pending image blocks the whole request');
  submit(chat);
  assert.equal(chat.requests.length, 0);
  first.onload();
  await flush();
  assert.deepEqual(draftImages(chat), ['first', 'second', 'third'].map(name => imagePart(name).image_url.url));
  submit(chat);
  assert.deepEqual(chat.requests[0].body.messages, [{role: 'user', content:
    ['first', 'second', 'third'].map(imagePart),
  }]);
  succeed(chat.requests[0]);
  await flush();
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_failed_image_remains_removable_and_preserves_other_selected_images(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  setText(chat, 'first prompt');
  submit(chat);
  const [failed, ready] = pasteImages(chat, 'failed', 'ready');
  failed.error = new Error('Image could not be read');
  failed.onerror();
  ready.onload();
  await flush();
  assert.equal(chat.elements.send.disabled, false, 'Stop remains available with a failed next image');
  submit(chat);
  await flush();
  assert.equal(chat.elements.attach.disabled, false, 'request was stopped');
  const failedItem = chat.elements.attachments.children[0];
  assert.equal(chat.elements.attachments.children.length, 2, 'failure remains visible');
  assert.ok(failedItem.children.some(child => child.tagName === 'SPAN' && child.textContent));
  assert.deepEqual(draftImages(chat), [imagePart('ready').image_url.url]);
  setText(chat, 'keep the readable image');
  assert.equal(chat.elements.send.disabled, true, 'failed image must be removed before sending');
  submit(chat);
  enter(chat);
  assert.equal(chat.requests.length, 1);
  removeImage(chat, 0);
  assert.equal(chat.elements.attachments.children.length, 1);
  assert.equal(chat.elements.send.disabled, false);
  submit(chat);
  assert.deepEqual(chat.requests[1].body.messages, [{role: 'user', content: [
    {type: 'text', text: 'keep the readable image'}, imagePart('ready'),
  ]}]);
  succeed(chat.requests[1]);
  await flush();
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_removing_pending_image_prevents_late_result_from_returning(self):
        self.run_chat(r"""
(async () => {
  for (const outcome of ['load', 'error']) {
    const chat = createChat();
    const reading = pasteImage(chat, 'removed');
    removeImage(chat, 0);
    assert.equal(chat.elements.attachments.children.length, 0);
    assert.equal(chat.elements.send.disabled, true, 'empty composer stays disabled');
    const next = pasteImage(chat, 'next');
    if (outcome === 'load') reading.onload();
    else {
      reading.error = new Error('Late read failure');
      reading.onerror();
    }
    await flush();
    assert.equal(chat.elements.attachments.children.length, 1, 'removed image cannot return');
    assert.deepEqual(draftImages(chat), []);
    assert.equal(chat.elements.send.disabled, true, 'next image is still loading');
    next.onload();
    await flush();
    submit(chat);
    assert.deepEqual(chat.requests[0].body.messages, [{role: 'user', content: [imagePart('next')]}]);
    succeed(chat.requests[0]);
    await flush();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_switching_chats_ignores_results_from_previous_draft(self):
        self.run_chat(r"""
(async () => {
  for (const navigation of ['new chat', 'recent chat']) {
    for (const outcome of ['load', 'error']) {
      const chat = createChat();
      setText(chat, 'saved conversation');
      submit(chat);
      succeed(chat.requests[0]);
      await flush();
      const reading = pasteImage(chat, 'old-chat');
      if (navigation === 'new chat') chat.elements['new-chat'].handlers.click();
      else chat.elements.recents.children[0].querySelector('.recent-open').handlers.click();
      assert.equal(chat.elements.attachments.children.length, 0);
      const next = pasteImage(chat, 'new-chat');
      if (outcome === 'load') reading.onload();
      else {
        reading.error = new Error('Old chat read failure');
        reading.onerror();
      }
      await flush();
      assert.equal(chat.elements.attachments.children.length, 1);
      assert.deepEqual(draftImages(chat), []);
      assert.equal(chat.elements.send.disabled, true);
      next.onload();
      await flush();
      enter(chat);
      assert.deepEqual(chat.requests[1].body.messages.at(-1), {role: 'user', content: [imagePart('new-chat')]});
      assert.equal(chat.requests[1].body.messages.length, navigation === 'new chat' ? 1 : 3);
      succeed(chat.requests[1]);
      await flush();
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


class ChatPageTest(HarnessTestCase):
    def test_root_serves_the_chat_page(self):
        harness = self.harness(FakeRuntime())
        status, content_type, payload = harness.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/html; charset=utf-8")
        self.assertIn(b"/v1/chat/completions", payload)
        for effort in (b"xhigh", b"medium", b"low", b"none"):
            self.assertIn(b'value="' + effort + b'"', payload)
        for label in (b"XHigh", b"Medium", b"Low", b"Off"):
            self.assertIn(b">" + label + b"</option>", payload)
        for label in (b"Splash", b"New chat", b"Recents"):
            self.assertIn(label, payload)
        self.assertIn(b"reasoning_effort", payload)
        self.assertIn(b"include_usage", payload)
        self.assertIn(b"tokens_per_second", payload)
        self.assertIn(b"allowedEfforts.has(savedEffort)", payload)
        self.assertIn(b"function storageGet", payload)
        self.assertIn(b"function storageSet", payload)
        self.assertIn(b"JSON.parse(storageGet(storeKey))", payload)
        self.assertIn(
            b"Chat still works if browser storage is full or disabled", payload
        )
        self.assertIn(b"function cleanMessage", payload)
        self.assertIn(b"if (input.value === '') input.value = text", payload)
        self.assertIn(b'id="image-input"', payload)
        self.assertIn(b'accept="image/*"', payload)
        self.assertIn(b"reader.readAsDataURL(file)", payload)
        self.assertIn(b"type: 'image_url'", payload)
        self.assertIn(b"input.addEventListener('paste'", payload)
        self.assertIn(b"className = 'waiting'", payload)
        self.assertIn(b"Generating response", payload)
        self.assertIn(b"@keyframes spin", payload)


if __name__ == "__main__":
    unittest.main()
