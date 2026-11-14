// What the transcript's page does apart from drawing its messages (transcript.js): send the app a
// message, draw the symbols it sends, copy and say so, follow the latest while the reader is at
// the bottom, let text that streams out at an even pace, show a thought's latest lines, keep room
// for messages not drawn yet, and count seconds.
'use strict';

const page = (() => {
  const post = message => window.webkit?.messageHandlers.page.postMessage(message);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  // The SF Symbols the page draws, which it can't name: masks, colored as text is, but the warning
  // sign, in its own colors (page.css).
  function setup({glyphs}) {
    const style = document.documentElement.style;
    for (const [name, glyph] of Object.entries(glyphs)) {
      style.setProperty(`--${name}`, `url("${glyph.url}")`);
      style.setProperty(`--${name}-width`, `${glyph.width}px`);
      style.setProperty(`--${name}-height`, `${glyph.height}px`);
    }
  }

  // Copies, and says so: the button shows a check for 1.5 s, which another copy holds longer.
  function copy(button, text) {
    post({copy: text});
    button.classList.add('copied');
    clearTimeout(button.copied);
    button.copied = setTimeout(() => button.classList.remove('copied'), 1500);
  }

  // The page follows the end of `content` while the reader is at the bottom, until they scroll up,
  // and again once they scroll back down; nothing moves it while they scroll. `latest`, shown once
  // they have scrolled up, glides back. toLatest follows again, from the bottom at once, as a page
  // does when another conversation opens or a message is sent. Text that streams, which the page
  // says with `ease` as it draws it, is followed by easing toward the bottom over a few frames, so
  // that its lines slide in rather than jump; anything else that changes the content's size keeps
  // the bottom where it is at once, as content opening above it, or a new width. Away from the
  // bottom, a message the page `keep`s that changes its height above the screen leaves what the
  // reader reads in place.
  function follower(content, latest) {
    const scroller = document.scrollingElement;
    let following = true;
    // The glide to the latest under way, and the easing after text that streams, as animation
    // frames.
    let gliding = null;
    let easing = null;
    // Whether the content's next change of size is text that streamed (`ease`).
    let streamed = false;
    // When the reader last scrolled by wheel, trackpad or keys, and whether the pointer is down, as
    // while selecting.
    let scrolled = -Infinity;
    let pressed = false;

    // Where the page last put the scroller, at the bottom as its content or the window changed.
    let placed = null;
    // The content's width, as last measured: a change of it lays out every line again.
    let width = null;

    const bottom = () => scroller.scrollHeight - scroller.clientHeight;
    const atBottom = () => bottom() - scroller.scrollTop <= 40;
    const readerScrolls = () => pressed || performance.now() - scrolled < 200;
    const toBottom = () => {
      scroller.scrollTop = bottom();
      placed = scroller.scrollTop;
    };

    // After each change of the content's size, as the system clamps the scroller of content that
    // shrank, as a line folding does: the page then puts it where the clamp left it, which tells
    // the clamp's scroll from the reader's. Measured by the observer, after the layout the frame
    // does anyway.
    function follow(resized) {
      const easy = streamed && !resized && !reduced.matches;
      streamed = false;
      if (!following || gliding !== null || readerScrolls()) return;
      if (easy && bottom() > scroller.scrollTop) return ease();
      stopEasing();
      toBottom();
    }

    // Toward the bottom, wherever it is as it goes, by a share of the way each frame, as a spring
    // that settles in about a fifth of a second; at least a pixel, so that it arrives.
    function ease() {
      if (easing !== null) return;
      let last = performance.now();
      const step = now => {
        const gap = bottom() - scroller.scrollTop;
        if (!following || gap <= 1) {
          easing = null;
          if (following) toBottom();
          return;
        }
        scroller.scrollTop += Math.max(1, gap * (1 - Math.exp(-(now - last) / 60)));
        placed = scroller.scrollTop;
        last = now;
        easing = requestAnimationFrame(step);
      };
      easing = requestAnimationFrame(step);
    }

    function stopEasing() {
      if (easing === null) return;
      cancelAnimationFrame(easing);
      easing = null;
    }

    function scrolling() {
      scrolled = performance.now();
      stopGliding();
      stopEasing();
    }

    addEventListener('wheel', scrolling, {passive: true});
    addEventListener('keydown', scrolling, {passive: true});
    addEventListener('pointerdown', () => {
      pressed = true;
      stopGliding();
      stopEasing();
    }, {passive: true});
    // A right click's button comes up on its menu, not the page.
    for (const type of ['pointerup', 'pointercancel', 'contextmenu']) addEventListener(type, () => { pressed = false; }, {passive: true});
    // A scroll that leaves the scroller where the page put it, while it follows, is the page's own,
    // and keeps the following; any other, the reader's by wheel, keys or a drag of the scroller,
    // follows where it ends at the bottom and stops it elsewhere. The page's own events come a
    // frame late, by when content that came meanwhile, as an answer that starts while the
    // thinking over it folds, can have put the bottom more than 40 px away: taken for the
    // reader's, they would stop the following.
    addEventListener('scroll', () => {
      if (gliding === null && !(following && scroller.scrollTop === placed)) following = atBottom();
      showLatest();
    }, {passive: true});
    // Content that grows, a line opening, the window resized.
    new ResizeObserver(([entry]) => {
      const resized = width !== null && entry.contentRect.width !== width;
      width = entry.contentRect.width;
      follow(resized);
    }).observe(content);
    addEventListener('resize', () => follow(true));

    // A message laid out only near the screen (transcript.css) takes its own height as it comes on
    // it, in place of the room kept for it: one that starts above the screen would move what the
    // reader reads by as much, and the scroller moves with it, as browsers anchor scrolling, which
    // WebKit doesn't. Only then: a message that grows as it streams, or as the reader opens a line
    // in it, stays where it is. The observer measures it after the layout the frame does anyway;
    // the event that says it is laid out again comes before that or after, so that it counts as
    // coming on the screen while it is still skipped, or where the event found it laid out at
    // another height than measured last. From the top down, as each moves where those under it are.
    const heights = new WeakMap();
    const anchor = new ResizeObserver(entries => {
      const drawn = [];
      for (const {target, borderBoxSize: [size]} of entries) {
        const last = heights.get(target);
        heights.set(target, size.blockSize);
        if (last !== undefined && (target.skipped || target.drawn) && !following) {
          drawn.push({top: target.getBoundingClientRect().top, change: size.blockSize - last});
        }
        target.drawn = false;
      }
      let moved = 0;
      for (const {top, change} of drawn.sort((a, b) => a.top - b.top)) if (top < moved) moved += change;
      if (moved !== 0) scroller.scrollTop += moved;
    });

    function skips(event) {
      const element = event.currentTarget;
      element.skipped = event.skipped;
      if (!event.skipped && heights.has(element)) element.drawn = element.offsetHeight !== heights.get(element);
    }

    function showLatest() {
      latest.classList.toggle('shown', !following);
      latest.setAttribute('aria-hidden', following);
      latest.tabIndex = following ? -1 : 0;
    }

    // Back to the latest, gliding as Motion.glide does: a smooth spring over 0.4 s, to wherever the
    // bottom is as it goes.
    latest.addEventListener('click', () => {
      following = true;
      showLatest();
      stopEasing();
      if (reduced.matches) return toBottom();
      const start = scroller.scrollTop;
      const began = performance.now();
      const spring = t => 1 - (1 + 7.4 * t) * Math.exp(-7.4 * t);
      const step = now => {
        const t = Math.min((now - began) / 400, 1);
        scroller.scrollTop = start + (bottom() - start) * spring(t) / spring(1);
        gliding = t < 1 ? requestAnimationFrame(step) : null;
      };
      gliding = requestAnimationFrame(step);
    });

    function stopGliding() {
      if (gliding === null) return;
      cancelAnimationFrame(gliding);
      gliding = null;
    }

    function toLatest() {
      stopGliding();
      stopEasing();
      following = true;
      toBottom();
      // No scroll comes where the page doesn't move, as for a short conversation.
      showLatest();
    }

    // Keeps `element`, a message, in place where it is above the screen.
    function keep(element) {
      anchor.observe(element);
      element.addEventListener('contentvisibilityautostatechange', skips);
    }

    return {toLatest, ease: () => { streamed = true; }, keep};
  }

  // Text that streams, let out at an even pace: what came is shown over the animation frames that
  // follow, a share of what waits each frame, so that text that comes in bursts, as the app sends
  // a reply every 50 ms, shows at about the rate it comes, about a tenth of a second behind it.
  // `draw` gets what to show of each of the texts `to` was given; all of them at once once they
  // no longer stream, so that a reply's end is never held back, and with Reduce Motion. A text
  // shows whole at once the first time it is given, as when the page opens a reply that streams,
  // and where it does not go on from what shows.
  function paced(draw) {
    let target = {};
    let shown = {};
    let frame = null;
    let last = 0;

    function to(texts, live) {
      target = texts;
      for (const [key, text] of Object.entries(texts)) {
        shown[key] ??= text;
        if (!live || reduced.matches || !text.startsWith(shown[key])) shown[key] = text;
      }
      draw({...shown});
      const behind = Object.entries(target).some(([key, text]) => shown[key].length < text.length);
      if (behind && frame === null) {
        last = performance.now();
        frame = requestAnimationFrame(step);
      } else if (!behind) {
        stop();
      }
    }

    function step(now) {
      const share = 1 - Math.exp(-(now - last) / 100);
      last = now;
      let behind = false;
      for (const [key, text] of Object.entries(target)) {
        const at = shown[key].length;
        if (at >= text.length) continue;
        let next = Math.min(text.length, at + Math.max(1, Math.ceil((text.length - at) * share)));
        // Never half a character of two UTF-16 units.
        if (next < text.length && text.charCodeAt(next - 1) >= 0xD800 && text.charCodeAt(next - 1) <= 0xDBFF) next++;
        shown[key] = text.slice(0, next);
        behind ||= next < text.length;
      }
      frame = behind ? requestAnimationFrame(step) : null;
      draw({...shown});
    }

    function stop() {
      if (frame === null) return;
      cancelAnimationFrame(frame);
      frame = null;
    }

    return {to, stop};
  }

  // Puts `text` in `element`, adding only what is new where it goes on from what the element
  // holds, as text that streams does.
  function write(element, text) {
    const had = element.written ?? element.textContent;
    if (had === text) return;
    if (had !== '' && text.startsWith(had) && element.childNodes.length === 1) element.firstChild.appendData(text.slice(had.length));
    else element.textContent = text;
    element.written = text;
  }

  // What a line shows of the thinking while the model thinks, in a few rows at the line's end: of a
  // long thought, its last lines alone, from a line's start, so that each token lays out a few
  // lines and not all of it.
  function latest(text) {
    if (text.length <= 2000) return text;
    const start = text.indexOf('\n', text.length - 2000);
    return text.slice(start === -1 ? text.length - 2000 : start + 1);
  }

  // Marks the line of each thinking `measure` is given tall while its text is taller than the
  // 96 px it shows (page.css), which then fades at the top: measured by the observer, after the
  // layout the frame does anyway.
  const tall = new ResizeObserver(entries => {
    for (const entry of entries) entry.target.closest('.line')?.classList.toggle('tall', entry.contentRect.height > 96);
  });
  const measure = text => tall.observe(text);

  // How tall the messages of `content` might be, for those never drawn yet: the room a long
  // conversation's messages off the screen keep (transcript.css, content-visibility), so that the
  // scroller is about as long as the conversation before the reader has seen all of it. The
  // estimate it gives takes a message's element and `text`, which the page lays out in rows of as
  // many characters, about 7 px each, as the column fits less `inset` px, `row` px each, and in
  // blocks that empty lines part, `gap` px apart, as Markdown's paragraphs, or, where `gap` is
  // null, with its empty lines as rows too, as text that keeps them; and `around`, the room the
  // rest of it takes. The column is `content`'s width, which the window's sets: a new one
  // estimates them all again, in the frame that lays them out at it, as the sidebar slides.
  function estimator(content) {
    let width = null;
    const column = () => parseFloat(getComputedStyle(content).width);

    function size(element) {
      const {lines, inset, row, gap, around} = element.estimate;
      const columns = Math.max(1, Math.floor((width - inset) / 7));
      let height = around + gap * Math.max(0, lines.length - 1);
      for (const block of lines) for (const length of block) height += row * Math.max(1, Math.ceil(length / columns));
      element.style.setProperty('--estimate', `${Math.min(height, 20000)}px`);
    }

    addEventListener('resize', () => {
      if (width === null || column() === width) return;
      width = column();
      for (const element of content.children) if (element.estimate) size(element);
    });

    return (element, text, {inset, row, gap, around}) => {
      width ??= column();
      const blocks = gap === null ? [text] : text.split(/\n\s*\n/).filter(block => block.trim() !== '');
      element.estimate = {lines: blocks.map(block => block.split('\n').map(line => line.length)), inset, row, gap, around};
      size(element);
    };
  }

  // Shows the whole seconds since `since`, from the first, changing once a second on the second
  // and without rolling, as the app's counters change while the model works, for as long as
  // `element` is in the page; a count of null stops the one under way.
  function count(element, since, show) {
    clearTimeout(element.tick);
    if (since === null) return;
    const elapsed = Date.now() - since;
    const seconds = Math.floor(elapsed / 1000);
    show(seconds < 1 ? null : `${seconds}s`);
    element.tick = setTimeout(() => element.isConnected && count(element, since, show), 1000 - elapsed % 1000);
  }

  return {post, setup, copy, follower, paced, write, latest, measure, estimator, count};
})();
