# comfyui-splash

A demo: a small node pack that shows Splash's video API working from
ComfyUI. It is not a finished integration.

The nodes generate MiniMax-H3 videos, with sound, through a local Splash
server. ComfyUI only sends the request and receives the finished mp4;
Splash does the work.

The pack lives in the Splash repository at `integrations/comfyui/`.

## 1. What you need

- A Mac with a Splash checkout that contains the video API. No Splash
  release ships it yet: it comes with the series of pull requests this pack
  is part of. Check out a Splash tree that contains that series and build
  it as Splash's own README describes. The Homebrew and app installs do not
  have the video API, nor `integrations/comfyui/`.
- The H3 model, installed once in a terminal with that checkout's `splash`:
  - `splash install MiniMaxAI/MiniMax-H3:pika-6` for Text to Video and
    First-Last-Frame to Video
  - `splash install MiniMaxAI/MiniMax-H3:pika-ref2va-6` for Reference to
    Video (optional)
- The nodes find `splash` in this order: `SPLASH_BIN` when it is set (then
  only there), the `splash` on your `PATH`, then `/opt/homebrew/bin/splash`
  and `/usr/local/bin/splash`. If the checkout's `splash` is not the one on
  your `PATH`, set `SPLASH_BIN` to it in the environment ComfyUI starts in.
- A recent ComfyUI. No extra Python packages are needed: the pack uses
  `aiohttp`, `numpy`, `pillow` and `av`, which ComfyUI already requires.

## 2. Install the pack

1. Link or copy the folder into ComfyUI's `custom_nodes`, as
   `comfyui-splash`. A link follows the checkout when you `git pull`:

   ```bash
   ln -s /path/to/splash/integrations/comfyui /path/to/ComfyUI/custom_nodes/comfyui-splash
   ```

   or a copy, which you redo after an update:

   ```bash
   cp -R /path/to/splash/integrations/comfyui /path/to/ComfyUI/custom_nodes/comfyui-splash
   ```

2. Restart ComfyUI.

The nodes appear under **video → Splash**.

## 3. Make your first clip

1. Open a **new blank workflow** (the `+` next to the workflow tab). The
   workflow ComfyUI opens with is its own image example, not a Splash
   workflow. Do not download its models.
2. Add **Splash H3 Text to Video** and **Save Video**, and connect `VIDEO`
   to `video`. Or load `example_workflows/splash_h3_t2v.json` from the pack.
3. Type a prompt, set `size` to `832x480` and `seconds` to `5`, and press
   **Run**.

You do not start the server yourself. When nothing answers at the node's
`base_url` on this Mac, the node runs
`splash serve --model MiniMaxAI/MiniMax-H3:<variant> --offline --port 8000`
in the background and waits until it is ready. Its output goes to
`splash-serve.log` in ComfyUI's user directory. The server keeps running
afterwards, also after ComfyUI is closed, so later runs reuse it. Stop it
yourself when you are done. Turn `auto_start` off to start servers only by
hand.

## 4. The three video nodes

- **Splash H3 Text to Video**: a prompt in, a clip with sound out.
- **Splash H3 First-Last-Frame to Video**: a prompt, a `first_frame`
  (required) and a `last_frame` (optional). Landscape or portrait follows
  the first frame.
- **Splash H3 Reference to Video**: a prompt and up to 9 `pictures`, 3
  `videos` and 3 `audio_clips` as references, at least one picture or
  video. Refer to them in the prompt as `<Picture 1>`, `<Video 1>`,
  `<Audio 1>`, numbered per kind in the order of their slots. A reference
  video's soundtrack also counts as an `<Audio …>`, before the audio clips.

Common inputs:

- `size` (Text and Reference nodes): `1344x768`, `768x1344`, `832x480` or
  `480x832`. The default is `1344x768`.
- `resolution` (First-Last-Frame node): `768p` or `480p`; the first frame picks
  landscape or portrait.
- `seconds`: 4 to 15, default 5.
- `seed`: default 42.
- The prompt is sent as typed, in any language.

Advanced inputs: `variant` (Text and First-Last-Frame nodes: `auto`,
`vdn-8`, `vdn-50`, `fasth3-8`, `fasth3-4`, `pika-6`; leave it on `auto`),
`generate_audio` (on by default), `preview`, `base_url`, `api_key`,
`auto_start`, `unload_comfy_models` (frees ComfyUI's own models before the
request; on by default).

One server serves all three nodes, at `http://127.0.0.1:8000`. It serves
every variant of MiniMax-H3 that was installed when it started, whichever
node started it. After you install another variant, stop the server; the
next run starts it again.

## 5. Using a Mac in another room

ComfyUI listens on its own machine only (`127.0.0.1`, port 8188 by
default). From your laptop, forward the port and open
`http://127.0.0.1:8188` in the browser:

```bash
ssh -N -L 8188:127.0.0.1:8188 <the-mac>
```

The run continues on the Mac if the tunnel drops; reconnect and find the
result in the queue history.

A node can also use a Splash server that is already running elsewhere: set
its `base_url` to that address. The node starts servers only for
`127.0.0.1` or `localhost`.

## 6. If something goes wrong

The node shows the problem as `Splash <code>: <message>`.

- **"Value not in list … safetensors" or "Missing models"**: you ran
  ComfyUI's default image workflow, not a Splash one. See step 3.
- **"… is not a Splash server"**: another program is using the port. Stop
  it, or change the node's `base_url`.
- **"… not an H3 video model"**: a Splash server for another model, such
  as a language model, is on that port (8000 is its default too). The
  message shows the command that starts a video server.
- **"the … variant of MiniMaxAI/MiniMax-H3 is not installed: splash install
  …, then start the server again"**: install that variant, then stop the
  server so that the next run starts it again. Or set `variant` back to
  `auto`.
- **"the pika-ref2va-6 variant takes references …"**: only the Reference
  variant is installed. Install `MiniMaxAI/MiniMax-H3:pika-6` for the Text
  and First-Last-Frame nodes.
- **"Connect a picture or a video to the Reference node …"**: the
  Reference node needs at least one picture or video; audio clips go beside
  one.
- **"… the `splash` command was not found"**: build the Splash checkout (step
  1), or set `SPLASH_BIN` to its `splash` command.
- **"`splash serve` exited …"** or **"… was not ready after …"**: the
  server did not start. When the node started it, the message names
  `splash-serve.log` (and quotes its last lines if `splash serve` exited).
- **"… refused GET /ready …"**: the server has an API key. Set the node's
  `api_key` to the server's `--api-key`.
- **Every request is logged.** ComfyUI's log has one `splash request` line
  per run: node, size, seconds, seed, the job id or the error, never the
  prompt text or the API key. Include it when reporting a problem.

## Running the tests

The tests check the pack against its own fake Splash server
(`tests/fake_splash.py`), with ComfyUI's modules stubbed. Run
`python -m pytest` from this folder, in an environment with `pytest`,
`pytest-asyncio`, `aiohttp`, `numpy`, `pillow` and `av`. Splash's CI does
not run them.

The three video nodes have been run on a Mac through ComfyUI at 480p
against one Splash server. Not yet run on a device: 768p first/last-frame,
video or audio references, and the ComfyUI browser front end with the
current input names; the message for a language-model server on the node's
port is covered by the tests only.

## Licence

The pack is under the Splash repository's Apache-2.0 licence. The nodes
import ComfyUI's node API at run time; the pack includes no ComfyUI code.
