"""Install, without the Hub, the models the Hugging Face cache holds whole
that Splash serves: what other tools, or another Splash such as a source
checkout, downloaded there. The desktop app runs it so that it offers those
models beside the ones it downloaded:

    python install/cached.py [--models DIR]

prints, one JSON line each, {"model": ID} for each model it installed. A
repository it cannot serve installs nothing: a model of no family Splash
serves, one missing files, or no model at all, such as a draft.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script, from the installation's install/ folder: import its
    # siblings as the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

from . import models, paths

SUGGESTED = Path(__file__).with_name("suggested-models.tsv")


def suggested():
    """The suggested models' IDs (suggested-models.tsv), by repository."""
    by_repository = {}
    for line in SUGGESTED.read_text().splitlines():
        if line and not line.startswith("#"):
            model = line.split("\t", 1)[0]
            by_repository.setdefault(model.split(":")[0], []).append(model)
    return by_repository


def candidates(name, files, suggested_models):
    """The model IDs repository name's cached files may install as, one for
    each GGUF target they hold whole: the first of its suggested IDs that
    selects it from them (upstream.select_gguf), or else its own, if that
    does: its split parts' folder, or its root file's name (its root split
    parts' before their -0000K-of-0000N) past the words it starts with of
    the model the repository is named for (the name without -GGUF), and
    past its first at least. Repositories name their files
    MODEL-VARIANT.gguf, so that ID selects the same file from the Hub's
    listing of all of them. Without any, the repository itself: an MLX
    model, or files the installer refuses."""
    from . import upstream

    def selected(model_id):
        try:
            return upstream.select_gguf(files, models.split_model_id(model_id)[1])[0]
        except models.ModelError:
            return None

    model = name.split("/")[1].lower().removesuffix("-gguf").split("-")
    targets = []
    for folder in sorted(
        {
            n.split("/")[0]
            for n in files
            if n.count("/") == 1
            and n.endswith(".gguf")
            and not upstream.projector_named(n)
        }
    ):
        try:
            targets.append((f"{name}:{folder}", upstream.split_gguf(files, folder)))
        except models.ModelError:
            pass  # not a whole split GGUF
    for file in upstream.root_ggufs(files):
        if not upstream.projector_named(file):
            split = upstream.SPLIT_NAME.search(file)
            words = (file[: split.start()] if split else Path(file).stem).split("-")
            shared = os.path.commonprefix([[w.lower() for w in words], model])
            variant = "-".join(words[max(len(shared), 1) :])
            if not split:
                targets.append((f"{name}:{variant}" if variant else name, [file]))
                continue
            # A split GGUF's parts at the root, beside other variants'
            # (upstream.root_split_gguf): one target, named before their
            # -0000K-of-0000N.
            try:
                parts = upstream.root_split_gguf(files, variant)
            except models.ModelError:
                continue  # not a whole split GGUF
            if (f"{name}:{variant}", parts) not in targets:
                targets.append((f"{name}:{variant}", parts))
    ids = []
    for own, target in targets:
        for model_id in (*suggested_models.get(name, ()), own):
            if selected(model_id) == target:
                ids.append(model_id)
                break
    return ids or [name]


def install_cached(models_root):
    """Install each model the Hub cache holds whole that is not installed in
    models_root, without the Hub; yield the IDs installed."""
    from . import hub, upstream

    suggested_models = suggested()
    for name in hub.cached_repositories():
        try:
            repository = hub.Repository.resolve(name)
        except (models.ModelError, OSError):
            continue
        for model in candidates(name, repository.files, suggested_models):
            try:
                selection = models.Selection.of(models_root, model)
                if selection.link.exists():
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    upstream.prepare(selection)
            except (models.ModelError, OSError):
                continue
            yield model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", type=Path, default=paths.MODELS)
    args = parser.parse_args(argv)
    # Only what the cache holds: the Hub is never asked.
    os.environ["HF_HUB_OFFLINE"] = "1"
    for model in install_cached(args.models):
        print(json.dumps({"model": model}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
