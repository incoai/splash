"""Install, without the Hub, the models the Hugging Face cache holds whole
that Splash serves: what other tools, or another Splash such as a source
checkout, downloaded there. The desktop app runs it so that it offers those
models beside the ones it downloaded:

    python install/cached.py [--models DIR]

A repository it cannot serve installs nothing: a model of no family Splash
serves, one missing files, or no model at all, such as a draft.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
    selects it (upstream.select_gguf), or else its own if that does: its
    split parts' folder, or its root target's name (upstream.root_targets)
    past the leading words of the repository's model name (without -GGUF),
    at least the first, which selects the same MODEL-VARIANT from the Hub's
    full listing. Without any, the repository itself: an MLX model, or files
    the installer refuses."""
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
    for stem, parts in upstream.root_targets(files):
        words = stem.split("-")
        shared = os.path.commonprefix([[w.lower() for w in words], model])
        variant = "-".join(words[max(len(shared), 1) :])
        targets.append((f"{name}:{variant}" if variant else name, parts))
    ids = []
    for own, target in targets:
        for model_id in (*suggested_models.get(name, ()), own):
            if selected(model_id) == target:
                ids.append(model_id)
                break
    return ids or [name]


def install_cached(models_root):
    """Install each model the Hub cache holds whole that is not installed in
    models_root, without the Hub."""
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
                pass  # the installer refuses it


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", type=Path, default=paths.MODELS)
    args = parser.parse_args(argv)
    # Only what the cache holds: the Hub is never asked.
    os.environ["HF_HUB_OFFLINE"] = "1"
    install_cached(args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
