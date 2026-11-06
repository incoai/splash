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
    split parts' folder (upstream.folder_targets), or its root target's name
    (upstream.root_targets) past the leading words of the repository's model
    name (without -GGUF), at least the first, which selects the same
    MODEL-VARIANT from the Hub's full listing. Without any, the repository
    itself: an MLX model, or files the installer refuses."""
    from . import upstream

    def selects(model_id, target):
        """Whether model_id selects target's files from files, all of them
        (a split GGUF's parts are selected in split order)."""
        try:
            chosen = upstream.select_gguf(files, models.split_model_id(model_id)[1])
        except models.ModelError:
            return False
        return sorted(chosen[0]) == list(target.files)

    model = name.split("/")[1].lower().removesuffix("-gguf").split("-")
    targets = [(f"{name}:{t.name}", t) for t in upstream.folder_targets(files)]
    for target in upstream.root_targets(files):
        words = target.name.split("-")
        shared = os.path.commonprefix([[w.lower() for w in words], model])
        variant = "-".join(words[max(len(shared), 1) :])
        targets.append((f"{name}:{variant}" if variant else name, target))
    ids = []
    for own, target in targets:
        for model_id in (*suggested_models.get(name, ()), own):
            if selects(model_id, target):
                ids.append(model_id)
                break
    return ids or [name]


def present(link):
    """Whether the installation at selection link link is on this Mac, as
    the app lists it (Runtime.installedModels): its record names files that
    are all here, links followed."""
    from . import assembly

    record = assembly.recorded(link)
    return record is not None and all(
        Path(entry["path"]).exists() for entry in record["files"].values()
    )


def install_cached(models_root):
    """Install each model the Hub cache holds whole whose installation in
    models_root is not on this Mac (present), without the Hub: one whose
    files are gone installs again from the files the cache holds now."""
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
                if present(selection.link):
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
