"""MiniMax-H3, a video model: assemblies of target format h3-video.

Splash installs it from MiniMax's release (families.H3: release), one
variant per installation (H3.variants): `MiniMaxAI/MiniMax-H3:vdn-8` (OpenVDN's 8-step
stage) or `:vdn-50` (its 50-step one), or FastVideo's FastH3 students
`:fasth3-8` and `:fasth3-4`, or Pika H3 Flex's `:pika-6` (the model without
a variant, freshly installed; an installation keeps its variant) and
`:pika-ref2va-6`. Its weights are
converted into the engine's images (h3.py, recipe RECIPE; fasth3.py, recipe
fasth3.RECIPE, for FastH3 and for a variant's spec, H3.variants) as they are
read from the release and the variant's companion repository at their
default branches' commits, by range
requests of their safetensors files, none of which is downloaded
(conversion.py), and from the files the recipe pins (h3.PINNED: TAEH3's;
fasth3.pinned). The images are a derived entry, .converted/<key>
(assembly.converted), keyed by the content IDs of the files read (the
recipe's inputs), the recipe, the DiT and its spec, so a commit that changes none of
them converts nothing: converted once, resumed after an interruption, and
collected once no assembly links it. The text encoder, Qwen3-VL-32B's MLX
checkpoint (H3.encoder), comes from its own repository at its default branch's
commit, as a language model's draft does: only what the engine reads of it
(encoder_package), its other shards neither downloaded nor linked.

The paths an assembly links (assembly.py lists a language model's):
  weights/dit-<dit>/, weights/vae/, weights/audio/, weights/preview/
                         the engine's images (runtime/video/model), a
                         DiT's runtime adapter in dit-<dit>/adapter/
  tokenizer/, LICENSE, NOTICE
                         FL2VA's tokenizer, which the server encodes prompts
                         with, and the release's license
                         (and notice, where it has one)
  encoder/               the text encoder's package: its config.json, the
                         shards holding only tensors the engine reads, from
                         the Hub cache, and the shards holding some, trimmed
                         to those, with an index of the package, from the
                         entry .converted/<package key>
                         (runtime/video/model/TextEncoder.hpp)

Its record's sources are the release and the encoder; `video` adds the
variants, the entry's key, the commits of the companion repository and
TAEH3's, and the encoder's package (encoder: its entry's key and the shards it
links, trims and leaves out). prepare_local_model takes a directory of images converted before
(dit-<dit>/, vae/, audio/, preview/, tokenizer/ and LICENSE), and links its
files as they are.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from server import protocol as wire

from . import assembly, conversion, hub, models, upstream
from .families import H3, Variant

# The text encoder's package: the tensors the engine reads of the encoder's
# checkpoint (TextEncoder.mm, VisionTowerWeights.mm), decoder layers 0 to
# H3.encoder_layers - 1, the token table and the vision tower; the later
# layers, the final norm and lm_head are never read and not installed. Any
# width of mlx-community's checkpoints (4-, 6-, 8-bit) by its tensors'
# names. H3.encoder_package names the trimming in its entry's key, and
# ENCODER_INDEX is the package's own index: the repositories' index is a
# bf16 checkpoint's, of 14 shards none of them has, and the engine reads none
# (SafetensorsCheckpoint.mm reads the shards' headers).
ENCODER_INDEX = "model.safetensors.index.json"
_ENCODER_LAYER = "language_model.model.layers."
_ENCODER_WHOLE = ("language_model.model.embed_tokens.", "vision_tower.")
# The format of the engine's int8 weights (QUANT_VIDEO_W8 of
# runtime/metal/abi/QuantFormat.h, which the engine checks the DiT's
# manifest against), and the recipe's name.
RECIPE = "h3-w8a8-vdn-v1"
# The revision of the recipes' numerics, in a derived entry's key with the
# format (recipe_identity): bumped when what the units write changes under the
# same format, so that an installation converts and tabulates anew. 2: LoRA
# merges rounded to bf16 as the official models hold them (fasth3-4: the
# student's checkpoint; vdn-8 / vdn-50: OpenVDN's in-place bf16 merge,
# adaLN and final layer included), and adaLN tables of the time embedder in
# exact fp32 (every variant; splash video-adaln).
REVISION = 2


def is_source(files):
    """Whether a repository's files are MiniMax's release of MiniMax-H3."""
    return H3.release_index in files


def is_local(path: Path):
    """Whether a local directory holds MiniMax-H3: images converted before,
    or MiniMax's release, which is refused."""
    return (path / H3.release_index).is_file() or _is_converted(path)


def _is_converted(path: Path):
    images = all((path / name).is_dir() for name in (*H3.images, "tokenizer"))
    return images and bool(variants(path))


def variants(path: Path):
    """The variants whose DiT a directory of images holds."""
    return [n for n, v in H3.variants.items() if (path / f"dit-{v.dit}").is_dir()]


def installed(model_root: Path):
    """The variants the assembly at model_root serves, name -> Variant:
    H3's whose DiT its weights hold, and the one a spec installed, as its
    record names it (video.spec)."""
    weights = model_root / "weights"
    rows = {n: H3.variants[n] for n in variants(weights)}
    recorded = (assembly.recorded(model_root) or {}).get("video") or {}
    if (spec := recorded.get("spec")) and (weights / f"dit-{spec['name']}").is_dir():
        rows[spec["name"]] = _spec_row(spec["name"], spec)
    return rows


def _variant(selection, installed):
    """The variant selection names: a model ID without one keeps the
    variant the installation at its link holds, or else takes
    H3.default_variant; one of a spec (selection.spec, or the one the
    installation records) is _spec_variant's."""
    recorded = (installed or {}).get("video", {})
    held = recorded.get("variants")
    name = selection.variant or (held or [H3.default_variant])[0]
    if spec := selection.spec or (held == [name] and recorded.get("spec")):
        return name, _spec_variant(name, spec)
    if name not in H3.variants:
        raise models.ModelError(
            f"{selection.repo_id} has no variant {name}: its variants are "
            + ", ".join(H3.variants)
        )
    return name, H3.variants[name]


def _spec_variant(name, spec):
    """The variant of a spec file (`splash install MODEL:NAME --spec FILE`):
    fasth3.py's recipe on a checkpoint of MiniMax-H3's DiT that Splash does
    not ship, its adapter read from adapter.repo, else the checkpoint's
    repository. The installer checks the checkpoint's structure only."""
    from . import fasth3  # numpy, which the launcher's environment lacks

    if name in H3.variants:
        raise models.ModelError(
            f"{name} is a variant Splash ships: name the spec's another"
        )
    if name in {v.dit for v in H3.variants.values()}:
        raise models.ModelError(
            f"{name} is the DiT of a variant Splash ships: name the spec's another"
        )
    if not wire.valid_dit_name(name):
        raise models.ModelError(
            f"a spec's variant is named in 1 to 64 of a-z, 0-9, '.', '_' and '-', "
            f"without '..', not {name}"
        )
    if not isinstance(spec, dict):
        raise models.ModelError("the spec is not a JSON object")
    spec = {
        "name": name,
        "adapter": None,
        "references": False,
        "qkfix": list(fasth3.QKFIX),
    } | spec
    adapter = spec["adapter"] or {}
    # fasth3.check_spec refuses the spec's and its schedule's unknown fields.
    for field, value, known, required in (
        ("", spec, None, ("checkpoint", "schedule")),
        ("checkpoint", spec.get("checkpoint"), ("repo", "revision", "prefix"), ("repo", "prefix")),
        ("adapter", adapter, ("repo", "file", "name", "alpha"), ("file", "name", "alpha") if adapter else ()),
        ("schedule", spec.get("schedule"), None,
         ("rungs", "video_shift", "audio_shift", "sampler", "sigma_precision")),
    ):  # fmt: skip
        what = f"the spec's {field}" if field else "the spec"
        if not isinstance(value, dict):
            raise models.ModelError(f"{what} is not a JSON object")
        if known and (extra := sorted(set(value) - set(known))):
            raise models.ModelError(f"{what} has no field {extra[0]}")
        if lacking := [k for k in required if k not in value]:
            raise models.ModelError(f"{what} lacks its field {lacking[0]}")
    if spec["name"] != name:
        raise models.ModelError(
            f"the spec names the variant {spec['name']}, not {name}"
        )
    fasth3.check_spec(spec)
    return _spec_row(name, spec)


def _spec_row(name, spec):
    """The Variant of a checked spec: its DiT dit-<name>, its rungs' steps,
    its adapter's repository (adapter.repo, else the checkpoint's)."""
    companion = (spec["adapter"] or {}).get("repo", spec["checkpoint"]["repo"])
    return Variant(len(spec["schedule"]["rungs"]), name, companion, "fasth3", spec)


def recipe_identity(recipe):
    """The recipe module's identity in its derived entries' key
    (conversion.key): its format and REVISION."""
    return f"{recipe.RECIPE}+r{REVISION}"


def _recipe(variant):
    """The module of the variant's recipe, h3.py or fasth3.py."""
    from . import fasth3, h3  # numpy, which the launcher's environment lacks

    return fasth3 if variant.recipe == "fasth3" else h3


def _spec(variant):
    """The keyword argument of the recipe's functions naming the variant's
    spec, if it has one."""
    return {"spec": variant.spec} if variant.spec else {}


def _refuse_options(model, *options):
    """A video model takes none of a language model's source options."""
    if any(options):
        raise models.ModelError(
            f"{model} is a video model: it takes no DFlash2 draft, vision "
            "projector or language-only option"
        )


def encoder_reads(name):
    """Whether the engine reads the tensor name of the encoder's checkpoint."""
    if name.startswith(_ENCODER_WHOLE):
        return True
    layer = name.removeprefix(_ENCODER_LAYER).split(".", 1)[0]
    return (
        name.startswith(_ENCODER_LAYER)
        and layer.isdigit()
        and int(layer) < H3.encoder_layers
    )


@dataclass(frozen=True)
class EncoderPackage:
    """The text encoder's package of one commit: config.json and the shards
    linked from the Hub cache as they are (linked), the shards rewritten with
    only the tensors the engine reads into the derived entry key (trimmed:
    name -> those tensors), with ENCODER_INDEX, and the shards left out
    (omitted). sources, each shard's conversion.Source, is None for a
    package an assembly records, which is not planned again."""

    key: str
    linked: tuple
    trimmed: dict
    omitted: tuple
    sources: dict | None = None

    @property
    def downloads(self):
        """The files of the encoder's repository the package links."""
        return {"config.json", *self.linked}

    def record(self):
        """What an assembly records of the package (video.encoder)."""
        return {
            "key": self.key,
            "package": H3.encoder_package,
            "layers": H3.encoder_layers,
            "linked": list(self.linked),
            "trimmed": sorted(self.trimmed),
            "omitted": list(self.omitted),
        }


def encoder_package(repo, recorded=None):
    """The package of repo, the encoder at its commit: planned from its
    shards' headers, each read once (by range requests, or in the Hub
    cache), when the Hub lists the commit; when it does not, the package an
    assembly records for it (recorded: (repo, revision) -> its record), or
    else one planned from the shards of its Hub cache snapshot. A ModelError
    refuses a checkpoint whose shards lack a tensor group the engine
    reads."""
    record = (recorded or {}).get((repo.name, repo.revision))
    if not repo.sizes and _is_package_record(record):
        return EncoderPackage(
            record["key"],
            tuple(record["linked"]),
            dict.fromkeys(record["trimmed"], ()),
            tuple(record["omitted"]),
        )
    shards = sorted(
        n for n in repo.files if "/" not in n and n.endswith(".safetensors")
    )
    if "config.json" not in repo.files or not shards:
        raise models.ModelError(
            f"{repo.name} is missing files MiniMax-H3 needs: config.json, "
            "safetensors shards"
        )
    linked, omitted, trimmed, sources, read = [], [], {}, {}, set()
    for name in shards:
        sources[name] = source = conversion.Source(repo, name)
        kept = [t for t in source.tensors if encoder_reads(t)]
        read.update(_encoder_group(t) for t in kept)
        if len(kept) == len(source.tensors):
            linked.append(name)
        elif kept:
            trimmed[name] = sorted(kept)
        else:
            omitted.append(name)
    groups = {*_ENCODER_WHOLE}
    groups.update(f"{_ENCODER_LAYER}{i}." for i in range(H3.encoder_layers))
    if missing := sorted(groups - read):
        raise models.ModelError(
            f"{repo.name}@{repo.revision} lacks tensors the engine reads: "
            + ", ".join(f"{group}*" for group in missing)
        )
    content = {f"{repo.name}/{n}": _content_id(repo, n) for n in (*linked, *trimmed)}
    key = conversion.key(content, H3.encoder_package, f"layers-{H3.encoder_layers}")
    return EncoderPackage(key, tuple(linked), trimmed, tuple(omitted), sources)


def _encoder_group(name):
    """The tensor group the engine reads that tensor name lies in: its
    decoder layer's prefix, or one of _ENCODER_WHOLE."""
    if name.startswith(_ENCODER_LAYER):
        layer = name.removeprefix(_ENCODER_LAYER).split(".", 1)[0]
        return f"{_ENCODER_LAYER}{layer}."
    return next(prefix for prefix in _ENCODER_WHOLE if name.startswith(prefix))


def _content_id(repo, name):
    """A shard's content ID: the one the Hub lists, else its Hub cache
    blob's name (assembly.file_record's digest)."""
    if repo.sizes.get(name, (None, None))[1]:
        return repo.sizes[name][1]
    return assembly.digest(repo.file(name))


def _is_package_record(record):
    return (
        isinstance(record, dict)
        and record.get("package") == H3.encoder_package
        and record.get("layers") == H3.encoder_layers
        and models.is_hex_digest(record.get("key"), 64)
        and all(
            isinstance(record.get(group), list)
            and all(
                isinstance(n, str) and models.is_safe_path(n) for n in record[group]
            )
            for group in ("linked", "trimmed", "omitted")
        )
    )


def recorded_packages(models_root: Path, *records):
    """The encoder packages assemblies record, by the encoder's (repo,
    revision): those of records, then of the assemblies under models_root."""
    resolved = models_root / ".resolved"
    found = [*records]
    if resolved.is_dir():
        found += [assembly.recorded(entry) for entry in sorted(resolved.iterdir())]
    packages = {}
    for record in found:
        if not isinstance(record, dict) or not isinstance(record.get("video"), dict):
            continue
        source = record.get("sources", {}).get("encoder")
        if isinstance(source, dict) and "encoder" in record["video"]:
            identity = (source.get("repo"), source.get("revision"))
            packages.setdefault(identity, record["video"]["encoder"])
    return packages


def _encoder_entry(held, root, package, model):
    """The staging folder of the package's derived entry, its shards trimmed
    into it, held by held; None when the entry is there."""
    if assembly.is_converted(root, package.key):
        return None
    if package.sources is None:
        raise models.ModelError(
            f"the text encoder's package {package.key[:12]} is gone and "
            "its repository is not listed: install with the Hub reachable"
        )
    stage = held.enter_context(assembly.converting(root, package.key))
    # Another installation may have published it meanwhile.
    if not assembly.is_converted(root, package.key):
        conversion.convert(
            stage, _encoder_units(stage, package), f"{model}'s text encoder"
        )
    return stage


def _encoder_derived(root, package, stage):
    """The assembly's files of the package's derived entry."""
    entry = assembly.converted(root, package.key, stage)
    return {f"encoder/{name}": path for name, path in entry.items()}


def _encoder_units(stage, package):
    """The units writing the package's entry into stage: each trimmed shard
    (_trimmed), then the package's index of every shard it links."""

    def index(_):
        weights, size = {}, 0
        for name in sorted({*package.linked, *package.trimmed}):
            source = package.sources[name]
            kept = package.trimmed.get(name) or list(source.tensors)
            for tensor in kept:
                begin, end = source.tensors[tensor]["data_offsets"]
                weights[tensor], size = name, size + end - begin
        metadata = {
            "total_size": size,
            "package": H3.encoder_package,
            "layers": H3.encoder_layers,
        }
        index = {"metadata": metadata, "weight_map": weights}
        return {ENCODER_INDEX: models.json_bytes(index)}

    units = [
        conversion.Unit(
            name, (name,), lambda _, name=name: {name: _trimmed(stage, package, name)}
        )
        for name in sorted(package.trimmed)
    ]
    return units + [conversion.Unit(ENCODER_INDEX, (ENCODER_INDEX,), index)]


def _trimmed(stage, package, name):
    """The package's shard name with only the tensors the engine reads, in
    their order in the shard, their names, dtypes, shapes and bytes as they
    are, under its header's metadata: a staging file in stage, its tensors'
    bytes copied in place by ranges read on conversion.WORKERS threads."""
    source, kept = package.sources[name], package.trimmed[name]
    header = {} if source.metadata is None else {"__metadata__": source.metadata}
    spans, offset = [], 0
    for tensor in sorted(kept, key=lambda t: source.tensors[t]["data_offsets"][0]):
        entry = source.tensors[tensor]
        begin, end = entry["data_offsets"]
        header[tensor] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + end - begin],
        }
        if spans and spans[-1][0] + spans[-1][2] == begin:
            spans[-1][2] += end - begin
        else:
            spans.append([begin, offset, end - begin])
        offset += end - begin
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    base = 8 + len(encoded)
    pieces = [
        (start + at, base + target + at, min(conversion.CHUNK, size - at))
        for start, target, size in spans
        for at in range(0, size, conversion.CHUNK)
    ]
    path = stage / f".{name}.partial"
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(base + offset)

        def copy(piece):
            start, at, size = piece
            view = memoryview(bytearray(size))
            source.read_into(start, view)
            while view:
                written = os.pwrite(stream.fileno(), view, at)
                view, at = view[written:], at + written

        with ThreadPoolExecutor(conversion.WORKERS) as pool:
            for _ in pool.map(copy, pieces):
                pass
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _release_files(repo):
    """The release's files an assembly links, by their assembly paths:
    FL2VA's tokenizer and the license files."""
    names = {n: n.removeprefix(H3.tokenizer_prefix) for n in repo.files}
    files = {n: path for n, path in names.items() if path.startswith("tokenizer/")}
    if not files or "LICENSE" not in repo.files:
        raise models.ModelError(
            f"{repo.name} is missing files MiniMax-H3 needs: {H3.tokenizer_prefix}tokenizer/, LICENSE"
        )
    return files | {n: n for n in H3.license_files if n in repo.files}


def images(directory: Path, dits):
    """The engine's images under directory, an entry or images converted
    before, by their names: each DiT's (dit-<dit>/) and the images the
    engine reads (H3.images), which must all be there; a DiT's runtime
    adapter by adapter/<name>."""
    files = {}
    for name in [f"dit-{d}" for d in dits] + list(H3.images):
        found = _files_of(directory / name) if (directory / name).is_dir() else {}
        if name.startswith("dit-"):
            found |= {n: found.get(n) for n in (H3.adaln, H3.dit_manifest)}
        if missing := [n for n, p in found.items() if p is None] or (
            not found and [""]
        ):
            raise models.ModelError(
                f"{directory} is missing files MiniMax-H3 needs: "
                + ", ".join(f"{name}/{n}" for n in missing)
            )
        files |= {f"{name}/{n}": p for n, p in found.items()}
    return files


def _companion(name, link, installed, target):
    """A companion source at its default branch's commit, as a draft is
    resolved: one Hub request, or without one, the Hub not answering for
    target, the commit the installation records, or the Hub cache's."""
    recorded = None
    if installed:
        sources = [*installed["sources"].values()]
        sources += installed["video"].get("sources", {}).values()
        recorded = next((s["revision"] for s in sources if s["repo"] == name), None)
    return hub.Repository.resolve(
        name,
        installation=link,
        installed=recorded,
        unreachable=target.unreachable_reason,
        unreachable_kind=target.unreachable_kind,
    )


def _listed(repo):
    """repo with its files listed: a commit the Hub did not list (a verified
    installation's, which stands in when the Hub cannot answer) is read from
    its snapshot in the Hub cache, which the installation pins."""
    if repo.files or repo.revision is None:
        return repo
    return hub.Repository.cached(
        repo.name,
        repo.revision,
        unreachable_reason=repo.unreachable_reason,
        unreachable_kind=repo.unreachable_kind,
    )


def _files_of(directory: Path):
    """Each file in directory, links followed, by its name, and those of its
    adapter/ (a DiT's runtime adapter) by adapter/<name>."""
    files = {path.name: path for path in sorted(directory.iterdir()) if path.is_file()}
    if (directory / "adapter").is_dir():
        files |= {
            f"adapter/{n}": path for n, path in _files_of(directory / "adapter").items()
        }
    return files


def _links(files, encoder):
    """The assembly's files (assembly path -> source file): the images under
    weights/, the tokenizer and license files as they are, the encoder's."""

    def linked(name):
        return (
            name
            if name in H3.license_files or name.startswith("tokenizer/")
            else f"weights/{name}"
        )

    links = {linked(name): path for name, path in files.items()}
    links |= {f"encoder/{name}": path for name, path in encoder.items()}
    missing = sorted(name for name, path in links.items() if not path.is_file())
    if missing:
        raise models.ModelError("missing files: " + ", ".join(missing))
    return links


def _record(model, target, encoder, variants, converted=None):
    return {
        "version": 1,
        "model": model,
        "family": H3.name,
        "target_format": H3.target_format,
        "vision_format": "none",
        "sources": {"target": target, "encoder": encoder.identity()},
        "video": {"variants": list(variants)} | (converted or {}),
    }


def prepare(selection, target, installed):
    """Install selection's MiniMax-H3 variant from target, MiniMax's release
    as this start resolved it, with its companion and its text encoder at
    their default branches' commits, as upstream.prepare installs a
    language model and its draft: the images converted once into their
    entry, unless it is there (assembly.converting, conversion.convert), and
    published with the release's and the encoder's files in the assembly at
    the selection link; and return what it did (upstream.UPDATED and the
    other outcomes). A ModelError refuses what cannot be installed."""
    _refuse_options(selection.model, selection.draft_model, selection.language_only)
    name, variant = _variant(selection, installed)
    with hub.as_model_errors(f"cannot install {selection.model}"):
        # A commit the Hub did not list is a verified installation's.
        if target.files and not is_source(target.files):
            raise models.ModelError(
                f"{selection.model} is not MiniMax's release of MiniMax-H3: serve "
                f"{H3.release}:{H3.default_variant}"
            )
        why = target.unreachable_reason
        companion, encoder = (
            _companion(repo, selection.link, installed, target)
            for repo in (variant.companion, H3.encoder)
        )
        # Whether the Hub answered for each source this start (upstream's
        # outcomes).
        answered = all(
            repo.files and not repo.unreachable_reason
            for repo in (target, companion, encoder)
        )
        recipe = _recipe(variant)
        if target.sizes and companion.sizes:
            inputs = recipe.inputs(variant.dit, target, companion, **_spec(variant))
            key = conversion.key(
                inputs, recipe_identity(recipe), variant.dit, variant.spec
            )
        elif key := installed and installed["video"].get("key"):
            pass  # without the Hub, the verified installation's entry
        else:
            raise models.ModelError(
                f"cannot list {H3.release} and {variant.companion} on the Hub: {why}"
            )
        target = _listed(target)
        encoder = _listed(encoder)
        package = encoder_package(
            encoder, recorded_packages(selection.models_root, installed)
        )
        names = _release_files(target)
        print(H3.license, flush=True)
        pinned = [
            {"repo": repo, "revision": c}
            for repo, c in recipe.pinned(variant.dit, **_spec(variant)).items()
        ]
        companions = [companion.identity(), *pinned]
        downloaded = target.download(names)
        files = _links(
            {path: downloaded[n] for n, path in names.items()},
            encoder.download(package.downloads),
        )
        root = selection.models_root
        previous = installed and selection.link.resolve()
        with contextlib.ExitStack() as held:
            stage = None
            if not assembly.is_converted(root, key):
                stage = held.enter_context(assembly.converting(root, key))
                # Another installation may have published it meanwhile.
                if not assembly.is_converted(root, key):
                    _convert(stage, variant, target, companion, selection.model)
            encoder_stage = _encoder_entry(held, root, package, selection.model)
            converted = {
                "key": key,
                "sources": {s["repo"]: s for s in companions},
                "encoder": package.record(),
            } | ({} if name in H3.variants else {"spec": variant.spec})
            record = _record(
                selection.model, target.identity(), encoder, [name], converted
            )
            published = upstream.publish(
                root,
                selection.link,
                record,
                files,
                derive=lambda: (
                    {
                        f"weights/{name}": path
                        for name, path in assembly.converted(root, key, stage).items()
                    }
                    | _encoder_derived(root, package, encoder_stage)
                ),
            )
    print(
        f"Splash model {selection.model} is installed in {selection.link}", flush=True
    )
    if not answered:
        return upstream.UNREACHABLE
    return upstream.UP_TO_DATE if published == previous else upstream.UPDATED


def _convert(stage, variant, release, companion, model):
    """Convert the variant's images into the staging folder stage, from the
    sources at their commits, resuming what an interrupted conversion
    wrote."""
    print(
        f"Converting {model} into the engine's images as it reads MiniMax's "
        f"release and {variant.companion} (about 83 GB); an interrupted "
        "conversion resumes.",
        flush=True,
    )
    units = _recipe(variant).units(
        stage,
        variant.dit,
        release,
        _listed(companion),
        lambda arguments: models.run_engine(
            arguments, "tabulation of the adaLN tables"
        ),
        **_spec(variant),
    )
    conversion.convert(stage, units, model)


def prepare_local(
    target: Path, models_root: Path, link: Path, draft, mmproj, language_only
):
    """prepare_local_model of MiniMax-H3 in the directory target, of images
    converted before, every variant it holds; the text encoder comes from
    the Hugging Face cache at its default branch's commit. MiniMax's release
    is refused. No network request: the encoder's package is one an
    assembly under models_root records for that commit, or is planned from
    its shards in the Hub cache."""
    _refuse_options(str(target), draft, mmproj, language_only)
    if not _is_converted(target):
        raise models.ModelError(
            f"{target} is MiniMax's release of MiniMax-H3, which Splash converts as "
            f"it installs it: serve {H3.release}:{H3.default_variant}"
        )
    held = variants(target)
    files = images(target, [H3.variants[v].dit for v in held])
    files |= {n: target / n for n in H3.license_files if (target / n).is_file()}
    files |= {f"tokenizer/{n}": p for n, p in _files_of(target / "tokenizer").items()}
    encoder = hub.Repository.resolve(
        H3.encoder, unreachable="preparing local files asks no Hub"
    )
    package = encoder_package(encoder, recorded_packages(models_root))
    encoder_files = encoder.download(package.downloads)
    print(H3.license, flush=True)
    record = _record(
        str(target),
        {"repo": str(target), "revision": None},
        encoder,
        held,
        {"encoder": package.record()},
    )
    with contextlib.ExitStack() as stack:
        stage = _encoder_entry(stack, models_root, package, str(target))
        return upstream.publish(
            models_root,
            link,
            record,
            _links(files, encoder_files),
            derive=lambda: _encoder_derived(models_root, package, stage),
        )
