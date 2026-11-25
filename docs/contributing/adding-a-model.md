# Adding a model family

A model request is the start of a feasibility study, not a promise. A family
needs a compatible architecture, a draft, kernels for its shapes, an installer
path, fixtures and real-model evidence. Start from the closest existing family;
do not add a parallel mechanism for one model.

## 1. Open the request

Use the model request issue form. Include the upstream repository, format,
quantisation, size, license, `config.json`, the attention and FFN shapes, and
whether a compatible draft exists. Without a draft, Splash can serve the target
only where the code already supports draft-free serving.

## 2. Classify the work

| Question | Why it matters |
| --- | --- |
| Dense or MoE, and how many experts are active? | Chooses the target layout and expert store. |
| Conventional attention, sliding window, linear or hybrid state? | Chooses the cache and state layout. |
| Does the vocabulary and chat template match an existing family? | Reuses the tokenizer and output parser. |
| Is there a DFlash2 draft at the same hidden width and capture layers? | Without one, decode loses speculative acceleration. |
| Which input formats? | GGUF, MLX and compressed-tensors have separate loaders. |
| Does it need vision? | Adds a tower, projector and preparation path. |

Read `docs/models.md` before choosing a layout. The model families and their
drafts, formats, vision and templates are described there.

## 3. Implement in the right layer

- `runtime/model/ModelDescriptor.mm` and `runtime/model/ModelDescriptor.hpp`:
  accept and validate the configuration.
- `runtime/model/ModelFactory.cpp` and `runtime/model/ModelFactory.hpp`: build
  the concrete model from its descriptor.
- A family header and target implementation beside the existing families in
  `runtime/model/`.
- `runtime/ops/` and `runtime/metal/`: only when the family needs an operation
  or kernel the existing operators cannot serve.
- `install/families.py`, `install/upstream.py` and `install/models.py`: pair
  the target with its draft and install the assembly.
- `docs/models.md`: the supported family, formats, draft and limits.

Do not put a family name in engine policy. The engine reads the descriptor and
the costs the model reports. Concrete models belong in the assembly files named
in `AGENTS.md`.

## 4. Add fixtures and tests

- A minimal `config.json` (and draft configuration) under
  `dev/tests/fixtures/model-configs/<family>/`.
- A loader test that accepts the fixture and rejects the configurations the
  family does not support.
- A native oracle run (`make test-real MODEL=... REVISION=...`) on a supported
  Mac.
- Vision parity where the family serves images.
- A weight-image record through the release check when the loader changes.

Pin the upstream revision in every result. A model that loads is an
intermediate milestone; it is not evidence of correct output or speed.

## 5. Prove it

Follow the release check in `docs/validate.md`. At minimum:

```sh
make check
make install test-real MODEL=<owner/repo[:variant]> REVISION=<commit>
make install test-http-real MODEL=<owner/repo[:variant]> REVISION=<commit>
```

For a speed claim, add `make test-performance-real` with a baseline and report
the ABBA comparison. For a new kernel path, add `make check-native-metal` and
the kernel identity comparison described in `docs/validate.md`.

## 6. Document the limits

State the supported formats, quantisations, Mac memory floor, context, vision
support, draft pairing and any feature that stays disabled. A family that runs
on one Mac at one quantisation is still useful, but the documentation must say
so.
