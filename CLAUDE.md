# Blueprint 3D

Turns 2D apartment floor plans (images from rental listings) into a 3D dollhouse view.
Personal/portfolio project.

## Architecture

```
plan image ──▶ backend (Python, FastAPI) ──▶ plan JSON ──▶ frontend (React Three Fiber) ──▶ 3D dollhouse
```

- **Everything runs locally.** Image processing, OCR and any future model inference happen on this machine (or its Docker containers); plans are never sent to a cloud service, and nothing is downloaded at runtime. OCR models are the ones bundled in the rapidocr wheel, loaded by explicit path (`ocr.py`), which bypasses RapidOCR's downloader; `tests/parsing/test_ocr.py` checks this with network access blocked. The learned parser's model is likewise loaded by explicit path from `models/` (onnxruntime, CPU); downloads happen only in `scripts/setup_cubicasa.py`, run by hand. A new dependency or parser (e.g. the vision-LLM idea on the roadmap) must keep to this.
- **The plan schema is the contract** between parsers and the renderer. It exists twice and must stay in sync:
  - `src/blueprint3d/schema.py`: source of truth (pydantic, frozen models, cross-reference validation)
  - `frontend/src/plan/schema.ts`: zod mirror (structure and defaults only)
  - `tests/fixtures/two_room_apartment.json`: shared sample, tested on both sides and loaded by the UI

  A schema change touches all three, plus tests on both sides.
- **Plan space:** centimetres, origin top-left, x right, y down. The 3D scene uses metres, with plan (x, y) mapped to world (x, z) and y up. Convert only through `CM_TO_M` in `frontend/src/geometry/units.ts`.
- **Openings** are positioned by `offset`, the distance along the wall from `start` to the opening's *centre*.
- **Geometry** is built in the frontend. `frontend/src/geometry/` holds the pure, unit-tested maths; `frontend/src/scene/` holds the React Three Fiber components. Keep maths out of components.
- **Parsers** (image to plan) live in `src/blueprint3d/parsing/` and return a `ParseResult` (plan plus meta: scale, scale source, warnings). They follow the `Parser` protocol in `pipeline.py`, which also holds the steps they share (locating the plan, scale bar, boundary config, OCR labels, `ParseMeta`). `registry.py` lists them: `opencv` always, `cubicasa` only when its exported model is in `models/`. `POST /api/plans/parse` takes a multipart image, an optional `cm_per_px` (always in *original* image pixels) and an optional `parser`; `GET /api/parsers` lists them with their availability. Roadmap: OpenCV baseline (done), CubiCasa5K pretrained model with our own raster-to-vector step (done), then training our own model, then optionally a vision LLM (local only). Keep new parsers comparable on the same samples.
- **The OpenCV parser** (`opencv_parser.py` orchestrates small single-purpose modules):
  1. Binarize the image.
  2. Walls are strokes wider than the first jump in the stroke-width histogram (`walls.py`). The plan is located with a rough mask first, then the threshold is recomputed inside the plan only, because page text skews it. The plan grows from a wall cluster with a straight run of 12x the wall thickness (bold headings stay below 8x), and the grown group must have such runs both horizontally and vertically; bar-shaped clusters (scale bars, scan borders) never take part. If no group qualifies, the walls are hatched or thin outlines: the plan is the largest ink component, and its hatching is closed into solid walls (`fill_hatching`).
  3. Extract horizontal/vertical wall pieces (`segments.py`), plus a fallback for angled walls.
  4. Merge collinear pieces; the gaps between them are openings (`openings.py`). An opening is a window if it contains 2 or more *solid* lines along the wall, checked on light-gray ink, and faces outside space. Dashed lines mean an open passage.
  5. Scale (`scale.py`), in priority order:
     - the user's value
     - the printed scale bar (`scale_bar.py`): the smallest regularly repeating spacing of the tallest marks, taken as 1 m
     - A4 proportions, assumed printed at 1:100
     - median door gap, assumed to be 80 cm
     - exterior wall thickness

     The bar and page-format assumptions are dropped if the door estimate disagrees by more than 1.6x.
  6. Rooms (`rooms.py`): enclosed space from thick walls plus plugged openings, with a fallback pass over all ink for spaces bounded by thin lines (balconies, angled bay windows). Small pockets (cupboards, counter strips, door-swing wedges) are absorbed into the room they open onto across thin lines, so floors have no holes.
  7. Complete room outlines that no wall explains (`boundaries.py`). Balconies are decided here, *before* walls are marked exterior and windows are checked for outside space: outside is space beyond all drawn lines plus balconies (`outside_space` in `rooms.py`), not the thick-wall pass's outside, which leaks through outlined walls. Beyond the crop edge also counts as outside. Probe outward from each uncovered stretch; probes stop at known walls.
     - **A room with 25% or more of its outline facing outside** (measured against its convex hull) is a `balcony`. All its outside stretches become railings (5 cm thick, half height), even when drawn as double lines.
     - **Any other room, outside behind 2+ parallel lines:** an exterior wall with a window. This covers angled walls. A single line (e.g. a door leaf drawn outside) is left alone.
     - **Another room across thin lines:** a wall with a door along the chord.
  8. Merge parallel walls of the same height that overlap and lie within 45 cm of each other into one wall spanning both faces (`merge.py`). This handles outlined (double-line) walls and duplicates from step 7.
  9. Name rooms (`ocr.py`, `room_names.py`): RapidOCR reads the plan area; text inside a room that matches the Swedish/English vocabulary sets its `name` and `kind` (fuzzy for words of 4+ letters, exact below, so cupboard markers like G/ST/KYL never match). Geometric balconies stay balconies. OCR failure only adds a warning. `OpenCvParser(read_text=None)` skips naming; tests inject fake readers.
  10. Convert to a cm `Plan` (`plan_builder.py`).
- **The learned parser** (`parsing/learned/`, name `cubicasa`): CubiCasa5K's pretrained network, exported to ONNX.
  1. `segmentation.py` runs it on the located plan (SHA-256 checked against the manifest), scaled so walls are `TARGET_WALL_PX` (10) thick, a value swept on the evaluation. Rotation averaging (`rotations=4`) exists but is off: 4x the time for about 1 room in 38.
  2. `vectorize.py`: walls are traced through their openings (Wall + Door/Window icon pixels as one stroke); icon stretches become openings of the model's kind, bare gaps become passages. The room barrier uses the walls extended by one thickness at each end, because the model stops walls just short of junctions. Outdoor rooms are balconies; Railing pixels become low railings.
  3. `learned_parser.py` then reuses the OpenCV pipeline's thin-line room pass (`merge_room_passes`) and boundary completion, so balcony outlines the model misses still close. A printed name decides a room's kind; otherwise the model's class does (`room_kinds_and_names`).
  - Tests never need the real model: `tests/parsing/learned/toy_model.py` builds a tiny ONNX stand-in, and tests on real weights are skipped when `models/cubicasa5k.onnx` is absent.
- **Unsupported drawing styles** are listed in `UNSUPPORTED_STYLES` in `tests/parsing/test_opencv_parser.py` (strict xfail, OpenCV parser only). Remove an entry once its style is supported.
- **Evaluation harness** (`src/blueprint3d/evaluation/`): run `uv run python -m blueprint3d.evaluation --parser all` after every parser change (`--parser opencv|cubicasa|all`, default opencv).
  - It parses each image in `data/` and scores it against `data/truth.json`: scale error, room and balcony counts, and living-area error against the printed area. Printed room names, and the room kinds they imply, are scored too, but only reported, not part of pass/fail. Living area follows the printed Swedish BOA definition (SS 21054): inside the exterior walls, interior walls included, balconies excluded.
  - It writes `eval-out/<parser>/overview.png` (captioned overlays of all plans) plus `report.json`; with `all` it also prints a side-by-side comparison.
  - Truth holds only facts readable off the drawing (see `truth.py`). A change that fixes one plan must not break another.
- **Tuning the parser:** change thresholds against real plans, not just the synthetic ones. Draw an overlay of walls, openings and rooms on the cropped image and look at it. Every fix gets a synthetic regression case in `tests/parsing/synthetic.py`.

## Layout

```
src/blueprint3d/   backend package (schema.py, api.py, parsing/)
tests/             backend tests (pytest) + shared fixtures; tests/parsing/synthetic.py draws test plans
frontend/          Vite + React + TypeScript app (api/, app/ hooks, geometry/, plan/, scene/, ui/)
docker/            Dockerfiles (build context is the repo root)
docs/              README figures, regenerated with `uv run python docs/make_figures.py` after parser changes
compose.yaml       dev stack (mounts models/ read-only into the backend)
scripts/           one-time setup and ONNX export of the learned model (ml group)
models/, vendor/, datasets/  exported model, CubiCasa's code + weights, training data: git-ignored, CC BY-NC
data/, samples/    real rental-site plans; each *.jpg/*.png gets a smoke test in test_opencv_parser.py
                   (and in tests/parsing/learned/test_learned_parser.py when the model is exported)
                   (samples/ is git-ignored; real plans may be copyrighted, so keep them out of git)
```

## Commands

The user's shell is PowerShell on Windows; suggest commands in that syntax.

### Backend: always use uv

Never use pip, `python -m venv` or a bare `python`/`pytest`; go through uv.

```powershell
uv sync                                    # install/update deps from uv.lock
uv add <pkg> / uv add --dev <pkg>          # add dependencies (never edit the lock by hand)
uv run pytest                              # tests
uv run pytest --cov=blueprint3d            # coverage (target 80%+)
uv run uvicorn blueprint3d.api:app --reload  # backend on :8000, docs at /docs
```

PyTorch lives in the `ml` dependency group (CUDA build from PyTorch's own index), used only by `scripts/` to export (and later train) models; the backend and Docker image never need it. A plain `uv sync` removes the group again, so use `uv run --group ml ...`:

```powershell
uv sync --group ml
uv run --group ml python scripts/setup_cubicasa.py         # vendor/ code at a pinned commit, models/ weights (SHA-256 checked)
uv run --group ml python scripts/export_cubicasa_onnx.py --check data/drheymansgata5.jpg  # models/cubicasa5k.onnx
```

`opencv-python` (the GUI build, pulled in by rapidocr) is excluded via `[tool.uv] override-dependencies`: it clashes with `opencv-python-headless` (both install `cv2`) and needs libGL in Docker. If `cv2` ever goes missing after a sync, run `uv sync --reinstall-package opencv-python-headless`.

### Frontend (from `frontend/`)

```powershell
npm run dev        # :5173, proxies /api to :8000 (override with VITE_API_PROXY_TARGET)
npm test           # vitest
npm run coverage
npm run lint       # oxlint
npm run build      # tsc -b + vite build
```

### Docker (dev stack)

```powershell
docker compose up --build   # frontend :5173, backend :8000; source is bind-mounted with hot reload
docker compose down
```

Adding a dependency changes the image, so rebuild with `--build` afterwards. Backend dependencies live in `/opt/venv` in the image and frontend `node_modules` in an anonymous volume, so the host's Windows builds never leak into the Linux containers.

## Workflow

- Test first: write a failing test, implement, refactor. Pure logic (schema, geometry, future parser post-processing) must be unit-tested.
- After frontend changes that affect rendering, check the result visually. Headless Edge works:
  `msedge --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --window-size=1280,800 --virtual-time-budget=8000 --screenshot=<out.png> http://localhost:5173/`
  (the dark strips at the right and bottom edges of these screenshots are a headless artifact, not a layout bug).
- For flows like uploads, drive the installed Edge with `playwright-core` (`chromium.launch({ channel: 'msedge' })`) from a scratch folder, and check for console errors. No browser download is needed.
- Don't use drei `<Html>` in the scene: each label mounts its own React root and breaks on remount under React 19. Use canvas-texture sprites (`scene/labels.ts`).
- On Windows, Python's `write_text` writes CRLF. Keep sources LF, e.g. pass `newline="\n"` in helper scripts.
- Commits follow conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`, `docs:`).

## Pre-commit (planned, not yet implemented)

When it's set up, add `ruff` to the dev dependencies and configure it in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 110
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "RUF"]
```

Then add `.pre-commit-config.yaml` with these hooks:

1. **pre-commit-hooks**: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-json`, `check-toml`, `check-merge-conflict`, and `check-added-large-files` (to guard against committing floor plan images or model weights).
2. **ruff-pre-commit**: `ruff-check --fix`, then `ruff-format`.
3. **uv-pre-commit**: `uv-lock`, so `uv.lock` stays in sync with `pyproject.toml`.
4. **Local frontend hooks**, limited to `^frontend/`: `npm --prefix frontend run lint` and `npm --prefix frontend run typecheck`.
5. **Tests** run at the `pre-push` stage (`uv run pytest`, `npm --prefix frontend test`), not on every commit, to keep commits fast.

Install with `uv tool install pre-commit`, then `pre-commit install --hook-type pre-commit --hook-type pre-push`.
Until it's set up, run `uv run pytest` and `npm test` / `npm run lint` manually before committing.
