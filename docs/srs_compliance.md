# SRS compliance audit

Checked against `docs/SRS.md`. Status: ✅ met and tested · ⚠️ partly · ❌ not met yet. "Evidence" points at tests or files.
**Last audit: after the first global training run (2026-10-10).** The application and training pipeline work end to end. Final field claims still need a larger Indonesia evaluation set.

## Section 20 — Acceptance criteria (18)
| # | Criterion | Status | Evidence / what is missing |
|---|---|---|---|
| 1 | open-weight model chosen and documented | ✅ | `docs/model.md` (MobileNetV3-Small, Large fallback, ImageNet pretrained) |
| 2 | model runs / exports via ONNX Runtime | ✅ | `models/ecoid-global-20261010.onnx`; PyTorch/ONNX parity passed on 20/20 samples |
| 3 | five classes can be predicted | ✅ | trained MobileNetV3-Small checkpoint and ONNX export |
| 4 | inference on CPU | ✅ | `OnnxBackend` uses `CPUExecutionProvider` only |
| 5 | Top-K output | ✅ | top-3 in every result (`tests/ai/test_pipeline.py`) |
| 6 | confidence available | ✅ | per-candidate score, shown as bars |
| 7 | Unknown threshold | ✅ | `config/inference.toml` (0.73, calibrated on validation data, inclusive), boundary tested |
| 8 | user can verify | ✅ | Verified / Rejected / Not sure; browser tests |
| 9 | observations stored locally | ✅ | SQLite + original photos; persistence test |
| 10 | inference works without internet | ✅ | whole flow with all non-loopback sockets/DNS blocked (`tests/ui/test_app_e2e.py`) |
| 11 | evaluation set, 50 labelled images | ✅ | 50 licensed iNaturalist images, 10 per class; local Indonesian field evaluation remains a follow-up |
| 12 | negative set, 100+ images | ✅ | 149 images across lookalike, non-plant, and low-quality groups; checker passes |
| 13 | Top-1 benchmark | ⚠️ | 43/50 (86.0%) on the independent iNaturalist evaluation set; local field validation remains pending |
| 14 | Top-3 benchmark | ⚠️ | 47/50 (94.0%) on the independent iNaturalist evaluation set; local field validation remains pending |
| 15 | inference latency | ✅ | MobileNetV3 ONNX benchmark: 12.4 ms average, 33.8 ms p95 total |
| 16 | unknown / negative evaluation | ⚠️ | 84/109 negatives rejected; benchmark is based on licensed iNaturalist images and needs local field validation |
| 17 | model name recorded on every observation | ✅ | `Observation.model`, from config; test in `tests/observation/test_manager.py` |
| 18 | real demo with a physical plant | ❌ | needs a physical field run with the trained model |

**12 ✅ · 5 ⚠️ · 1 ❌.**

## Functional requirements (section 9)
| FR | Status | Notes |
|---|---|---|
| FR-001 image input from camera / local file | ✅ | file picker, phone camera input, webcam (tested with Chromium's fake camera); a *physical* camera is a manual check |
| FR-002 preprocessing 224×224, normalise, tensor | ✅ | `app/ai/preprocessing`; train/serve parity is verified by `export_model.py` (not yet run) |
| FR-003 local inference via ONNX Runtime, no cloud | ✅ | |
| FR-004 Top-1 / Top-3 | ✅ | |
| FR-005 Unknown handling | ✅ | "Low confidence" banner uses the SRS wording |
| FR-006 manual verification | ✅ | |
| FR-007 observation storage | ✅ | all listed fields + extras (`identification_status`, `user_species`, `human_label`, `captured_at`, `location_source`, `location_accuracy_m`); schema v2 with migration |

## Other sections
| Section | Status | Notes |
|---|---|---|
| 2 scope: 5 classes + UNKNOWN | ✅ | `app/ai/contract/labels.py` |
| 3 principles | ✅ | local-first, human-in-the-loop, uncertainty-aware, "Possible identification" wording, short flow |
| 4 hardware | ⚠️ | CPU-only ✅; RAM ≥ 4 GB / storage ≥ 1 GB are not measured |
| 5 model + training strategy | ✅ | two-stage fine-tune completed with MobileNetV3-Small |
| 6 dataset strategy | ⚠️ | global iNaturalist dataset collected and validated; Indonesian field/evaluation coverage remains incomplete |
| 7 model contract | ✅ | `IdentificationResult`; UI depends only on it |
| 8 unknown detection | ✅ / ⚠️ | threshold + rule ✅; calibrated threshold is 0.73 on the current validation set, but calibration should be repeated after field data is added |
| 10 model information | ✅ | ⓘ dialog: model, runtime, "Local / CPU", "5 plant species", file hash |
| 11 offline behaviour | ✅ | wording matches the SRS ("offline inference after model installation") |
| 12 local storage | ✅ | original photo kept byte-identical, separate from the 224 px tensor |
| 13 benchmark | ⚠️ | metrics and provisional real-model numbers exist; final numbers wait for a complete independent evaluation set |
| NFR-001 offline | ✅ | |
| NFR-002 ≤ 10 s / image | ✅ | measured MobileNetV3-Small ONNX p95 is 33.8 ms per image on the development CPU |
| NFR-003 privacy | ✅ | no outbound connections (tested), CSP, loopback-only by default, location opt-in, `docs/privacy.md` |
| NFR-004 reproducibility | ⚠️ | pinned runtime deps, seed, model SHA-256, dataset metadata, and attribution are recorded; final field evaluation is still pending |
| NFR-005 modularity | ✅ | `ModelBackend` protocol; backend swap tested |
| 15–16 architecture / pipeline | ✅ | `docs/architecture.md` |
| 17 out of scope | ✅ | nothing out-of-scope was built (no cloud inference, disease diagnosis, marketplace, RAG, chatbot…) |
| 21 claims | ✅ | no accuracy claim anywhere; statistics screens say they are not a benchmark |

## Section 18 — roadmap
| SRS phase | Status |
|---|---|
| Core identification | ✅ (pipeline + initial trained model); field validation is pending |
| Field observation | ✅ (photo storage, verification, notes, timestamp, history, search, corrections) |
| **Eco Mapper** (GPS, observation map, statistics, export) | ✅ — offline map; GPS from EXIF / device / typed |
| Micro-Farm Companion (garden tracking, plant care, growth monitoring) | ❌ not started |
| Community Science (shared observations, dataset contribution, collaborative verification) | ❌ not started — and it conflicts with the SRS's own out-of-scope list ("social network", "collaborative database"), so it needs a decision first |

Note: this repository numbers its build phases differently (1 pipeline, 2 model/data, 3 application, 4 field observation, 5 Eco Mapper); see `docs/SRS.md` section 18.

## Section 19 — repository structure
| Item | Status |
|---|---|
| README.md, CONTRIBUTING.md | ✅ |
| **LICENSE** | ✅ | MIT License |
| docs/architecture, model, dataset, benchmark, privacy | ✅ (plus SRS, data_collection, ci, phase*.md, this file) |
| app/ai/{inference, preprocessing, contract} | ✅ |
| app/ai/**model**/ | ⚠️ not present: model weights live in `models/`, the runtime wrapper in `app/ai/inference/backend.py` |
| app/observation, storage, ui | ✅ |
| data/{train, evaluation, negative, metadata} | ✅ layout (+ `val`, `negative_val`, explained in `docs/dataset.md`); contents empty |
| models/ | ✅ (empty until you train) |
| tests/{ai, observation, storage} | ✅ (+ data, ui, ui_browser) |
| scripts/{evaluate, benchmark, export_model}.py | ✅ |
| .github/workflows | ✅ |

## The shortest path to ✅ everywhere that is still ❌
1. Add more Indonesian field photos and complete the 50-image evaluation set; keep the split independent from training. *(criteria 11, 12)*
2. Re-run calibration and final evaluation on the expanded, frozen evaluation set. *(13–16)*
3. Walk to a real plant with the app and record a physical demo. *(18)*
