# SRS Final — EcoID

## Offline AI Field Mapper

**Status:** Final Implementation Specification
**Project:** EcoID — Offline AI Field Mapper

---

## 1. Pendahuluan

### 1.1 Tujuan

EcoID adalah aplikasi **offline-first AI field mapper** yang membantu pengguna mengidentifikasi dan mendokumentasikan tanaman melalui kamera/perangkat gambar dengan memanfaatkan **open-weight vision model yang dijalankan secara lokal**.

EcoID tidak dirancang untuk menggantikan observasi manusia. Sistem menghasilkan kandidat identifikasi berdasarkan gambar, kemudian pengguna melakukan verifikasi terhadap tanaman sebenarnya.

> **EcoID doesn't ask you to trust the AI. It asks you to verify it.**

Tujuan utama:

1. menjalankan inferensi AI secara lokal;
2. menghasilkan kandidat identifikasi tanaman;
3. menangani hasil dengan confidence rendah;
4. memungkinkan pengguna memverifikasi hasil;
5. menyimpan observasi secara lokal;
6. menyediakan benchmark yang dapat direproduksi.

---

# 2. Scope

Scope dibatasi secara sengaja pada **5 kelas tanaman**:

| Class | Scientific Name     | Common Name |
| ----- | ------------------- | ----------- |
| 1     | `Mangifera indica`  | Mangga      |
| 2     | `Cocos nucifera`    | Kelapa      |
| 3     | `Musa acuminata`    | Pisang      |
| 4     | `Carica papaya`     | Pepaya      |
| 5     | `Manihot esculenta` | Singkong    |

Selain lima kelas tersebut, sistem harus mampu menangani input yang tidak cukup meyakinkan sebagai:

```text
UNKNOWN / LOW_CONFIDENCE
```

---

# 3. Prinsip Desain

EcoID mengikuti lima prinsip utama.

### 3.1 Local-first

Inferensi utama dilakukan pada perangkat pengguna.

Tidak ada ketergantungan terhadap API AI cloud untuk fungsi identifikasi.

### 3.2 Human-in-the-loop

AI hanya memberikan kandidat.

Keputusan akhir dapat dilakukan pengguna melalui:

* Verified
* Rejected
* Uncertain

### 3.3 Uncertainty-aware

Model tidak boleh selalu memaksakan salah satu dari lima kelas.

Jika confidence tidak memenuhi threshold, sistem harus menunjukkan:

> **Low Confidence — manual verification required**

### 3.4 Evidence over authority

Hasil AI harus disajikan sebagai kemungkinan berdasarkan gambar, bukan sebagai fakta biologis yang pasti.

### 3.5 Touch Grass

Interaksi dengan AI dibuat sesingkat mungkin.

Alur ideal:

**Take photo → AI inference → Look at the real plant → Verify → Put the phone down**

---

# 4. Target Hardware

Aplikasi ditargetkan berjalan pada:

* laptop developer;
* CPU lokal;
* perangkat edge dengan kemampuan CPU yang setara.

GPU tidak diperlukan untuk fungsi utama.

### Minimum expectation

| Component | Target                                  |
| --------- | --------------------------------------- |
| CPU       | Modern x86-64 CPU                       |
| GPU       | Tidak diperlukan                        |
| RAM       | ≥ 4 GB                                  |
| Storage   | ≥ 1 GB untuk aplikasi + model           |
| Network   | Tidak diperlukan setelah model tersedia |

Angka hardware minimum dapat direvisi setelah benchmark aktual.

---

# 5. AI Model Architecture

## 5.1 Model Strategy

EcoID menggunakan **open-weight lightweight vision model**.

**Model default: MobileNetV3-Small (ImageNet pretrained)**, dengan **MobileNetV3-Large sebagai fallback** apabila benchmark menunjukkan akurasi Small tidak memadai.

Kriteria pemilihan yang menjadi acuan:

1. kemampuan klasifikasi tanaman;
2. ukuran model;
3. CPU inference latency;
4. akurasi;
5. lisensi;
6. kemampuan export ke ONNX;
7. reproducibility;
8. dokumentasi;
9. kemampuan menangani input di luar lima kelas.

## 5.2 Training Strategy

Model dilatih dengan **transfer learning** dari bobot ImageNet:

```
Base      : MobileNetV3-Small (ImageNet pretrained)
Input     : 224 × 224
Freeze    : semua layer kecuali blok terakhir + classifier head
Fine-tune : head baru (5 kelas), lalu unfreeze 15–30 layer terakhir
            dengan learning rate rendah (1e-4, cosine decay)
Augment   : rotation, horizontal flip, brightness/contrast, random crop
            (TANPA shear/warp agresif yang mengubah bentuk daun)
```

Training dari nol (scratch) **tidak diperbolehkan**.

---

# 6. Dataset Strategy

Dataset dipisahkan menjadi:

```text
Training Dataset (300–500 gambar per kelas)
        │
        ├── Mangifera indica
        ├── Cocos nucifera
        ├── Musa acuminata
        ├── Carica papaya
        └── Manihot esculenta

Evaluation Dataset (50 labelled images)
        │
        ├── 10 × Mangifera indica
        ├── 10 × Cocos nucifera
        ├── 10 × Musa acuminata
        ├── 10 × Carica papaya
        └── 10 × Manihot esculenta

Negative / Out-of-class Dataset (100+ gambar, terpisah)
        │
        ├── tanaman tropis yang menyerupai kelas target
        │     (rambutan, nangka, sirsak, jeruk)
        ├── objek non-tanaman
        └── gambar blur / low-quality
```

### Sumber data

1. **Sumber terbuka**: iNaturalist, GBIF, subset PlantCLEF — dengan pengecekan lisensi per gambar.
2. **Koleksi sendiri**: foto lapangan multi-organ (daun, batang, buah, habit).
3. **Dataset penyakit tanaman** (mis. dataset leaf/fruit disease): hanya boleh digunakan sebagai **augmentasi negative samples**, bukan training utama.

### Aturan dataset

* Training image dan evaluation image **tidak boleh berasal dari gambar yang sama**.
* Evaluation set idealnya berbeda sumber/kondisi pengambilan dari training set.
* **Negative samples wajib menyertakan tanaman yang secara visual menyerupai kelas target** — bukan hanya objek non-tanaman.
* Semua data web-scraped wajib melewati deduplikasi dan QC manual label.

---

# 7. Model Contract

Model tidak boleh terikat langsung dengan UI.

EcoID mendefinisikan interface konseptual:

```text
Image
  ↓
Preprocessing
  ↓
Vision Model
  ↓
IdentificationResult
```

### 7.1 Input

```text
identify(image)
```

### 7.2 Output

```json
{
  "candidates": [
    {
      "label": "Mangifera indica",
      "score": 0.91
    },
    {
      "label": "Carica papaya",
      "score": 0.05
    },
    {
      "label": "Manihot esculenta",
      "score": 0.02
    }
  ],
  "status": "IDENTIFIED",
  "model": "EcoID Vision Model"
}
```

Model wrapper bertanggung jawab terhadap:

* preprocessing;
* inference;
* post-processing;
* Top-K extraction;
* confidence calculation;
* unknown decision.

UI tidak boleh bergantung langsung pada implementasi internal model.

---

# 8. Unknown Detection

Threshold:

```text
UNKNOWN_THRESHOLD = 0.65
```

Aturan:

```text
max(score) >= 0.65
        ↓
IDENTIFIED

max(score) < 0.65
        ↓
LOW_CONFIDENCE / UNKNOWN
```

Contoh tampilan:

```text
Possible identification

Mangifera indica    61%
Cocos nucifera      18%
Carica papaya       11%

⚠ Low confidence

The model does not have enough confidence
to provide a reliable identification.

Please inspect the plant and verify manually.
```

### Engineering note

`0.65` adalah **threshold awal**, bukan klaim bahwa 65% adalah probabilitas kebenaran.

Setelah benchmark, threshold dikalibrasi berdasarkan:

* false positive rate;
* false negative behavior;
* calibration;
* unknown detection performance (unknown rejection rate pada negative set).

Metrik tambahan yang dicatat: **selective accuracy** — akurasi hanya pada sampel yang berada di atas threshold.

---

# 9. Functional Requirements

## FR-001 — Image Input

Sistem harus menerima gambar tanaman dari:

* kamera;
* file lokal.

## FR-002 — Image Preprocessing

Sistem harus melakukan preprocessing sebelum inference:

```text
Image
 → Resize (224 × 224)
 → Normalize
 → Tensor conversion
 → Model input
```

## FR-003 — Local Inference

Sistem harus menjalankan inference menggunakan model lokal melalui **ONNX Runtime**.

Cloud inference tidak boleh menjadi dependency untuk fungsi utama.

## FR-004 — Top-K Prediction

Sistem harus menghasilkan kandidat identifikasi dengan score.

Minimal:

```text
Top-1
Top-3
```

## FR-005 — Unknown Handling

Sistem harus dapat menghasilkan `UNKNOWN` / `LOW_CONFIDENCE` ketika hasil model berada di bawah threshold.

## FR-006 — Manual Verification

Pengguna dapat memilih:

```text
VERIFIED
REJECTED
UNCERTAIN
```

## FR-007 — Observation Storage

Setiap observasi menyimpan:

```json
{
  "id": "...",
  "image_path": "...",
  "predicted_species": "Mangifera indica",
  "confidence": 0.91,
  "alternative_predictions": [],
  "verification_status": "VERIFIED",
  "timestamp": "...",
  "latitude": null,
  "longitude": null,
  "notes": "",
  "model": "EcoID Vision Model"
}
```

GPS bersifat opsional.

---

# 10. Model Information

Pengguna harus dapat mengetahui model yang digunakan.

```text
Model
EcoID Vision Model

Runtime
ONNX Runtime

Inference
Local / CPU

Classes
5 plant species
```

Tujuannya adalah **transparency**, bukan sekadar UX.

---

# 11. Offline Behavior

EcoID membedakan dua kondisi:

### First installation

Internet **dapat diperlukan** untuk:

* mendapatkan application package;
* mendapatkan model;
* mendapatkan model assets.

### Setelah model tersedia

Aplikasi harus dapat melakukan:

```text
Image → Inference → Identification → Verification → Storage
```

tanpa koneksi internet.

Klaim yang benar adalah:

> **Offline inference after model installation**

bukan:

> "EcoID works completely offline from the first installation."

---

# 12. Local Storage

Foto observasi disimpan secara lokal.

Sistem harus memisahkan:

```text
Original / verification image
```

dari:

```text
Model input tensor
```

Model menggunakan input `224 × 224`, sementara foto asli dipertahankan dalam resolusi lebih tinggi untuk kebutuhan verifikasi manusia. Ini mencegah pengorbanan kualitas foto hanya karena model membutuhkan input kecil.

---

# 13. Benchmark

Benchmark wajib reproducible.

### Metrics

**Top-1 Accuracy**

```text
correct top prediction / total labelled samples
```

**Top-3 Accuracy** — prediksi benar berada dalam tiga kandidat teratas.

**Inference Latency** — milliseconds / image, mencatat:

* average latency;
* median latency;
* hardware yang digunakan.

**Unknown Evaluation** — negative samples untuk mengukur kemampuan model menolak input di luar lima kelas (unknown rejection rate).

**Selective Accuracy** — akurasi hanya pada sampel di atas threshold.

### Format laporan

```text
EcoID Vision Model
==================

Evaluation samples: 50

Top-1 Accuracy: XX%
Top-3 Accuracy: XX%

Average inference latency: XXX ms
Median inference latency: XXX ms

Unknown detection:
  Negative samples: XX
  Correctly rejected: XX

Selective accuracy (above threshold): XX%
```

**Angka tidak boleh diisi sebelum benchmark nyata dilakukan.** Ini menjaga integritas proyek.

---

# 14. Non-Functional Requirements

### NFR-001 — Offline
Inference harus dapat berjalan tanpa internet setelah model tersedia.

### NFR-002 — Performance
Target awal: `≤ 10 seconds / image`, diperketat berdasarkan hasil benchmark.

### NFR-003 — Privacy
Foto dan lokasi tidak dikirim ke server untuk fungsi identifikasi.

### NFR-004 — Reproducibility
Model, preprocessing, dataset metadata, dan benchmark harus terdokumentasi.

### NFR-005 — Modularity
Model dapat diganti tanpa menulis ulang UI dan observation layer.

---

# 15. Architecture

```text
┌─────────────────────────────┐
│            UI               │
│                             │
│ Camera / Gallery / History  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Application Layer        │
│                             │
│ Observation Manager         │
│ Verification Manager        │
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐ ┌────────────────┐
│ AI Inference │ │ Local Storage  │
│              │ │                │
│ Preprocess   │ │ Observations   │
│ ONNX Runtime │ │ Images         │
│ Postprocess  │ │ Metadata       │
└──────┬───────┘ └────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ IdentificationResult        │
│                             │
│ Top-K candidates             │
│ Confidence                  │
│ Unknown status              │
│ Model name                  │
└─────────────────────────────┘
```

---

# 16. AI Pipeline

```text
              User
                │
                ▼
          Capture Image
                │
                ▼
          Preprocessing
                │
                ▼
        Local Vision Model
                │
                ▼
         Top-K Candidates
                │
                ▼
       Confidence Evaluation
          ┌─────┴─────┐
          │           │
       >= 0.65       < 0.65
          │           │
          ▼           ▼
     IDENTIFIED    UNKNOWN /
                    LOW CONF.
          │           │
          └─────┬─────┘
                ▼
        Human Verification
                │
          ┌─────┼─────┐
          ▼     ▼     ▼
       Verified Rejected Uncertain
                │
                ▼
         Save Observation
```

---

# 17. Out of Scope

Fitur berikut **tidak masuk scope saat ini**:

* plant disease diagnosis;
* pesticide recommendation;
* fertilizer recommendation;
* automated farming;
* IoT;
* remote AI inference;
* social network;
* marketplace;
* collaborative database;
* complex RAG;
* chatbot biologis;
* automatic plant care;
* 100% identification guarantee.

---

# 18. Roadmap

### Phase 1 — Core Identification
Model, ONNX runtime, preprocessing, Top-K, confidence, Unknown.

### Phase 2 — Field Observation
Photo storage, verification, notes, timestamp, history.

### Phase 3 — Eco Mapper
GPS, observation map, field statistics, export.

### Phase 4 — Micro-Farm Companion
Garden tracking, plant care, growth monitoring, observation history.

### Phase 5 — Community Science
Shared observations, dataset contribution, collaborative verification, open ecological dataset.

> **Catatan implementasi:** urutan pengerjaan yang dipakai di repo ini berbeda dari roadmap produk di atas
> (Phase 1 Pipeline correctness → Phase 2 Model/data quality → Phase 3 Application → Phase 4 Field observation
> → Phase 5 Eco Mapper). Lihat `docs/phase1.md` dan `docs/phase2.md`.

---

# 19. Repository Structure

```text
ecoid/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
│
├── docs/
│   ├── architecture.md
│   ├── model.md
│   ├── dataset.md
│   ├── benchmark.md
│   └── privacy.md
│
├── app/
│   ├── ai/
│   │   ├── model/
│   │   ├── inference/
│   │   ├── preprocessing/
│   │   └── contract/
│   │
│   ├── observation/
│   ├── storage/
│   └── ui/
│
├── data/
│   ├── train/
│   ├── evaluation/
│   ├── negative/
│   └── metadata/
│
├── models/
│
├── tests/
│   ├── ai/
│   ├── observation/
│   └── storage/
│
├── scripts/
│   ├── evaluate.py
│   ├── benchmark.py
│   └── export_model.py
│
└── .github/
    └── workflows/
```

---

# 20. Acceptance Criteria

* [ ] model open-weight telah dipilih dan didokumentasikan;
* [ ] model dapat diekspor/dijalankan melalui ONNX Runtime;
* [ ] lima kelas tanaman dapat diprediksi;
* [ ] inference berjalan di CPU;
* [ ] Top-K output tersedia;
* [ ] confidence tersedia;
* [ ] threshold Unknown tersedia;
* [ ] pengguna dapat melakukan verification;
* [ ] observasi tersimpan secara lokal;
* [ ] inference dapat berjalan tanpa internet setelah model terpasang;
* [ ] evaluation dataset (50 gambar) tersedia;
* [ ] negative dataset (100+ gambar) tersedia;
* [ ] benchmark Top-1 tersedia;
* [ ] benchmark Top-3 tersedia;
* [ ] inference latency tersedia;
* [ ] unknown/negative evaluation dilakukan;
* [ ] model name dicatat pada setiap observasi;
* [ ] demo nyata menggunakan tanaman fisik dapat dilakukan.

---

# 21. Batasan Klaim

Sebelum benchmark, jangan menulis:

> "EcoID accurately identifies five plant species."

Lebih aman:

> **"EcoID generates local plant identification candidates for five target species and asks the user to verify the result."**

Setelah benchmark:

> **"On our 50-image evaluation set, EcoID achieved X% Top-1 accuracy and Y% Top-3 accuracy, with Z% unknown rejection rate on negative samples."**

Klaim harus **mengikuti data**.
