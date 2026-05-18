# Spec: Benchmark Tab — ICAR Accreditatie Integratie

**Datum:** 2026-05-01
**Status:** Revised (post spec-review v2)
**Project:** bovi dashboard

---

## Achtergrond & Probleemstelling

Het ICAR Portal (een apart project) implementeert een accreditatie-workflow voor melkvee-organisaties: genereer een testset van koeien, laat externe partijen hun eigen 305-dag melkopbrengst-berekening uitvoeren, en vergelijk hun resultaten met de ICAR-referentiewaarden. Bovi heeft al de onderliggende bouwstenen (preset datasets in Azure Blob, test-interval endpoint, CSV-parser voor ICAR-formaat) maar mist de workflow om challenges aan te maken, submissions bij te houden, en resultaten te vergelijken.

**Probleem:** Organisaties die hun berekeningmethode willen benchmarken hebben nu een apart portaal nodig. Bovi heeft de technische capaciteit om dit in één platform te bieden, inclusief directe vergelijking met bovi's eigen modellen.

---

## Doelen

1. Nieuwe `/benchmark` tab toevoegen aan het bovi dashboard
2. Gebruikers kunnen kiezen tussen bovi-modellen (Pad A) of eigen CSV-upload (Pad B)
3. Resultaten worden vergeleken met ICAR-referentie én bovi-modellen
4. PDF-rapport downloadbaar met drie vergelijkingssmaken
5. Auth bewust uitgesteld — DB-schema is auth-ready maar geen middleware in deze fase

---

## Scope

**In scope:**
- Challenge aanmaken vanuit bestaande preset datasets (Aurora, Sunnyside; ICAR als toekomstige extensie)
- Batch 305-dag opbrengst berekening voor klassieke modellen via bestaand TIM-endpoint
- Submission opslaan en statistieken berekenen (Pearson, RMSE, MAE, MAPE, per pariteit)
- PDF-rapport generatie (drie smaken)
- Lijst van challenges en submissions in dashboard

**Buiten scope (later):**
- Auth0 authenticatie en autorisatie
- Admin-view voor alle submissions
- Per-organisatie data-isolatie
- ICAR dataset als preset-bron (vereist uitbreiding `DatasetKey` in `datasets.py`)
- Autoencoder in Pad A (vereist aparte batch-autoencoder orchestratie — defer)

---

## Architectuurbeslissingen

| Beslissing | Keuze | Reden |
|---|---|---|
| Nieuwe tab vs. uitbreiding `/curves` | Nieuwe `/benchmark` tab | Conceptueel andere workflow; houdt `/curves` simpel |
| Auth | Uitgesteld | Core-functionaliteit eerst; schema klaar voor later |
| DB-opslag | SQLite (bestaand; Azure Files in prod) | Nieuwe tabellen, consistent met rest van bovi |
| Dataset-bron | Aurora + Sunnyside preset blobs | ICAR dataset toevoegen aan `DatasetKey` is buiten scope |
| Reference berekening | Bestaand `/curves/test-interval` | Endpoint is al batch-capable via `test_ids` array; geen nieuwe proxy route nodig |
| Upload-formaat Pad B | CSV (niet Excel) | Herbruikbaar met nieuwe `parse_submission_csv()` op basis van bestaande parseer-infrastructuur; Excel vereist openpyxl dependency |
| Grootte-limiet challenges | Max 1000 koeien (size=medium) | Large (5000) overschrijdt de 60s proxy timeout bij synchrone verwerking |
| PDF-rapport | Gegenereerd in FastAPI (fpdf2) | Server-side generatie, geen browser-afhankelijkheid |
| Flavor 2/3 voor Pad B | Via optioneel `bovi_yields` veld | Backend berekent automatisch bovi-TIM bij elke submission, zodat alle drie smaken altijd beschikbaar zijn |

---

## Datamodel

### Challenge

```python
class Challenge(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: str | None = Field(default=None)        # nullable — auth-ready
    dataset: str                                      # "aurora" | "sunnyside"
    size: str                                         # "small" | "medium" (large buiten scope)
    period: str                                       # "recent" | "old" | "mixed"
    cow_metadata: dict = Field(sa_column=Column(JSON))
    # {cow_id: {parity: int, dim: list[int], milk_kg: list[float]}}
    # Opgeslagen bij aanmaken; parity nodig voor by_parity stats
    reference_yields: dict = Field(sa_column=Column(JSON))
    # {cow_id: float} — TIM-berekend bij aanmaken
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
```

### Submission

```python
class Submission(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    challenge_id: int = Field(foreign_key="challenge.id")
    user_id: str | None = Field(default=None)        # nullable — auth-ready
    submission_type: str                             # "bovi_model" | "own_method"
    model_type: str | None = Field(default=None)    # "tim" | "wood" | "wilmink" | etc.
    organization: str | None = Field(default=None)
    country: str | None = Field(default=None)
    calculation_method: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    submitted_yields: dict = Field(sa_column=Column(JSON))
    # {cow_id: float} — ingediende 305-dag opbrengsten
    bovi_yields: dict = Field(sa_column=Column(JSON))
    # {cow_id: float} — altijd aanwezig: TIM-berekend bij elke submission
    # Maakt Flavor 2 + 3 altijd beschikbaar, ook voor Pad B
    stats: dict = Field(sa_column=Column(JSON))
    # Zie stats-structuur hieronder
    failed_cow_ids: list = Field(sa_column=Column(JSON), default_factory=list)
    # Koeien waarvoor berekening mislukt is (uitgesloten uit stats)
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
```

### Stats-structuur (opgeslagen in `Submission.stats`)

```json
{
  "overall": {
    "pearson": 0.97,
    "rmse": 45.2,
    "mae": 38.1,
    "mape": 4.3,
    "n": 298
  },
  "by_parity": {
    "1":  {"pearson": 0.96, "rmse": 48.1, "mae": 40.2, "mape": 4.8, "n": 120},
    "2":  {"pearson": 0.97, "rmse": 43.5, "mae": 37.0, "mape": 4.1, "n": 95},
    "3+": {"pearson": 0.98, "rmse": 41.0, "mae": 35.5, "mape": 3.9, "n": 83}
  },
  "failed_count": 2
}
```

`n` reflecteert het aantal koeien na uitsluiting van `failed_cow_ids`.

---

## API-endpoints

### Nieuwe benchmark-routes (`apps/backend/api/src/bovi_api/routes/benchmark.py`)

```
POST   /benchmark/challenges
  Body: {dataset: "aurora"|"sunnyside", size: "small"|"medium", period: "recent"|"old"|"mixed"}
  Response: Challenge
  Actie:
    1. Haal preset-blob op via bestaand GET /datasets/presets/{dataset}/{size}/{period}
    2. Sample N koeien (all voor small/medium — max 1000)
    3. Roep bestaand POST /curves/test-interval aan (batch via test_ids array)
       voor alle koeien tegelijk
    4. Sla reference_yields + cow_metadata (parity, dim[], milk_kg[]) op

GET    /benchmark/challenges
  Response: list[Challenge] (newest first, max 100)

GET    /benchmark/challenges/{id}
  Response: Challenge

GET    /benchmark/challenges/{id}/export
  Response: CSV (text/csv)
  Formaat: cow_id,parity,dim,milk_kg  (één rij per testdag-meting)
  Bron: challenge.cow_metadata — bevat dim[] en milk_kg[] per koe
  Noot: TestDate/CalvingDate zijn NIET aanwezig in PresetCow; export bevat DIM-waarden

POST   /benchmark/challenges/{id}/submissions
  Body (Pad A): {submission_type="bovi_model", model_type, organization?, country?, notes?}
  Body (Pad B): multipart/form-data: file (CSV) + {submission_type="own_method",
                organization?, country?, calculation_method?, notes?}

  Pad A actie:
    1. Haal cow_metadata op uit Challenge
    2. Roep POST /curves/test-interval (of ander proxy-endpoint) aan
       met het gekozen model voor alle koeien
    3. Sla submitted_yields op (= resultaat van het gekozen model)
    4. bovi_yields = submitted_yields (Pad A: bovi IS de methode)
    5. Bereken stats

  Pad B actie:
    1. Parseer CSV via nieuwe parse_submission_csv() → {cow_id: float}
    2. Valideer: alle cow_ids aanwezig in Challenge? Zo niet → 422
    3. Koeien met failed parse → failed_cow_ids
    4. Roep POST /curves/test-interval aan voor alle koeien (TIM) → bovi_yields
    5. Bereken stats voor submitted_yields vs reference_yields

  Foutafhandeling:
    - Als >20% van koeien mislukt → 422 (te veel missende data)
    - Als <20% mislukt → accepteer, sla failed_cow_ids op, bereken stats op resterende koeien

  Response: Submission (inclusief stats)

GET    /benchmark/submissions
  Response: list[Submission] (newest first)

GET    /benchmark/submissions/{id}
  Response: Submission

GET    /benchmark/submissions/{id}/report
  Query: ?flavor=icar|bovi|all  (default: all)
  Response: PDF (application/pdf)
```

### Geen nieuwe proxy-routes

Het bestaande `POST /curves/test-interval` (in `proxy.py`) ondersteunt al meerdere koeien via een `test_ids` array (`list[int | str]`). Cow-IDs uit de preset blobs zijn strings — dit is compatibel: de Function App echoot `test_id` terug zoals aangeleverd, zodat de response direct gemapt kan worden op `{cow_id: float}`.

De benchmark-route handler roept dit endpoint **niet** aan via een interne HTTP-call naar zichzelf, maar via een gedeelde utility-functie. De blob-fetch logica in `datasets.py` wordt geëxtraheerd naar `fetch_preset_cows(dataset, size, period) -> list[PresetCow]` — een gewone async functie die zowel door `datasets.py` als door `benchmark.py` geïmporteerd kan worden. Geen nieuwe route toe te voegen aan `proxy.py`.

---

## Submission flow

```
Challenge aangemaakt (N koeien, cow_metadata + reference_yields in DB)
         │
         ├─── Pad A: "Bovi berekent"
         │    Gebruiker kiest model (wood/tim/wilmink/...)
         │    → Backend batch-predict via POST /curves/test-interval
         │    → submitted_yields = bovi_yields (zelfde model)
         │    → stats berekend tegen reference_yields
         │
         └─── Pad B: "Eigen methode"
              Gebruiker downloadt CSV (cow_id, parity, dim, milk_kg)
              → Rekent extern
              → Upload CSV terug (cow_id, yield_305day)
              → Backend: parse_submission_csv() → submitted_yields
              → Backend: automatisch TIM berekenen → bovi_yields
              → stats berekend voor submitted_yields vs reference_yields
```

---

## Rapport-smaken

| Smaak | Vergelijking | Altijd beschikbaar? |
|---|---|---|
| 1 (`icar`) | Eigen methode ↔ ICAR referentie | Ja |
| 2 (`bovi`) | Eigen methode ↔ Bovi TIM | Ja (bovi_yields altijd aanwezig) |
| 3 (`all`) | Eigen methode ↔ ICAR referentie ↔ Bovi TIM | Ja |

Elke smaak bevat: scatter plot, statistieken-tabel (overall + per pariteit).

---

## Nieuwe hulpfuncties backend

### `parse_submission_csv(file: bytes) -> dict[str, float]`

Nieuw in `apps/backend/api/src/bovi_api/benchmark_ingestion.py`.
Parseert het Pad B CSV-formaat: `cow_id,yield_305day`.
Retourneert `{cow_id: float}`. Gooit `ValueError` bij ongeldige rijen (worden opgeslagen in `failed_cow_ids`).
**Niet** hetzelfde als `parse_csv()` uit `herd_stats_ingestion.py` — die aggregeert naar herd-niveau.

### `calculate_comparison_stats(submitted, reference, parities) -> dict`

Nieuw in `apps/backend/api/src/bovi_api/benchmark_stats.py`.
Berekent Pearson, RMSE, MAE, MAPE overall en per pariteit (1, 2, 3+).
Gebruikt `scipy.stats.pearsonr` en `sklearn.metrics`. **Beide zijn expliciete nieuwe dependencies** — toevoegen aan `apps/backend/api/pyproject.toml`: `scipy>=1.13` en `scikit-learn>=1.5`.

---

## Frontend-structuur

```
apps/frontend/dashboard/src/app/(dashboard)/benchmark/
├── page.tsx                        → lijst challenges + "Nieuwe challenge" knop
├── new/page.tsx                    → challenge aanmaken (dataset, size, period)
├── [id]/page.tsx                   → challenge detail: submit + resultaten
├── components/
│   ├── challenge-card.tsx
│   ├── submission-form-bovi.tsx     → Pad A: model-kiezer + submit knop
│   ├── submission-form-upload.tsx   → Pad B: CSV-upload veld + metadata
│   ├── submission-form.tsx          → wrapper: tabbed switcher tussen Pad A en Pad B
│   └── comparison-results.tsx      → statistieken-tabel + PDF-downloadknop (flavor select)
└── hooks/
    ├── use-challenges.ts
    └── use-submissions.ts
```

`submission-form.tsx` is een dunne wrapper met een Mantine `Tabs`-component. De twee sub-componenten hebben volledig gescheiden state en React Query mutation-calls.

### Uitbreidingen op bestaande bestanden

**`navigation.ts`** — `/benchmark` toevoegen aan sidebar
**`api-client.ts`** — 8 nieuwe API-functies (zie endpoints hierboven)
**`types/api.ts`** — Zod-schemas voor `Challenge`, `Submission`, `ComparisonStats`

---

## Hergebruik van bestaande code

| Bestaand | Hergebruik |
|---|---|
| `GET /datasets/presets/{dataset}/{size}/{period}` | Cow-pool ophalen + cow_metadata voor Challenge |
| `POST /curves/test-interval` (bestaande proxy) | Batch reference-berekening + bovi_yields bij Pad B |
| `blob_utils.py` (bovi-core) | Azure Blob toegang in `datasets.py` (al in gebruik) |
| `sampleTestDays()` (frontend) | Nog niet ingezet; autoencoder-pad uitgesteld |
| Mantine `Tabs` | Switcher in `submission-form.tsx` |

**Niet hergebruikt (eerder incorrect aangenomen):**
- `parse_csv()` uit `herd_stats_ingestion.py` — aggregeert naar herd-niveau, niet bruikbaar voor per-koe Pad B parsing

---

## Auth-pad voor later

Schema is nu al auth-ready (`user_id` nullable op beide tabellen). Wanneer auth wordt toegevoegd:
1. Auth0 tenant opzetten → `@auth0/nextjs-auth0` SDK
2. Next.js middleware voor `/benchmark` routes
3. `user_id` vullen bij aanmaken Challenge/Submission
4. Admin-view: filter op rol in JWT
5. Org-isolatie: Organization model + query-filter

Geen DB-migratie nodig.

---

## Bekende beperkingen & toekomstige extensies

| Beperking | Reden | Pad |
|---|---|---|
| ICAR dataset niet als challenge-bron | `DatasetKey` uitbreiden + blob-pad toevoegen | Aparte taak |
| Autoencoder in Pad A | Vereist eigen batch-orchestratie | Aparte taak |
| Large (5000 koeien) challenges | 60s proxy timeout | Background tasks of async job queue |
| Export bevat DIM, niet TestDate/CalvingDate | PresetCow heeft deze velden niet | Uitbreiding blob-schema |

---

## Verificatie

1. **Backend**: `cd apps/backend/api && just test`
   - `POST /benchmark/challenges` → Challenge met `cow_metadata` + `reference_yields`
   - `POST /benchmark/challenges/{id}/submissions` (Pad A) → stats aanwezig, `bovi_yields == submitted_yields`
   - `POST /benchmark/challenges/{id}/submissions` (Pad B CSV) → stats aanwezig, `bovi_yields` automatisch gevuld
   - Pad B met >20% ongeldige koeien → 422
   - `GET /benchmark/submissions/{id}/report?flavor=all` → PDF binary response

2. **Frontend**: `just run-dashboard`
   - `/benchmark` zichtbaar in sidebar
   - Challenge aanmaken → CSV exporteerbaar
   - Pad A: model kiezen → stats + rapport
   - Pad B: CSV uploaden → stats + rapport
   - Alle drie report-smaken selecteerbaar

3. **Integratie**: `just run-api` + `just run-dashboard`
   - End-to-end flow met `size=small` dataset (Aurora)
