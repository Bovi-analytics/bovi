# Onderzoeksnotities voor de overdracht

Dit is een compact, controleerbaar spoor van de inventarisatie die aan de
overdrachtsdocumentatie ten grondslag ligt. Het is geen extra handleiding.

## Scope

Geïnventariseerd: `bovi`, `bovi-core`, `bovi-models-douwe`,
`bovi-models-template`, `bovi-models-tutorial` en `lactation_curve_core`.
Samen bevatten zij 109 versiebeheerde Markdown-bestanden; 59 daarvan zijn
primaire inhoudelijke documentatie en 50 zijn agentinstructies, checklists,
scratch-notities of toolingdocumentatie.

## Vastgestelde feiten

- `bovi` is de actuele Python-3.12-monorepo. Hij bevat de core, vier
  modelpackages, centrale API, twee model-Function-Apps, dashboard en
  infrastructuur.
- De actuele core blijft lichtgewicht; TensorFlow, PyTorch en Ultralytics
  horen bij modelpackages en niet bij `packages/bovi-core`.
- De dashboardgrens loopt via de centrale API. De browser gebruikt de
  dashboardproxy `/api/bovi`, niet een rechtstreeks modelapp-endpoint.
- De huidige registry ontdekt modellen en predictors via de
  `bovi.models`- en `bovi.predictors`-entry-pointgroepen. Dat vervangt de
  oude aanname dat een consument ieder modelpakket handmatig moet importeren.
- `bovi-models-template` bewaart de nuttige 3-lagenintentie: generieke core,
  domeinspecifieke modellen/experimenten, en heldere projectstructuur.

## Actualiteitscontrole

| Bron | Laatste relevante documentatie | Betekenis |
| --- | --- | --- |
| `bovi` root README | 2026-07-17 | Actuele operationele ingang |
| `bovi` lactationcurve README | 2026-06-03 | Actuele package-informatie, versie 1.1.6 |
| `bovi` BESTPRED-index | 2026-07-17 | Expliciet onderscheid tussen actief en historisch materiaal |
| `bovi-models-template` user docs | 2026-05-08 | Goede ontwerpbron |
| `bovi-models-template` README/config | 2026-01-15 / 2025-12-17 | Python 3.11 en los `../bovi-core`: niet als actuele setup gebruiken |
| standalone `bovi-core` README/config | 2025-10-08 | Historische voorloper, Python 3.11 |
| `bovi-models-douwe` docs | 2026-05-08 | Nuttig experimentverhaal, oude editable core-koppeling |
| `bovi-models-tutorial` | 2026-05-08 | Python-3.12-onboarding die al naar `bovi` verwijst |
| `lactation_curve_core` | 2026-05-13 | Voorganger van de gemigreerde monorepo-package |

## Verouderde claims die niet zijn overgenomen als instructie

- `project_template_v2`, een los `bovi-models`-referentiepakket en
  placeholders zoals `yourorg`/`YourName`.
- Python 3.11, losse `uv pip install -e`-stappen en een directe editable
  afhankelijkheid op de standalone core.
- Handmatige imports als verplichte registry-discovery. In de monorepo zijn
  package-entry-points de standaard; handmatige registratie blijft een
  implementatiedetail waar nodig.
- Een template of zelfstandig experimentrepo als deployment- of releasebron.

## Nog te doen bij een volgende uitbreidingsronde

- Alleen wanneer de handoff meer diepgang nodig heeft: per model een korte
  operationele runbook toevoegen, beginnend bij de centrale API-contracten.
- Voor BESTPRED en lactationcurve expliciet afspreken welke historische
  numerieke pariteitstest of publicatiedocumentatie de nieuwe eigenaar moet
  kennen.
