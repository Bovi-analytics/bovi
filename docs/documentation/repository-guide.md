# Repositorykaart: structuur, doel en overdracht

## Het model in één oogopslag

De oorspronkelijke documentatie in `bovi-models-template` beschrijft een
3-lagenmodel: een generieke core, herbruikbare modelvoorbeelden en afzonderlijke
experimenten. Dat ontwerp is nog steeds een goed denkkader. De actuele
implementatie heeft de eerste twee lagen echter in de `bovi`-monorepo
geconsolideerd en voegt een expliciete deploymentlaag toe.

```text
Experimenten en leerwerk                         Product en operatie
--------------------------                       --------------------
template / Douwe / tutorial  ---- inzichten --->  bovi monorepo
standalone core / curve      ---- migratie --->       |
                                                     +-- bovi-core
                                                     +-- model packages
                                                     +-- centrale API
                                                     +-- Azure model apps
                                                     +-- dashboard
```

De handoff-werkregel is daarom eenvoudig: gebruik de siblingrepos om het
*waarom*, de voorbeelden en de evolutie te begrijpen; gebruik `bovi` om te
bepalen wat nu gebouwd, getest, uitgerold of aangepast wordt.

## 1. `bovi`: de actuele productrepo

`bovi` is de monorepo voor het werkende platform. Hij gebruikt Python 3.12 en
beheert zijn Python-workspace met `uv`; de root-commando's zijn `just sync`,
`just test`, `just lint` en de verschillende `just run-*`-commando's.

| Onderdeel | Locatie | Verantwoordelijkheid |
| --- | --- | --- |
| Framework | `packages/bovi-core/` | Kleine, gedeelde abstractions: config, opslag, basismodellen, registry en utilities. Geen zware model-frameworks in deze laag. |
| Modelleerpackages | `packages/models/` | `lactation-autoencoder` (TensorFlow), `bovi-yolo` (objectdetectie), `lactationcurve` (klassieke curves/ICAR) en `bestpred` (Python-port). |
| Centrale API | `apps/backend/api/` | Eén publiek contract, authenticatie, SQLite-persistentie, ingestie en proxy/coördinatie. |
| Modelapps | `apps/backend/models/` | Afzonderlijk te deployen Azure Function Apps voor de autoencoder en klassieke curves. |
| Dashboard | `apps/frontend/dashboard/` | Next.js-interface; browsercode gaat via de lokale `/api/bovi`-proxy naar de centrale API. |
| Infrastructuur | `apps/infrastructure/` en `packages/infrastructure/pulumi/` | Deployment- en cloudconfiguratie. |

De operationele datastroom is:

```text
Dashboard -> dashboardproxy (/api/bovi) -> centrale API
          -> SQLite/opslag en interne model Function Apps
```

De dashboardregel is bewust hard: het dashboard spreekt nooit rechtstreeks
met een modelapp. De centrale API is de integratiegrens.

### Model-extensies in de monorepo

`bovi-core` houdt de plugin-registries generiek. Concrete modelpackages
publiceren model- en predictor-entry points (`bovi.models` en
`bovi.predictors`); de registry ontdekt ze lazy via package metadata. Dat is
de actuele vorm van het oude zelfregistratiepatroon. Een nieuw model hoort dus
in een eigen modelpackage, registreert via de standaardregistry/entry-point-
conventie en trekt zijn ML-afhankelijkheden niet de core in.

## 2. `bovi-models-template`: de ontwerpbron

Dit is de beste plek om de intentie achter de indeling te leren kennen. De
mensgerichte documentatie staat in `misc/users/`:

| Thema | Startdocument |
| --- | --- |
| Navigatie | `misc/users/overview.md` |
| 3-lagenarchitectuur | `architecture/project_overview.md` |
| Mappen en verantwoordelijkheden | `architecture/project_structure.md` |
| Registry en uitbreidbaarheid | `architecture/registry_system.md` |
| Samenwerken | `workflows/collaboration.md`, `team_workflow.md`, `git_workflow.md` |
| Lactatie-experiment | `experiments/lactation_pipeline.md` |

De template laat een zelfstandig experimentrepo zien met `src/models/`,
`data/experiments/`, `notebooks/`, `tests/` en `misc/`. Het doel is: start met
een duidelijke projectstructuur, houd domeinspecifieke modellen en data daar,
en breng alleen breed herbruikbare infrastructuur naar de core.

**Belangrijke actualiteitsnoot.** De architectuur- en experimentdocumenten
zijn op 8 mei 2026 bijgewerkt, maar de root-README is van 15 januari 2026 en
de packageconfiguratie van 17 december 2025. Die configuratie vereist Python
3.11 en koppelt editable aan de zelfstandige `../bovi-core`. Ook veel
voorbeelden gaan uit van handmatige imports voor registry-discovery en noemen
`project_template_v2` of een los `bovi-models`-pakket. Behoud het ontwerp,
maar volg voor commando's, packagepaden en discovery de monorepo.

## 3. `bovi-models-douwe`: het experimentvoorbeeld

Dit repo is een concrete afgeleide van de template. Het documenteert vooral
hoe een lactatie-autoencoder en YOLO-model in een zelfstandig
experimentrepo werden georganiseerd:

- `src/models/lactation/` en `src/models/yolo/` bevatten domeinimplementaties;
- `data/experiments/` bevat configuraties, input en modelversies;
- `notebooks/experiments/` bevat uitvoerbare verkenningen;
- `misc/users/` bevat architectuur, pipeline-uitleg en workflows.

Gebruik het als verhalend en technisch voorbeeld van de vroegere
lactatiepipeline, niet als deploymentbron. Het project gebruikt nog de
zelfstandige, editable `../bovi-core` en Python 3.11. De laatste commit en de
gebruikersdocumentatie zijn van mei 2026, dus de domeinkennis is waardevol,
maar interfaces kunnen afwijken van `bovi`.

## 4. `bovi-models-tutorial`: de onboardingrepo

Deze repo leert de concepten stapsgewijs via notebooks: Python-basics, config,
registry, databronnen, transforms, dataloaders, modellen, lactatie end-to-end
en vervolgens Databricks/Unity Catalog. Begin bij de root-README en de
genummerde notebooks als iemand nieuw is in het framework.

Het tutorialrepo is al op Python 3.12 en haalt `bovi-core`, `bovi-yolo` en
`lactation-autoencoder` als subdirectories uit de `bovi`-Git-repository. Dat
maakt het de beste leerbron van de siblingrepos, maar niet de plek voor
productcode of releases.

## 5. Standalone voorlopers

| Repo | Wat het bewaart | Huidige behandeling |
| --- | --- | --- |
| `bovi-core` | Vroege core met registry, base classes en utilities | Historische voorloper. README en `pyproject.toml` zijn van 8 oktober 2025 en vragen Python 3.11; de actuele core staat in `bovi/packages/bovi-core`. |
| `lactation_curve_core` | Oorspronkelijke lactationcurve package plus curve-fitting API | Gemigreerde voorloper. De standalone repo heeft in mei 2026 een releasehistorie; de actuele package in `bovi` is `lactationcurve` 1.1.6 en documenteert ook ISLC en BESTPRED-methoden. |

Bewaar deze repos voor herkomst, publicatiegeschiedenis, auteurschap en
vergelijking. Wijzigingen aan de huidige productfunctionaliteit horen in de
monorepo tenzij expliciet anders afgesproken.

## 6. Een compacte leesroute voor een overdracht

1. Deze repositorykaart en de root-README van `bovi`.
2. `bovi-models-template/misc/users/overview.md` en
   `architecture/project_overview.md` voor het oorspronkelijke ontwerp.
3. De actuele package/app die bij iemands taak hoort.
4. Alleen daarna `bovi-models-douwe`, tutorialnotebooks of standalone repos
   voor een experiment, een ontwerpbeslissing of historische pariteit.

Zo blijft de overdracht leesbaar: één actuele productbron, met oude repos als
begrijpelijke achtergrond in plaats van concurrerende handleidingen.
