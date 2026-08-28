# Bovi — overdracht en documentatiekaart

Deze map is het korte startpunt voor de Bovi-overdracht. Hij legt vast welke
repository vandaag de operationele bron is, welke repositories vooral
ontwerp-, experiment- of leermateriaal bevatten, en waar verdieping staat.

## Begin hier

1. Lees [Repositorykaart](repository-guide.md) voor de samenhang, eigenaarschap
   en actuele status van alle relevante repositories.
2. Lees de root-[README](../../README.md) en
   [CLAUDE.md](../../CLAUDE.md) van `bovi` voor de actuele monorepo, lokale
   commando's en architectuurregels.
3. Kies daarna de package- of app-documentatie die bij de overdracht past.

De beknopte kern is:

```text
                         bovi (actuele product- en integratierepo)
                                      |
      +-------------------------------+------------------------------+
      |                               |                              |
  gedeelde core                  model packages                  deployables
  packages/bovi-core             packages/models/*               apps/*
      |                               |                              |
  contracten, registry,       autoencoder, YOLO,             centrale API,
  config, storage             lactationcurve, BESTPRED        model apps, dashboard

Historische/ondersteunende repos: bovi-models-template, bovi-models-douwe,
bovi-models-tutorial, standalone bovi-core en lactation_curve_core.
```

## Bronnen en betrouwbaarheid

De data hieronder is de laatste inhoudelijke wijziging van de betreffende
documentatie of repository, gecontroleerd op 28 augustus 2026. Een recente datum
is geen garantie dat elke codevoorwaarde nog klopt; de monorepo en haar
`pyproject.toml`-bestanden zijn leidend voor het actuele gedrag.

| Bron | Rol bij overdracht | Behandel als |
| --- | --- | --- |
| [`bovi`](../../README.md) | Product, packages, API, modelapps en dashboard | **Actuele bron** |
| `../bovi-models-template` | Oorspronkelijk 3-lagenontwerp en projectindeling | Ontwerp- en templatebron |
| `../bovi-models-douwe` | Werkend voorbeeld van lactatie- en YOLO-experimenten | Historische experimentbron |
| `../bovi-models-tutorial` | Leerpad en notebooks voor het framework | Onboardingbron |
| `../bovi-core` | Vroege zelfstandige core | Historische voorloper |
| `../lactation_curve_core` | Vroege zelfstandige lactationcurve-package/API | Gemigreerde voorloper |

De repo's in de laatste vijf rijen zijn niet als verwijderd of onbruikbaar
bestempeld. Voor deze overdracht zijn ze alleen niet de bron voor operationele
versies, deployment of huidige afhankelijkheden.

## Actuele documentatie in `bovi`

| Onderwerp | Startpunt | Wanneer nodig |
| --- | --- | --- |
| Monorepo en lokaal werken | [root README](../../README.md), [CLAUDE.md](../../CLAUDE.md) | Altijd |
| Gedeeld ML-framework | `packages/bovi-core/` en de packageconfiguratie | Registry, config, storage of modelbasis |
| Klassieke curves en ICAR | [lactationcurve README](../../packages/models/lactationcurve/README.md) | Curve fitting, LCCs, 305-dagenopbrengst |
| BESTPRED-port | [BESTPRED-documentatie-index](../../packages/models/bestpred/docs/README.md) | Fortran-pariteit, FDD of beste-voorspelling |
| Dashboard | [dashboard README](../../apps/frontend/dashboard/README.md) | Lokale UI of API-proxy |
| API en modelapps | `apps/backend/api/` en `apps/backend/models/` | Contract, opslag en deployment |
| Besluiten/plannen uit 2026 | `docs/superpowers/{plans,specs}/` | Historische ontwerpcontext, niet als runtime-handleiding |

## Onderhoudsafspraak

- Houd deze map op overzichtsniveau; dupliceer geen package-API's of
  implementatiedetails.
- Leg een structurele verandering eerst bij de actuele package/app vast en
  pas daarna de repositorykaart aan als de relatie tussen repos verandert.
- Markeer oudere materiaal expliciet als *historisch* of *template*. Verwijder
  het niet alleen omdat het niet meer uitvoerbaar is: het beschrijft de reden
  achter de opsplitsing.
- Gebruik de huidige `bovi`-workspace voor Python 3.12, `uv`, `just sync` en
  `just test`; de oudere siblingrepos bevatten nog Python-3.11- en
  losse-installatie-instructies.
