# Fortran Quirks And Cleanup Targets

Dit document houdt expliciet bij welke huidige BESTPRED Fortran-gedragingen we
nu nog reproduceren voor parity, maar later bewust willen verwijderen,
corrigeren of herontwerpen in de Python-implementatie.

Gebruik deze indeling:

- `Compat now`: Python bootst dit nu expres na om current Fortran output te
  matchen.
- `Desired Python`: hoe we dit later willen modelleren zonder Fortran-artefact.
- `Status`: `documented`, `ported-for-compat`, `needs-design`, of `resolved`.

## Werkwijze

Elke keer dat we een nieuwe Fortran-afwijking, stateful side effect,
EOF-artefact, verborgen fallback of outputbug ontdekken:

1. voeg hem hier toe;
2. noteer waar hij zit in de Fortran-code;
3. leg uit waarom het geen gewenste eindtoestand is;
4. noteer of Python hem al tijdelijk kopieert voor compatibility.

Dit document is dus geen bugtracker voor gewone porttaken, maar een register
van gedrag dat waarschijnlijk niet thuishoort in een schone Python-versie.

## Bekende Quirks

### 1. Source-11 `doprev` valt impliciet terug naar 140

- Fortran locatie:
  - `bestpred_main.f90` zet `doprev = 200` in de source-11 simulator.
  - `bestpred_fmt4.f90` leest daarna bytes `246:248` opnieuw uit Format-4.
- Huidig gedrag:
  - source 11 vult die bytes niet;
  - daardoor wordt `doprev` effectief `0`;
  - daarna valt het via `bestpred_fmt4.f90` terug naar `140`.
- Compat now:
  - Python volgt dit nu, omdat het anders de current oracle niet matcht.
- Waarom ongewenst:
  - de caller zet `200`, maar downstream wint een lege fixed-width field;
    dat is verborgen gedrag, geen expliciet model.
- Desired Python:
  - source-input normalisatie moet `previous_days_open` een eenduidige,
    expliciete waarde geven zonder impliciete overwrite uit lege bytes.
- Status: `ported-for-compat`

### 2. Source-11 herd SCS wordt stateful telkens opnieuw door 100 gedeeld

- Fortran locatie:
  - `bestpred_fmt4.f90` schaalt `herd305(:,4)` voor SCS.
- Huidig gedrag:
  - in source 11 wordt dezelfde caller-array tussen records hergebruikt;
  - row 1 gebruikt `3.08`, row 2 `0.0308`, row 3 `0.000308`, enzovoort.
- Compat now:
  - Python heeft dit source-11 compatibiliteitsgedrag expliciet ingebouwd.
- Waarom ongewenst:
  - dit is mutable caller-state, geen inhoudelijke SCS-logica.
- Desired Python:
  - herd means moeten immutabel per record worden opgebouwd en precies eenmaal
    naar interne units worden geconverteerd.
- Status: `ported-for-compat`

### 3. Source-10 schrijft eerst een header-only zero-test row

- Fortran locatie:
  - `bestpred_main.f90`, source-10 lees/flush-loop.
- Huidig gedrag:
  - per inputregel wordt eerst een record zonder segmenten verwerkt;
  - pas na EOF-driven flush komt de echte detail-row.
- Compat now:
  - Python genereert nu ook twee rows per source-10 inputregel.
- Waarom ongewenst:
  - dit is een main-loop artefact, geen echte business-output.
- Desired Python:
  - één inhoudelijke outputrow per invoerlactatie, tenzij een expliciete
    debug- of auditmodus anders vraagt.
- Status: `ported-for-compat`

### 4. Source-10 eerste `results_v2.dcr` row heeft corrupte identifiers

- Fortran locatie:
  - gevolg van dezelfde source-10 flushvolgorde in `bestpred_main.f90`.
- Huidig gedrag:
  - de eerste zero-test row in current Fortran heeft garbage/null identifiers.
- Compat now:
  - Python matcht voor source-10 numerics, maar houdt deze corrupte ID-output
    niet als gewenst publiek gedrag aan.
- Waarom ongewenst:
  - output-identiteit mag nooit afhangen van onvolledig geflushte headerstate.
- Desired Python:
  - alleen geldige, volledig gevormde records mogen geschreven worden.
- Status: `documented`

### 5. Source-15 gebruikt dezelfde twee-row quirk als source 10

- Fortran locatie:
  - `bestpred_main.f90`, source-15 branch rond het lezen van `format4.means`.
- Huidig gedrag:
  - eerst header-only zero-test row zonder means override;
  - daarna detail-row met herd-mean override.
- Compat now:
  - Python volgt dit precies.
- Waarom ongewenst:
  - herd means horen een recordeigenschap te zijn, geen late flush-mutatie.
- Desired Python:
  - parse één compleet record met directe means-binding.
- Status: `ported-for-compat`

### 6. Repo-root `format4.means` is leeg

- Fortran locatie:
  - source 15 verwacht een row per format4-record op unit 15.
- Huidig gedrag:
  - de meegeleverde rootfile is leeg;
  - current Fortran loopt dan op EOF tijdens de means-read.
- Compat now:
  - Python gebruikt een aparte fixturefile onder
    `python/tests/fixtures/source15_current/format4.means`.
- Waarom ongewenst:
  - checked-in voorbeelddata voor een ondersteunde source hoort runnable te
    zijn.
- Desired Python:
  - fixturedata en productie-invoer valideren op complete bronparen.
- Status: `documented`

### 7. Source-14 schrijft per bestand een extra EOF/zero-test row

- Fortran locatie:
  - `bestpred_main.f90`, source-14/24 loop na EOF van een bronbestand.
- Huidig gedrag:
  - na de laatste echte koe wordt nog een extra row met `DIM=0` geschreven.
- Compat now:
  - Python bewaart deze row expliciet via een compatibility-tag.
- Waarom ongewenst:
  - EOF is control flow en hoort geen extra business-record op te leveren.
- Desired Python:
  - file-einde beëindigt de stream zonder synthetische outputrow.
- Status: `ported-for-compat`

### 8. Source-24 laat in current Linux-run een lege `results_v2.dcr` achter

- Fortran locatie:
  - wrapper-flow in `bestpred_main.f90` voor source 24.
- Huidig gedrag:
  - `pcdart.bpo` wordt wel geschreven;
  - `results_v2.dcr` blijft in onze run leeg.
- Compat now:
  - Python source-24 tests gebruiken geconcateneerde source-14
    `results_v2.dcr`-oracles voor de numerieke wrapper-validatie.
- Waarom ongewenst:
  - een wrapper source hoort dezelfde interne output-contracten niet stilzwijgend
    kwijt te raken.
- Desired Python:
  - source 24 moet een pure list-wrapper zijn boven source 14, zonder
    afwijkende side effects of ontbrekende outputs.
- Status: `needs-design`

### 9. `results_v2.dcr` wordt soms boven de kernlaag geschreven met caller-state

- Fortran locatie:
  - `bestpred_main.f90` schrijft `results_v2.dcr`, niet alleen `bestpred()`.
- Huidig gedrag:
  - sommige outputkolommen komen uit caller-state in plaats van uit de intern
    gecorrigeerde solver-state.
  - voorbeeld: source-11 M/F/P herd-outputkolommen blijven synthetische caller
    herd means in plaats van de intern 3X-gecorrigeerde waarden.
- Compat now:
  - Python heeft hiervoor expliciete source-specifieke output-assembly.
- Waarom ongewenst:
  - outputcontract hangt nu af van waar in de stack je kijkt, niet van één
    eenduidige result-structuur.
- Desired Python:
  - één expliciete output-assembler per publiek formaat, gevoed vanuit een
    heldere interne resultmodel-laag.
- Status: `ported-for-compat`

### 10. DIM boven `maxlen` blijft in output staan terwijl intern geclamped wordt

- Fortran locatie:
  - `bestpred.f90` en caller-outputflow.
- Huidig gedrag:
  - record met output-DIM `366` wordt intern behandeld alsof `365` het maximum
    is, maar de geschreven output houdt `366`.
- Compat now:
  - Python doet hetzelfde om de oracle te matchen.
- Waarom ongewenst:
  - interne berekeningsscope en publiek gerapporteerde DIM horen niet stil van
    elkaar af te wijken.
- Desired Python:
  - expliciet veldonderscheid maken tussen `reported_dim` en `effective_dim`,
    of invoer vooraf normaliseren.
- Status: `ported-for-compat`

### 11. Source-26 milk-only pad gebruikt speciale missing-component guards

- Fortran locatie:
  - `bestpred.f90` outputguards voor component-DCR en SCS-missing pad.
- Huidig gedrag:
  - als componenttests ontbreken, gebruikt Fortran speciale fallbackregels
    zoals `DCRc = DCRvec(2)` en nuloutput/`NaN`-mix voor SCS.
- Compat now:
  - Python volgt deze guards nu.
- Waarom ongewenst:
  - deze regels zijn functioneel verdedigbaar, maar zitten nu verstopt als
    ad-hoc outputcondities in plaats van als expliciete missing-data policy.
- Desired Python:
  - missing-data gedrag moet in benoemde beleidslogica zitten, niet als
    verspreide historische guards.
- Status: `ported-for-compat`

### 12. Source-14 EOF-row `pcdart.bpo` gebruikt `INT()` op bijna-integer floats

- Fortran locatie:
  - `bestpred_main.f90`, source-14 Albert/PCDART writer voor `pcdart.bpo`.
- Huidig gedrag:
  - de extra EOF/zero-test row wordt ook 305 keer naar `pcdart.bpo`
    geschreven;
  - sommige waarden die inhoudelijk exact een herd mean lijken, worden eerst
    als floating-point solverwaarde opgebouwd en daarna met Fortran `INT()`
    afgekapt;
  - voorbeeld in `test241.txt`: fat wordt in `results_v2.dcr` als `900.`
    afgerond, maar in de EOF-row van `pcdart.bpo` als `899` geschreven.
- Compat now:
  - Python schrijft de inhoudelijke value (`900`) en de tests vergelijken deze
    EOF-artifact output numeriek tolerant.
- Waarom ongewenst:
  - dit combineert twee artefacten: een synthetische EOF-row en een zichtbare
    truncatie van een bijna-integer float.
- Desired Python:
  - geen EOF-row schrijven in de schone outputmodus; waar integer output nodig
    is, expliciet kiezen tussen afronden en afkappen.
- Status: `documented`

## Richting Voor De Python-Eindstaat

De beoogde Python-versie hoeft niet bug-for-bug compatibel te blijven zodra:

1. we current Fortran volledig kunnen reproduceren;
2. we de afwijking expliciet in dit document hebben vastgelegd;
3. we een test hebben die het huidige gedrag bewust markeert als legacy;
4. we een expliciete Python-beslissing nemen over het gewenste gedrag.

De volgorde is dus:

1. begrijpen;
2. reproduceren;
3. documenteren;
4. vervangen.
