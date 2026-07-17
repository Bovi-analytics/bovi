# BESTPRED Python Port Refactor Plan

Dit document is het voortgangs- en werkplan voor de gefaseerde port van de
Fortran BESTPRED code naar een losse Python package in deze repo.

Lees ook eerst:

- `docs/bestpred-ai-wiki.md`
- `docs/fortran-quirks.md`
- `python/docs/bovi_fdd_alignment.md`
- `AGENTS.md`

## Huidige Status

We zitten in fase 3, met fase 2 grotendeels klaar.

Fase 1 heeft de Python package, tooling, typed interfaces en test-oracle
neergezet. Fase 2 is gestart met traceerbaarheid van de Fortran-kern en eerste
Python-ports van `interpolate()`, `vary()` en de table/lookup-laag van
`covary()`. Daarna zijn `ymean()` en `adjust3X()` als losse helpers toegevoegd.
De eerste matrix-solve helper staat ook klaar.

Nog open voordat de port 100% werkend en gevalideerd is:

0. M/F/P yield drift verklaren en oplossen voordat er verder breed wordt
   geport.
   - Status: opgelost voor het eerste source-11 record.
   - Oorzaak: de Python source-11 simulatie gebruikte `previous_days_open=200`,
     omdat `bestpred_main.f90` rond regel 911 `doprev = 200` zet. In de echte
     Fortran-flow leest `bestpred_fmt4.f90` daarna `doprev` opnieuw uit bytes
     246-248 van het Format-4 record. Source 11 vult die bytes niet, waardoor
     `doprev` effectief `0` wordt en vervolgens door
     `bestpred_fmt4.f90:265` naar `140` valt.
   - Effect voor eerste source-11 record:
     - Voor fix: 305 milk Python `21752.67`, Fortran oracle `21900`, drift
       `-147.33 lb`.
     - Na fix: 305 milk Python `21899.76`, Fortran oracle `21900`, drift
       `-0.24 lb`.
     - Expanded milk: Python `21951.06`, Fortran oracle `21951`, drift
       `+0.06 lb`.
   - De tijdelijke Fortran debug-run met `DEBUGmsgs=2` in `/tmp` was voldoende:
     Fortran printte `doprev=140` en `agefac(1)=1.007882659...`; Python stond
     voor de fix op `agefac(1)=1.000986756...`.
1. `predict_records()` volledig implementeren voor source 11.
   - Huidig: partial source-11 rows uit Python MT M/F/P + ST SCS
     305/365/laclen/LTD, inclusief 305 reliability, DCR, persistency,
     persistency reliability, expanded yields, herd305 output en bumpiness.
   - Performance: deterministic standaardcurves, daily-SD matrices,
     305-varianties en persistency-varianties worden gecachet per
     parameterconfiguratie. De `mtrait=3` source-11 route berekent M/F/P nu
     lazy via MT en rekent ST M/F/P niet meer uit voordat die velden toch
     overschreven worden.
   - Nog open: MT/SCS integratie buiten `mtrait=3` en bumpiness-reductie.
2. De volledige `bestpred.f90` orchestration porten:
   - trait loops;
   - ST/MT routes;
   - testdagselectie;
   - deviations;
   - covariance targets voor 305/365/part/persistency.
3. DCR, yield reliability, persistency, persistency reliability en
   bumpiness/smoothness porten.
   - Huidig: single-trait DCR, `RELyld`, `PERSvec`, `RELpers`, `Yvec`,
     `herd305` en `bump` zijn geport voor source 11; M/F/P heeft nu ook een
     eerste `mtrait=3` MT-route.
   - Nog open: bumpiness/smoothness reductie.
4. End-to-end golden tests maken tegen current Fortran output.
5. Source 14/24 smoke/golden coverage toevoegen.
   - Status: gedaan voor parser, `results_v2.dcr` numerics en `pcdart.bpo`
     output.
   - Source 14 parser staat nu in `python/src/bestpred/io/source14.py` en de
     CLI kan source 14 draaien naar `results_v2.dcr`-compat output.
   - Source 24 file-list parser hergebruikt die source-14 parser per bestand.
   - De `pcdart.bpo` writer staat in `python/src/bestpred/io/pcdart.py` en is
     via de CLI beschikbaar met `--pcdart-output`.
   - Golden tests staan in `python/tests/test_source14.py` en
     `python/tests/test_source24.py`.
   - Open restant: de huidige source-24 Fortran-wrapper laat in onze Linux-run
     een lege `results_v2.dcr` achter, dus de source-24 numerieke fixture is
     voorlopig gebaseerd op geconcateneerde source-14 current outputs van
     `test241.txt` en `test242.txt`.
6. Package-data oplossen voor `aiplage.h`, `adjust.scs` en de geparsede
   Fortran curve-tabellen.
   - Status: afgerond voor de Python-port.
   - De runtime resources staan nu onder `python/src/bestpred/data/`.
   - `load_aiplage_data()`, `load_scs_adjustment_data()` en
     `load_regional_curve_tables()` gebruiken standaard `importlib.resources`
     en hebben alleen nog een expliciet pad nodig voor debug/probe fixtures.
   - `python/tests/test_package_data.py` bewaakt dat de default loaders niet
     meer afhankelijk zijn van repo-root bestanden.
7. Daarna pas de Bovi Python best-predict API naast deze authentieke port
   leggen en het migratiepad bepalen.
   - Status: gestart.
   - Analyse staat in `python/docs/bovi_fdd_alignment.md`.
   - Eerste adapterboundary staat in
     `python/src/bestpred/adapters/farm_data_definitions.py`: FDD `Cow`/`Herd`
     plus tijdelijke BESTPRED lactatie/testdag DTO's naar `Format4Record`.
   - Bovi `lactationcurve.characteristics.best_predict` blijft voor nu
     read-only geanalyseerd: dataframe-invoer, melk-only public path, vaste
     305-d curve/covariance assets.

Belangrijk uitgangspunt:

- De huidige Fortran/Linux output en de herstelde macOS binary output zijn
  byte-identiek.
- Die output is de source of truth voor de Python-port.
- `DCRexample.results.dcr` is legacy/manual reference en matcht de huidige
  source niet.
- Niet elk current-Fortran gedrag is gewenst eindgedrag. Bekende artefacten,
  verborgen fallbacks en outputbugs worden centraal bijgehouden in
  `docs/fortran-quirks.md`.

## Fase 1: Package, Tooling En Oracle

Status: afgerond.

Gedaan:

- Aparte Python package aangemaakt in `python/`.
- Package/distributienaam: `bestpred-py`.
- Importnaam: `bestpred`.
- `uv` project opgezet met `uv.lock`.
- Tooling toegevoegd:
  - `pytest`
  - `ruff`
  - `basedpyright`
- Directe lokale dependency toegevoegd op `farm-data-definitions`.
- Pydantic v2 domain models toegevoegd voor:
  - parameters
  - sources
  - traits
  - source-11 testplannen
  - Format 4 records
  - DCR result rows
- Parser toegevoegd voor `bestpred.par`.
- Parser toegevoegd voor `DCRexample.txt`.
- Parser en compatibility writer toegevoegd voor `results_v2.dcr`.
- Source-11 simulatie toegevoegd voor de test-day segmenten uit
  `bestpred_main.f90`.
- CLI toegevoegd met:
  - `--source`
  - `--input`
  - `--par`
  - `--output`
  - `--oracle-output`
- Current Fortran golden fixtures toegevoegd onder
  `python/tests/fixtures/source11_current/`.
- Legacy manual expected output toegevoegd onder
  `python/tests/fixtures/legacy_manual_expected/`.
- Bovi/Farm Data Definitions analyse toegevoegd in
  `python/docs/bovi_fdd_alignment.md`.
- Test toegevoegd voor `farm-data-definitions` breed mapping.

Validatie:

```bash
cd python
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

Laatste bekende status:

```text
pytest: 52 passed
ruff check: passed
ruff format --check: passed
basedpyright: 0 errors
```

## Fase 2: Fortran Traceerbaarheid En Kernel-Skeleton

Status: gestart.

Doel: de Fortran-kern opsplitsen in Python-functies met testbare boundaries,
zonder meteen alles tegelijk te porten.

Taken:

1. Maak een Fortran trace table.
   - Breng `bestpred.f90` onder in kleine port-units:
     - `interpolate`
     - `ymean`
     - `adjust3X`
     - `vary`
     - `covary`
     - matrix solve
     - DCR/reliability
     - persistency
     - output assembly
   - Documenteer per unit:
     - Fortran bestandsnaam en regelnummers
     - input arrays/scalars
     - output arrays/scalars
     - shape conventies
     - dependencies
   - Status: `interpolate()` trace staat in
     `docs/fortran-kernel-trace.md`.
   - Status: `vary()` en `covary()` trace staan ook in
     `docs/fortran-kernel-trace.md`.

2. Voeg kernel modules toe.
   - Aanbevolen structuur:
     - `python/src/bestpred/core/curves.py`
     - `python/src/bestpred/core/covariance.py`
     - `python/src/bestpred/core/adjustments.py`
     - `python/src/bestpred/core/prediction.py`
     - `python/src/bestpred/core/reliability.py`
   - Houd `kernel.py` als orchestration boundary.
   - Status: `curves.py` is toegevoegd voor lineair, Wood en
     Morant/Gnanasakthy.
   - Status: `covariance.py` is toegevoegd voor observation covariance en
     lactatie-covariance tabellen.

3. Maak debug fixtures voor intermediate values.
   - Gebruik kleine source-11 records.
   - Voeg tijdelijk Fortran debug-output toe vanuit geisoleerde `/tmp` runs.
   - Commit alleen stabiele fixturebestanden onder `python/tests/fixtures/`.
   - Status: open.

4. Verwijder geen Fortran-code.
   - De Fortran blijft de oracle totdat de Python-kernel volledig matcht.

## Fase 3: Curves En Adjustments

Status: deels gestart voor curves.

Doel: alle standaardcurves, age/season/PDO correcties, SCS correcties en 3X
milking correcties in Python beschikbaar maken.

Taken:

1. Port `interpolate()` uit `bestpred.f90`.
   - Gedaan voor:
     - `L`: lineaire interpolatie.
     - `W`: Wood curves voor melk, vet en eiwit.
     - `G`: Morant/Gnanasakthy curves voor SCS.
     - `R/C/T`: regio/seizoen Wood curves.
     - `S/D/U`: regio/seizoen SCS curves.
   - Regionale/seizoensmethodes worden uit de originele `data` blocks in
     `bestpred.f90` geparsed om handmatige overtypfouten te vermijden.
2. Port `adjust3X()` uit `bestpred.f90`.
   - Gedaan voor de deterministic factorberekening in
     `python/src/bestpred/core/adjustments.py`.
3. Port `aiplage.c` naar pure Python.
   - Gedaan in `python/src/bestpred/core/age.py`.
   - Coefficients worden uit de originele `aiplage.h` geparsed.
   - Tests staan in `python/tests/test_age.py`.
4. Port `adjscs.c`, `ageadjs.c` en `adjust.scs` naar pure Python.
   - Gedaan voor de actieve build-route: `ageadjs.c::adjscs()` met additive
     SCS adjustment en `adjust.scs` parsing in
     `python/src/bestpred/core/scs.py`.
   - Tests staan in `python/tests/test_scs.py` en zijn gepind aan C-probe
     output van de actieve `ageadjs.c` routine.
5. Voeg unit tests toe die Python-output vergelijken met Fortran/C output voor
   representatieve inputs.

Acceptatie:

- Curve arrays matchen Fortran binnen afgesproken tolerantie.
- Age factors en SCS factors matchen Fortran/C output.
- Geen runtime-afhankelijkheid op C of Fortran.

## Fase 4: Covariance En Prediction Kernel

Status: open.

Doel: de kernberekening van BESTPRED in NumPy/SciPy implementeren.

Taken:

1. Port `vary()`.
2. Port `covary()`.
3. Bouw observation covariance matrix.
4. Bouw lactation/testday covariance vectoren.
5. Gebruik `scipy.linalg.solve` of factorization in plaats van expliciete
   inverse, tenzij exacte Fortran-debug nodig is.
6. Port `ymean()` en deviations.
7. Maak `predict_records()` functioneel voor source 11.
   - Gestart: `predict_records(records, parameters)` geeft nu partial
     `results_v2.dcr`-vormige rows terug op basis van de Python ST
     305/365/laclen/LTD path voor melk, vet, eiwit en SCS, inclusief ST 305
     reliability, DCR, persistency, persistency reliability, expanded yields,
     herd305 output en bumpiness.

Acceptatie:

- Python source-11 output matcht current Fortran numeriek veld-voor-veld.
- Waar formatting deterministisch is, matcht de compatibility output
  byte-identiek met `python/tests/fixtures/source11_current/results_v2.dcr`.

## Fase 5: DCR, Reliability, Persistency En Outputs

Status: deels gestart.

Doel: alle Fortran-outputvelden correct vullen, niet alleen yield prediction.

Taken:

1. Port DCR berekening.
   - Gedaan voor source-11 single-trait en gestart voor M/F/P `mtrait=3`.
2. Port yield reliability.
   - Gedaan voor source-11 single-trait 305-d `RELyld` en gestart voor M/F/P
     `mtrait=3`.
3. Port persistency berekening.
   - Gedaan voor source-11 single-trait `PERSvec`.
   - De Python-port volgt `bestpred.f90:1247-1249` en
     `bestpred.f90:1358-1363`: `dcov = covary(..., stat=2) *
     varfac**2 / agefac`, daarna `dcov * inv(var) * dev`.
4. Port persistency reliability.
   - Gedaan voor source-11 single-trait `RELpers`.
   - De Python-port volgt `bestpred.f90:1415-1416`: `RELpers =
     dvari(1,1) / varfac**2`.
5. Port bumpiness/smoothness metric.
   - Gedaan voor source-11 single-trait outputvelden.
   - M/F/P `mtrait=3` output blijft op `0.0`, conform de huidige Fortran-route
     waar bumpiness alleen via de ST SCS-route in de fixture zichtbaar wordt.
   - Nog open: eventuele reliability-reductie als `nbump > 0` later actief
     gemaakt wordt.
6. Maak structured production output naast de compatibility writer.

Acceptatie:

- `DcrResultRow` kan volledig uit Python-kernelresultaten worden opgebouwd.
- CLI zonder `--oracle-output` produceert source-11 en source-10 output die
  de current Fortran oracle numeriek matcht.
  - Status: gedeeltelijk. Er is nu een golden test voor alle numerieke
    source-11 velden `0..42` over alle 43 source-11 rows. De eerder open M/F/P
    yield drift, SCS expanded-output drift en DIM 366 gap zijn opgelost.
    Source 10 en 15 `format4.dat`/`format4.means` zijn nu ook geport met
    golden coverage voor de checked-in fixtures.

## Fase 6: Extra Sources En Bovi Integratie

Status: gestart.

Doel: na source-11 parity uitbreiden naar echte inputbronnen en later Bovi.

Taken:

1. Implementeer source 10 parser voor `format4.dat`.
   - Status: gedaan.
   - `python/src/bestpred/io/source10.py` parseert de checked-in Format-4
     fixture en bootst de huidige `bestpred_main.f90` source-10 flow na:
     eerst een header-only zero-test row, daarna de geflushte detail-row.
   - De Python-kernel hergebruikt dezelfde partial-output route als source 11,
     maar zonder de source-11 herd-column override.
   - Golden tests staan in `python/tests/test_source10.py`.
2. Implementeer source 15 met `format4.means`.
   - Status: gedaan.
   - `python/src/bestpred/io/source15.py` parseert vaste-width means rows en
     hergebruikt de source-10 Format-4 parser.
   - De Python-port volgt de huidige `bestpred_main.f90` source-15 flow:
     eerst een header-only zero-test row zonder means override, daarna de
     geflushte detail-row met herd-mean override.
   - Bij cow/fresh mismatch volgt Python de huidige Fortran-guard en zet de
     detail-row means op nul.
   - De repo-root `format4.means` is leeg; de golden fixture gebruikt daarom
     een expliciet testbestand onder
     `python/tests/fixtures/source15_current/format4.means`.
3. Implementeer source 14/24 voor PCDART-achtige input.
   - Status: deels gedaan.
   - `python/src/bestpred/io/source14.py` parseert herd headers en detailregels
     fixed-width en zet segmenten om naar `Format4Record`.
   - De Python-port bewaart ook de huidige source-14 EOF/zero-test artifactrow
     per bestand via `compatibility_tag='source14_eof_zero'`.
   - Source 24 leest `pcdart_files.txt` en flatten't de onderliggende
     source-14 bestanden.
   - `python/src/bestpred/io/pcdart.py` schrijft nu de echte Albert/PCDART
     daily-output (`pcdart.bpo`) voor source 14 en source 24.
4. Voeg golden fixtures toe per source.
   - Status: gedaan voor source 10, 11, 14, 15 en een numerieke source-24
     wrapper-fixture.
5. Ontwerp adapters naar `farm-data-definitions` lactation/test-day concepten.
   - Status: gestart.
   - `BestpredLactationInput`, `BestpredTestDayInput`,
     `BestpredHerdMeansInput` en `format4_record_from_fdd()` vormen nu de
     tijdelijke adapterboundary.
   - Tests staan in `python/tests/test_farm_data_definitions_adapter.py`.
6. Maak voorstel voor ontbrekende `Lactation` en `TestDay` modellen in
   `farm-data-definitions`.
   - Status: concept gedocumenteerd in `python/docs/bovi_fdd_alignment.md`.
7. Bereid integratiepad voor naar `../bovi/packages/models/lactationcurve`.
   - Status: analyse gestart; nog geen wijzigingen in `../bovi`.
   - Belangrijk verschil: Bovi's huidige best-predict API is dataframe/melk-only,
     terwijl BESTPRED multi-trait records met component sample flags, herd
     means, DCR, reliability en persistency nodig heeft.
   - Er is nu een lokale vergelijkings-CLI:
     `cd python && uv run bestpred compare-bovi --source 11 --input tests/fixtures/source11_current/DCRexample.txt --par tests/fixtures/source11_current/bestpred.par --limit 10`.
     Die stuurt dezelfde source-input naar `bestpred-py` en Bovi's huidige
     dataframe `best_predict_method` en toont een tabel plus summary in kg.

Niet doen in deze fase zonder expliciete opdracht:

- `../bovi` wijzigen.
- `../farm-data-definitions` wijzigen.
- Database-tabellen of SQLModel persistence in `bestpred-py` toevoegen.

## Eerstvolgende Taken

Pak hier verder op:

1. Los de M/F/P yield drift op voordat verdere port-breedte wordt toegevoegd.
   - Status: gedaan.
   - Oorzaak: source-11 `doprev` valt in de echte Format-4 flow terug naar
     `140`, niet `200`.
   - Regressie: eerste source-11 record M/F/P yields matchen nu binnen
     Fortran-outputrounding.
2. Traceer en port de multi-trait prediction loop rond `bestpred.f90:881-1100`.
   - Status trace: gedaan in `docs/fortran-kernel-trace.md`.
   - Status port: gestart in
     `python/src/bestpred/core/kernel.py::predict_source11_mfp_multi_trait_debug`.
3. Los SCS expanded output (`Yvec(1,4)`, result index 34) op.
   - Status: gedaan.
   - Oorzaak: Fortran gebruikt bij `mtrait=3` voor SCS niet de expanded-yield
     formule, maar de single-trait fallback `PERSvec(4) / RELpers(4)`, daarna
     gedeeld door `305` (`bestpred.f90:1513-1528`).
   - Effect eerste source-11 record: Python schrijft nu
     `-0.588683854 / 0.724951811 / 305 = -0.002662399`, wat in het
     `results_v2.dcr` formaat als `-0.00` verschijnt en de Fortran oracle
     matcht.
4. Traceer de single-trait prediction loop rond `bestpred.f90:1160-1370`.
   - Status: gedaan in `docs/fortran-kernel-trace.md`.
5. Voeg `python/src/bestpred/core/prediction.py` toe.
   - Status: gedaan met `solve_prediction_system()`.
6. Bouw een minimale source-11 prediction path voor één record en één trait.
   - Status: eerste debugpad gedaan voor source-11 ST milk 305 in
     `python/src/bestpred/core/kernel.py::predict_source11_milk_305_debug`.
   - Status: de M/F/P age-factor uit `aiplage.c` is nu geport en de
     debug-kernel gebruikt die standaard.
7. Voeg intermediate fixtures toe voor:
   - testdagdeviaties
   - observation covariance matrix
   - covariance vector naar 305-d yield
   - Status: nog geen file fixtures, wel runtime debug state en tests in
     `python/tests/test_kernel.py`.
8. Sluit daarna `predict_records()` aan op de eerste echte Python-kernelstap.
   - Status: gedaan voor source-11 partial output met MT M/F/P en ST SCS.
9. Breid de golden parity uit van het eerste source-11 record naar alle
   source-11 fixture records en documenteer per veld welke indexes nog niet
   exact/parity-gevalideerd zijn.
   - Status: gestart.
   - Performance-optimalisatie gedaan: cached curves/variance/persistency en
     lazy ST/MT orchestration.
   - Golden test dekt nu alle 43 source-11 rows.
   - All-record parity binnen output-rounding is vastgelegd voor alle
     numerieke result indexes `0..42` over alle 43 source-11 rows.
   - De voormalige out-of-range row 25 met DIM `366` is opgelost: Python
     clamp't intern naar `maxlen=365`, sluit de DIM 366-testdag uit en behoudt
     de originele output-DIM.
10. Eerstvolgende open taak: onderzoek de grootste resterende all-record
    drifts, te beginnen bij row 26 component/SCS null-output en row 2 yield /
    SCS reliability-bumpiness drift.
    - Row 26 component/SCS null-output: opgelost.
      - Bronrecord is `Milk-only`: alle testdagen hebben `times_sampled=0`;
        daardoor zijn fat/protein/SCS testdagwaarden nul behalve de MT
        componentvoorspelling uit melk.
      - Fortran-regel `bestpred.f90:1505`: als `ntests(3) == 0`, dan wordt
        `DCRc = DCRvec(2)` in plaats van het gemiddelde van fat/protein DCR.
        Python volgt dit nu via `_component_dcr_output()`.
      - Fortran ST SCS-loop (`bestpred.f90:1164-1316`) krijgt voor row 26
        `size == 0`; SCS DCR/yields/persistency/reliability/bump blijven nul,
        terwijl `Yvec(1,4)` als `NaN` in de fixture blijft. Python volgt dit
        nu via `_fill_missing_scs_output()`.
      - Regressietest:
        `test_source11_milk_only_record_uses_fortran_missing_component_guards`.
    - Row 2 yield / SCS drift: opgelost.
      - Row 2 heeft 10 milk observations maar alleen 5 sampled
        fat/protein/SCS observations; de andere 5 zijn milk-only.
      - M/F/P 305/365/laclen/expanded yields liggen nu binnen
        outputrounding. De grote drift kwam niet uit 3X of herd-output, maar
        uit de MT observation covariance voor melk-LER records.
      - Fortran `adjust3X()` trace (`bestpred.f90:2169-2296`) geeft dezelfde
        globale factoren die Python gebruikt (`milk=0.87719`, `fat=0.90909`,
        `protein=0.90090` voor row 2).
      - Intermediate Fortran trace in `/tmp/bestpred-row2-trace` verklaarde de
        herd-outputkolommen: binnen `bestpred()` is `herd305(1,:)` wel
        3X-gecorrigeerd (`17543.86/636.36/540.54` voor row 2), maar
        `results_v2.dcr` wordt in `bestpred_main.f90:783-786` vanuit de caller
        geschreven. Voor source 11 blijven de M/F/P herd-outputkolommen daarom
        de synthetische source-11 herd means `20000/700/600`. Python volgt dit
        nu via een expliciete source-11 output-assembly stap. Dit is
        fixture/caller-compatibiliteit, geen generieke BESTPRED-kernregel.
      - Dezelfde trace liet zien dat Fortran source 11 de SCS herd mean
        stateful muteert: `bestpred_fmt4.f90:315` deelt `herd305(:,4)` door
        `100`, en source 11 hergebruikt die caller-array tussen records. Row 1
        gebruikt daardoor `3.08`, row 2 `0.0308`, row 3 `0.000308`, enzovoort.
        Python reproduceert dit nu in de source-11 simulator voor oracle
        parity. Ook dit is source-11/demo-inputcompatibiliteit, geen generieke
        modelregel.
      - Beoordeling: dit is zeer waarschijnlijk een bug of ongewenst side
        effect in de Fortran source-11 demo-flow. Het moet niet naar de
        productie-kernel of naar echte sources worden gepromoveerd. Houd dit
        geisoleerd in de source-11 simulator zolang de current Fortran
        source-11 output de oracle is.
      - Row 2 ST SCS drift: opgelost.
        - Intermediate Fortran trace liet zien dat Python en Fortran dezelfde
          SCS deviations, `cov305` en persistency-covariances hadden, maar dat
          de observation covariance diagonalen te laag waren in Python.
        - Oorzaak: `_segment_observation_covariance()` zette bij LER
          (`MRD > 1`) ook voor SCS `sample = Xmilk`. Fortran doet bij
          `bestpred.f90:1172-1173` alleen `weigh = Xmilk`; `sample` blijft voor
          fat/protein/SCS gelijk aan de sampling frequency. Daardoor miste
          Python de AM/PM variance-inflatie uit `vary()`:
          `.3 * (Xmilk / max(sample1, sample2) - 1)`.
        - Voor row 2 SCS hoort de diagonal multiplier `1.15` te zijn
          (`Xmilk=3`, `sample=2`). Na de fix matchen DCR, SCS yields,
          `PERSvec(4)`, `RELpers(4)` en `bump(4)` binnen
          Fortran-outputrounding over alle geporteerde rows.
      - Row 2 MT M/F/P drift: opgelost.
        - Intermediate Fortran trace in `/tmp/bestpred-mfp-trace` liet zien
          dat Fortran voor melk-LER records (`MRD > 1`) `q(5)=weigh=Xmilk`
          gebruikt. Python gaf in de MT-route nog `times_weighed=2` door,
          waardoor de melk observation-covariance diagonalen te hoog waren en
          de M/F/P solve de yields omlaag trok.
        - Fix: `_samples_for_trait()` geeft voor melk met `ler_days > 1`
          `times_milked` terug, conform `bestpred.f90:892-923`.
        - Daarna bleven alleen partial velden 15-17 over. Oorzaak:
          Fortran berekent partial records via `covsum`, een expliciete som
          over `vary(actual_observation, pseudo_lactation_day)` voor dag
          `1..length` (`bestpred.f90:992-997`), niet via de `covary(...,
          length)` lookup. Python heeft nu een aparte
          `_covariance_to_partial_for_observation()` route.
        - Effect row 2 partial na fix:
          Python `(18791.39, 684.35, 599.51)` tegenover oracle
          `(18791, 684, 600)`, dus binnen outputrounding.
      - DIM 366 gap: opgelost.
        - Fortran clamp't `length > maxlen` vroeg in `bestpred.f90` naar
          `maxlen`, maar de resultrow behoudt de originele DIM uit source 11.
        - De testdag op DIM 366 wordt uitgesloten uit de MT/ST loops, waardoor
          `size == 0` en DCR/reliability/persistency nul blijven.
        - MT expanded yield voor M/F/P wordt `NaN`, omdat Fortran bij
          `regrel == 0` geen guard heeft in de `mtrait > 1` expanded-yield
          formule. SCS volgt het missing-SCS pad omdat er geen bruikbare
          testdag binnen `maxlen` is.
      - Resterende drift: geen numerieke field drift meer voor source 11.

## Werkregels

- Draai Fortran experimenten in `/tmp`, niet direct in de repo.
- Gebruik current Fortran output als oracle.
- Gebruik `DCRexample.results.dcr` alleen om legacy drift te documenteren.
- Houd de Python package databasevrij.
- Houd publieke modellen Pydantic v2.
- Houd numerieke arrays intern in NumPy.
- Voeg tests toe per port-unit voordat de volgende Fortran-subroutine wordt
  geport.
