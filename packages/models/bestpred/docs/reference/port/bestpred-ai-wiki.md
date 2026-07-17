# BESTPRED AI Wiki

Dit document is de herbruikbare projectcontext voor deze repo. Gebruik dit als eerste referentiepunt in nieuwe threads voordat je code wijzigt, extra analyses doet of aan de Python-port begint.

Voor de actuele Python-port voortgang, fase-indeling en eerstvolgende taken:
zie `docs/python-port-refactor-plan.md`.

Voor de trace van Fortran-kernel naar Python-port-units:
zie `docs/fortran-kernel-trace.md`.

Voor bekende legacy-artefacten en Fortran-quirks die we nu soms nog bewust
reproduceren voor parity, maar later uit Python willen verwijderen:
zie `docs/fortran-quirks.md`.

## Projectdoel

We willen de originele BESTPRED implementatie uit deze repo begrijpen, reproduceerbaar draaien op Linux, en daarna gefaseerd naar Python omzetten. De reden is dat de sibling repo `../bovi` al een Python best-predict methode heeft, maar die is veel smaller dan de authentieke Fortran implementatie.

Het einddoel is niet simpelweg "Fortran vertalen naar Python", maar:

1. De huidige Fortran-code betrouwbaar kunnen bouwen en draaien.
2. Golden outputs vastleggen op basis van reproduceerbare inputs.
3. De rekenlogica in kleine, testbare Python modules porten.
4. De Bovi Python best-predict methode verbeteren met de authentieke BESTPRED logica.

Voorlopig blijft al het werk in deze `bestpred` repo, los van `../bovi`.

## Belangrijkste conclusie tot nu toe

De huidige Fortran-code bouwt en draait op Linux met `gfortran`, maar de output van de huidige code op `DCRexample.txt` komt niet exact overeen met de meegeleverde `DCRexample.results.dcr`.

De oorspronkelijke macOS Mach-O binary is ook teruggehaald uit git en via Darling op WSL2 gedraaid. Die macOS binary produceert byte-identiek dezelfde `results_v2.dcr` als de huidige Linux build. De mismatch komt dus niet door gfortran, Linux, compilerflags of onze herbuild. De huidige broncode en de meegeleverde macOS binary zijn onderling consistent; `DCRexample.results.dcr` lijkt bij een oudere of andere release/configuratie te horen.

Dat verschil lijkt inhoudelijk, niet slechts floating-point/platformverschil. De beste profielrun (`source11-current`) gaf:

```text
43 / 43 regels vergeleken
1935 numerieke velden vergeleken
549 velden verschillen > 0.01
461 velden verschillen > 0.1
301 velden verschillen > 1
116 velden verschillen > 10
max absoluut verschil: 429
gemiddeld absoluut verschil: 9.7017
```

Voorbeeld eerste record:

```text
Expected 305-d ME milk: 21753
Actual   305-d ME milk: 21900
Verschil: +147 lb

Expected 305-d ME fat: 778
Actual   305-d ME fat: 787
Verschil: +9 lb

Expected 305-d ME SCS: 2.21
Actual   305-d ME SCS: 2.25
Verschil: +0.04
```

Werkhypothese: `DCRexample.results.dcr` en de waarden in de manual horen bij een oudere of andere release/configuratie dan de huidige broncode, macOS binary en `bestpred.par`.

## Repo-overzicht

Belangrijkste bestanden:

| Bestand | Rol |
| --- | --- |
| `Best Prediction Manual.pdf` | Manual met theorie, installatie, parameters, file formats en voorbeeld-output. |
| `README` | Korte distributietekst/licentiecontext. |
| `CHANGEFILE.txt` | Changelog; cruciaal om te begrijpen waarom de meegeleverde golden output mogelijk niet meer overeenkomt. |
| `bestpred_main.f90` | Standalone entrypoint. Leest `bestpred.par`, opent inputbestanden, verwerkt bronnen, vormt Format 4 records en roept `bestpred_fmt4()` aan. |
| `bestpred_parm.f90` | Leest Fortran NAMELIST `&bestpred` uit `bestpred.par` en valideert defaults. |
| `bestpred_fmt4.f90` | Parseert AIPL Format 4 records, leest test-day segmenten, bepaalt leeftijd/pariteit/ras/staat en roept C-correcties aan voordat `bestpred()` wordt aangeroepen. |
| `bestpred.f90` | Kernimplementatie van BESTPRED. Bevat `bestpred()`, `interpolate()`, `vary()`, `covary()`, `adjust3X()`, `ymean()`, matrixroutines en output writers. |
| `aiplage.c`, `aiplage.h` | Officiele leeftijd/season/previous-days-open correcties voor M/F/P. |
| `ageadjs.c`, `adjscs.c`, `adjust.scs` | SCS age/stage/month adjustment logica en data. De actieve Linux build linkt `ageadjs.c`; `adjust.scs` wordt door de Python-port geparsed in `python/src/bestpred/core/scs.py`. |
| `makefile.gnu` | Linux/GNU buildfile. Werkt met `gfortran` 13.3 op Ubuntu 24.04. |
| `DCRexample.txt` | Simulatie/testplan input voor source 11. Beste startpunt voor reproducibility. |
| `DCRexample.results.dcr` | Meegeleverde AIPL/reference output voor `DCRexample.txt`; waarschijnlijk legacy golden. |
| `format4.dat` | Kleine Format 4 input, nuttig als smoke test voor source 10 en 15. |
| `format4.means` | Repo-root source-15 meansbestand, maar momenteel leeg; niet bruikbaar als fixture voor de current broncode. |
| `pcdart.bpi`, `pcdart.bpo`, `test241.txt`, `test242.txt`, `pcdart_files.txt` | Source 14/24 PCDART-achtige testdata, later gebruiken. |
| `standard_curves.py` | Python script om standaardcurves te plotten; bijgewerkt in 2025, niet zomaar aannemen als exact equivalent van `bestpred.f90`. |
| `bestpred_curves.py` | Oud Python 2 plot-script voor curve-output. Niet belangrijk voor de port. |
| `bestpred_fortran_analysis.html` | Visueel analyseverslag dat eerder is gemaakt. Handig voor uitleg/diagrammen. |
| `scripts/run_bestpred_profiles.py` | Reproduceerbare harness om parameterprofielen te draaien en output tegen `DCRexample.results.dcr` te vergelijken. |
| `python/` | Nieuwe losse Python package `bestpred-py` voor de gefaseerde port. Gebruikt `uv`, Pydantic v2, Ruff, basedpyright en pytest. |

Actuele Python-port status: source-11 vult partial `results_v2.dcr` velden via
MT M/F/P + ST SCS voor 305/365/laclen/LTD yields, DCR, `RELyld`, `PERSvec`,
`RELpers`, `Yvec`, `herd305` en `bump`. De eerdere M/F/P yield drift is
verklaard en opgelost. Oorzaken waren source-11 `doprev` fallback naar `140`,
de MT melk-LER samplefrequentie (`MRD > 1` gebruikt `weigh = Xmilk`) en de
aparte Fortran `covsum`-route voor partial records. De latere SCS
expanded-output drift is ook opgelost: bij `mtrait=3` gebruikt Fortran voor SCS
`PERSvec(4)/RELpers(4)/305` in plaats van de M/F/P expanded-yield formule. De
huidige open kernstukken zijn volledige source-11 parity, extra sources en
bumpiness/smoothness reductie.
De Python-kernel cachet inmiddels deterministic standaardcurves,
daily-SD matrices, 305-varianties en persistency-varianties per
parameterconfiguratie. De `mtrait=3` route rekent M/F/P lazy via MT en slaat
onnodige ST M/F/P solves over. De all-record golden test valideert nu alle 43
numerieke velden voor alle 43 source-11 fixture rows binnen
Fortran-outputrounding. De laatste gap was cow 0025 met DIM `366`; Python
clamp't die nu intern naar `maxlen=365`, net als `bestpred.f90`, terwijl de
output-DIM `366` blijft. Source 10, 14, 15 en 24 zijn nu op de Python-CLI
aangesloten. Voor source 10 en 15 wordt `format4.dat` geparsed; voor source 14
en 24 worden PCDART/Albert records fixed-width geparsed uit `test241.txt`,
`test242.txt` en `pcdart_files.txt`. De current Fortran numerics voor de
checked-in fixtures matchen nu ook voor die routes, met twee belangrijke
compatibiliteitsnuances:

1. Source 10/source 15: de huidige main-loop schrijft eerst een header-only
   zero-test row weg, daarna pas de geflushte detail-row. Voor source 15 krijgt
   alleen die tweede detail-row een `format4.means` override.
2. Source 14/source 24: de huidige main-loop schrijft per source-14 bestand nog
   een extra EOF/zero-test row weg nadat de laatste echte koe verwerkt is.
   Python bewaart dat artefact expliciet, omdat het wel in de current
   `results_v2.dcr` numerics zit voor deterministische testfiles zoals
   `test241.txt` en `test242.txt`.

De repo-root `format4.means` is leeg en veroorzaakt in de huidige Fortran EOF
op source 15; de Python golden fixture gebruikt daarom een expliciet testbestand
onder `python/tests/fixtures/source15_current/format4.means`.

Voor source 24 is nog een extra afwijking gedocumenteerd: de huidige
Fortran-wrapper schrijft wel correcte `pcdart.bpo`, maar laat in onze Linux-run
een lege `results_v2.dcr` achter. De Python golden fixture voor source 24
gebruikt daarom de geconcateneerde current source-14 `results_v2.dcr` outputs
van `test241.txt` en `test242.txt` als numerieke oracle voor de file-list
wrapper. De echte `pcdart.bpo` output is inmiddels wel geport en wordt voor
source 14 en 24 tegen current Fortran fixtures gevalideerd.

## BESTPRED in gewone taal

BESTPRED voorspelt lactatie-yields door gemeten testdagen te vergelijken met een verwachte standaardcurve. Het model werkt met afwijkingen:

```text
yi = E(yi) + ti

E(yi) = verwachte dagproductie voor managementgroep / standaardcurve
ti    = individuele afwijking van die verwachting
```

De kernvorm is:

```text
y_hat = sum(mu) + 1' C Vm^-1 tm

mu = standaard lactatiecurve
tm = gemeten afwijkingen op testdagen
Vm = variantie/covariantie tussen gemeten testdagen
C  = covariantie tussen lactatie-som of dagcurve en gemeten testdagen
```

De Python-code in `../bovi` bevat deze basisalgebra voor single-trait 305-dagen melk, maar niet de officiele laag eromheen.

## Runtime-flow Fortran

Globale flow:

1. `bestpred_main.f90`
   - roept `read_parms()` aan
   - opent inputbestanden afhankelijk van `source`
   - opent `results_v2.dcr` en `lctcurve.dat`
   - voor `source = 11`: simuleert Holstein testdagrecords uit `DCRexample.txt`
   - maakt een Format 4-achtige record

2. `bestpred_fmt4.f90`
   - parseert cow id, birth, herd, fresh date, lactation length, parity, test-day segmenten
   - leest DIM, supervision, status, milking frequency, sample/weigh frequency, MRD, yield values
   - berekent `agefac` via `aiplage()` en SCS factor via `adjscs4()`
   - schaalt herd SCS
   - roept `bestpred()` aan

3. `bestpred.f90`
   - bepaalt ras, pariteitgroep, regio en seizoen
   - bouwt standard curves en SD curves via `interpolate()`
   - bouwt covarianties via `vary()` en `covary()`
   - berekent deviations per testdag via `ymean()` en correcties
   - inverteert observation covariance matrix
   - berekent `cov * inv(var) * dev`
   - vult `YLDvec`, `PERSvec`, `RELyld`, `RELpers`, `DCRm`, `DCRc`, `DCRs`, `DAILYbp`, `DAILYherd`

4. `bestpred_main.f90`
   - schrijft `results_v2.dcr`
   - eventueel curve/data outputs

## Belangrijke inputbronnen

`bestpred.par` parameter `source`:

| Source | Betekenis |
| --- | --- |
| `10` | Leest AIPL Format 4 uit `format4.dat`. Huidige repo-default. |
| `11` | Leest testing plans uit `DCRexample.txt` en simuleert lactaties. Manual-default voor build-validatie. |
| `12` | USDA master file `input.dcr` (niet aanwezig). |
| `14` | DRMS/PCDART input uit `INfile`, standaard `pcdart.bpi`, output naar `OUTfile`, standaard `pcdart.bpo`. |
| `15` | Zoals source 10, maar leest 305-d herd means uit `format4.means`. |
| `24` | Leest lijst van source 14 files uit `pcdart_files.txt`. |

## Build en run

`gfortran` is geinstalleerd door de gebruiker. Verified:

```text
GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```

Build:

```bash
make -f makefile.gnu
```

Dit bouwt een Linux ELF binary:

```text
bestpred: ELF 64-bit LSB pie executable, x86-64
```

De build geeft veel warnings over oude Fortran DO-labels. Dat is verwacht volgens de changelog en blokkeert de build niet.

Smoke test `source = 10`:

```bash
./bestpred
```

Met de huidige `bestpred.par` draait dit `format4.dat` en eindigt met exitcode 0.

Voor `source = 11` liever niet direct in de repo draaien, omdat BESTPRED vaste outputnamen overschrijft. Gebruik de harness.

## Profiel-harness

Script:

```bash
scripts/run_bestpred_profiles.py --out-dir /tmp/bestpred-profile-runs
```

Wat het doet:

- kopieert `bestpred`, `bestpred.par`, `DCRexample.txt`, `DCRexample.results.dcr`, `adjust.scs` naar geisoleerde run directories
- patcht alleen de gekopieerde `bestpred.par`
- draait `./bestpred`
- vergelijkt generated `results_v2.dcr` tegen `DCRexample.results.dcr`

Beschikbare profielen staan in `PROFILES` in het script:

- `source11-current`
- `source11-linear`
- `source11-wood-linear`
- `source11-linear-g`
- `source11-no3x`
- `source11-old3x`
- `source11-new3x`
- `source11-mtrait4`

Beste profiel tot nu toe:

```text
source11-current
```

Dat profiel patcht:

```text
source = 11
WRITEcurve = 0
WRITEdata = 0
CURVEsingle = 0
ONscreen = 0
maxshow = 0
```

## Originele macOS binary via Darling

De checked-in `bestpred` binary was oorspronkelijk een macOS Mach-O executable. Omdat `bestpred` in de werkboom tijdens deze sessie is vervangen door de Linux ELF build, is de originele binary uit git teruggezet naar een tijdelijke bestandsnaam:

```text
/tmp/bestpred-macrun/bestpred-macos
```

Die binary is via Darling getest op WSL2. Darling moet buiten de Codex sandbox draaien, omdat setuid/root ownership in de sandbox niet correct zichtbaar is.

Laatste betrouwbare run:

```text
Run directory: /tmp/bestpred-macrun-final.CLCHSV
Command: darling /tmp/bestpred-macrun-final.CLCHSV/bestpred-macos
Input: DCRexample.txt
Parameterfile: bestpred.par met source = 11 en output dumps uit
Output: results_v2.dcr
```

Resultaten:

```text
Originele macOS binary output == Linux gfortran output: byte-identiek
Originele macOS binary output != DCRexample.results.dcr

43 / 43 regels vergeleken
1892 numerieke velden vergeleken vanaf kolom 3
549 velden verschillen > 0.01
461 velden verschillen > 0.1
301 velden verschillen > 1
116 velden verschillen > 10
72 velden verschillen > 100
max absoluut verschil: 429
gemiddeld absoluut verschil: 9.9222
```

Conclusie: de macOS binary bewijst dat de current-source output niet een Linux-porting probleem is. Voor een Python port is de current Fortran/macOS output de reproduceerbare oracle. `DCRexample.results.dcr` blijft waardevol als legacy/manual reference, maar is niet de golden output voor deze repository-state.

## Waarom matcht `DCRexample.results.dcr` niet?

De manual zegt dat `DCRexample.txt` met default parameters moet matchen met `DCRexample.results.dcr`, op kleine platformverschillen na. Onze verschillen zijn groter.

Waarschijnlijke oorzaken:

1. De repo is geen volledig consistente release-snapshot.
   - De manual, `DCRexample.results.dcr`, `bestpred.par` en source lijken niet allemaal uit exact dezelfde staat te komen.

2. De huidige `bestpred.par` wijkt af van de manual-default voor de test.
   - Manual default voor validation is `source = 11`.
   - Repo heeft `source = 10`.
   - We patchen dit in de harness, maar dat bewijst dat de checked-in config niet direct de manual-test draait.

3. De changelog noemt inhoudelijke wijzigingen na de voorbeeld-outputperiode.
   Belangrijke wijzigingen:
   - 2009: SCS adjustment naar additive factors.
   - 2009: SCS herd averages naar fenotypische schaal.
   - 2014: `interpolate()` interface aangepast voor regio/seizoen.
   - 2015: `vary()` aangepast voor code 4/robotic herds.
   - 2015: Format 4 state-code fallback.

4. De manualwaarden komen exact overeen met `DCRexample.results.dcr`.
   - Dit bewijst dat manual en expected file bij elkaar horen.
   - Het bewijst niet dat de huidige broncode nog dezelfde output hoort te geven.

5. Profielvariaties hebben de mismatch niet opgelost.
   - `INTmethod='L'`, `INTmethodSCS='L'`, old/new/no 3X, `mtrait=4` zijn geprobeerd.
   - Alle alternatieven waren even slecht of slechter dan `source11-current`.

6. De originele macOS binary matcht de Linux build exact.
   - Daarmee valt "Linux/gfortran wijkt af van de originele binary" af als verklaring.
   - De expected file is waarschijnlijk ouder dan de binary/source combinatie die in deze repo staat.

Werkhypothese: `DCRexample.results.dcr` is legacy expected output. De huidige Fortran-code en originele macOS binary produceren een andere, maar intern consistente output.

## Vergelijking met Bovi Python

Python best-predict staat in:

```text
../bovi/packages/models/lactationcurve/src/lactationcurve/characteristics/best_predict.py
```

Tests:

```text
../bovi/packages/models/lactationcurve/tests/test_best_predict.py
```

Verschillen:

| Onderdeel | Fortran BESTPRED | Bovi Python |
| --- | --- | --- |
| Traits | Milk, fat, protein, SCS | Alleen melk |
| Prediction | Single-trait en multi-trait | Single-trait |
| Lactation length | 305, 365, custom laclen, partial | 305 fixed |
| Standard curve | Dynamisch per ras/pariteit/trait/regio/seizoen | Een vaste NumPy curve |
| Covariance | `vary()`/`covary()` met DIM, MRD, supervision, trait corr, SD | Vaste 305x305 matrix of eenvoudige AR(1) fit |
| Age/season/PDO | C helpers `aiplage`, `adjscs` | Geen |
| 3X milking | `adjust3X()` | Geen |
| DCR/reliability | Ja | Geen |
| Persistency | Ja | Geen |

De Bovi Python-code heeft de juiste algebra-vorm, maar mist de officiele ingredienten en outputlaag.

## Portingstrategie

Niet een grote mechanische transliteratie doen. Eerst Fortran reproduceerbaar als oracle gebruiken.

Voorgestelde fasen:

1. Leg current Fortran golden vast.
   - Gebruik de huidige Linux build en `source11-current`.
   - Bewaar exact de gepatchte `bestpred.par`, stdout en `results_v2.dcr`.
   - Noem dit expliciet `current_fortran`, niet `AIPL_reference`.

2. Bouw een parser voor `results_v2.dcr`.
   - Fixed-width volgens manual Table 4.6.
   - Parse per record de velden: IDs, DCRs, YLDvec, PERSvec, RELs, Yvec, herd305, bump.

3. Port de basis kernel.
   - Begin met single-trait milk 305.
   - Gebruik NumPy/SciPy `solve` of `cho_solve`, geen expliciete inverse tenzij debug nodig is.

4. Port official curves.
   - `interpolate()` naar Python.
   - Maak arrays voor `dyield`, `meanyld`, `sd`.
   - Cache per breed/parity/region/season/method/maxlen.

5. Port covariance.
   - `vary()` en `covary()`.
   - Inclusief MRD, sample/weigh, supervision, trait correlations en SCS/MFP verschillen.

6. Port correcties.
   - `adjust3X()`.
   - `aiplage`/`adjscs` porten naar Python of tijdelijk binden via C/ctypes.

7. Breid uit naar full outputs.
   - M/F/P/SCS.
   - MT en ST.
   - DCR, REL, persistency.
   - 305/365/laclen/partial.

## Python package status

Er staat nu een aparte Python package in `python/`.

Belangrijk:

- Package/distributie: `bestpred-py`.
- Importnaam: `bestpred`.
- Tooling: `uv`, `ruff`, `basedpyright`, `pytest`.
- Directe lokale dependency: `farm-data-definitions`.
- Publieke modellen zijn Pydantic v2 modellen; de core package bevat bewust geen SQLAlchemy/SQLModel tabellen.

Milestone 1 is scaffold + oracle-infrastructuur:

- `bestpred.par` parser via `f90nml`.
- `DCRexample.txt` parser voor source 11.
- `results_v2.dcr` parser en compatibility writer voor golden tests.
- Source-11 simulatie van testday segmenten uit `bestpred_main.f90`.
- CLI met `--source`, `--input`, `--par`, `--output`, `--oracle-output`.
- Current Fortran fixtures onder `python/tests/fixtures/source11_current/`.
- Legacy manual expected onder `python/tests/fixtures/legacy_manual_expected/`.
- Bovi/FDD alignment document: `python/docs/bovi_fdd_alignment.md`.

De numerieke BESTPRED-kernel is gedeeltelijk geport. `bestpred.core.kernel.predict_records()`
schrijft nu partial source-11 rows op basis van de Python ST
305/365/laclen/LTD route voor SCS en de Python MT-route voor melk, vet en
eiwit, inclusief 305 reliability, DCR, persistency, persistency reliability,
expanded yields, herd305 output en bumpiness. Er is nu een golden test die de
velden vastlegt die binnen Fortran-outputrounding matchen; M/F/P yield drift is
opgelost door de source-11 previous-days-open fallback exact te volgen.
Eventuele bumpiness-reductie blijft nog open. De CLI kan met
`--oracle-output` nog steeds de Fortran oracle-output uitschrijven voor
compatibility tests/debug. Dat is geen productie-outputpad, maar een
validatiemechanisme tijdens de port.

Driftdiagnose: de eerste source-11 M/F/P yield drift kwam door
`previous_days_open`. `bestpred_main.f90` zet tijdelijk `doprev=200`, maar
`bestpred_fmt4.f90` leest `doprev` opnieuw uit bytes 246-248 van het Format-4
record. Source 11 vult die bytes niet, waardoor Fortran `doprev < 1` ziet en
naar `140` valt. Python volgt nu die effectieve flow. Voor eerste record ging
305 milk daardoor van `21752.67` naar `21899.76` tegenover Fortran `21900`.

SCS expanded-output diagnose: `Yvec(1,4)` / result index `34` leek eerst een
state/init nuance, maar komt direct uit `bestpred.f90:1513-1528`. Bij
`mtrait=3` zet Fortran SCS `Yvec` op `PERSvec(4) / RELpers(4)` en deelt daarna
door `305`. Voor het eerste source-11 record is dat `-0.002662399`, wat in de
Fortran output als `-0.00` wordt geschreven. Python volgt dit nu.

All-record parity status: de huidige brede golden test dekt alle 43 source-11
rows. Alle numerieke result indexes `0..42` matchen binnen de
Fortran-outputrounding. De voormalige gap was cow 0025 met DIM `366`: Fortran
clamp't `length > maxlen` intern naar `365`, sluit de DIM 366-testdag uit en
schrijft een zero-observation MT row met herd/standard-curve waarden. Python
volgt dat nu.

Drift-onderzoek status: row 26 (`Milk-only`) is opgelost. Fortran zet
`DCRc = DCRvec(2)` als `ntests(3)==0` en laat ST SCS bij `size==0` grotendeels
nul, met `Yvec(1,4)` als `NaN`; Python volgt die missing-data guards nu. Row 2
M/F/P drift is opgelost. Intermediate Fortran trace liet zien dat
`results_v2.dcr` voor source 11 de caller-herd means `20000/700/600` schrijft,
niet de 3X-gecorrigeerde interne `herd305` uit `bestpred()`. Ook muteert
source 11 de SCS herd mean stateful door `bestpred_fmt4.f90:315`: row 1 gebruikt
`3.08`, row 2 `0.0308`, row 3 `0.000308`, enzovoort. Python reproduceert dit
nu alleen in de source-11 simulator/output-assembly voor oracle parity. Dit is
bewust geen generieke BESTPRED-kernlogica en moet worden behandeld als een
waarschijnlijke bug/side effect in de Fortran source-11 demo-flow. Behoud dit
alleen zolang current source-11 oracle parity het doel is. Aanvullend
MT-onderzoek liet zien dat Fortran bij melk-LER records (`MRD > 1`) `q(5)` op
`Xmilk` zet, terwijl Python daar nog `times_weighed` gebruikte. De laatste
partial drift kwam doordat Fortran partial records via `covsum` berekent: een
expliciete som van `vary(actual_observation, pseudo_lactation_day)` voor dag
`1..length`, niet via de `covary(..., length)` lookup. Python volgt beide
routes nu.

ST SCS drift status: opgelost. Python had bij LER-records (`MRD > 1`) voor SCS
ten onrechte `sample = Xmilk` gezet in de observation covariance wrapper.
Fortran zet in de ST-loop alleen `weigh = Xmilk`; `sample` blijft voor
fat/protein/SCS de sampling frequency. Daardoor miste Python de AM/PM
variance-inflatie in `vary()`. Na de fix matchen SCS DCR/yields/persistency/
reliability/bumpiness binnen outputrounding over alle geporteerde source-11
rows.

Fase 2 is inmiddels gestart. De eerste numerieke port-unit staat in:

```text
python/src/bestpred/core/curves.py
python/src/bestpred/core/covariance.py
python/src/bestpred/core/adjustments.py
python/src/bestpred/core/age.py
python/src/bestpred/core/prediction.py
python/src/bestpred/core/scs.py
```

Geport uit `bestpred.f90:2461-3275`:

- `L`: lineaire interpolatie vanuit maandgemiddelden.
- `W`: Wood curves voor melk, vet en eiwit.
- `G`: Morant/Gnanasakthy curves voor SCS.
- `R/C/T`: regio-, seizoen- en regio+seizoen-Wood curves voor melk/vet/eiwit.
- `S/D/U`: regio-, seizoen- en regio+seizoen-Morant/Gnanasakthy curves voor
  SCS.

Nog niet geport:

- full prediction orchestration. DCR, reliability en persistency zijn
  getraceerd maar nog niet als aparte Python helpers geimplementeerd.

Belangrijke nuance: bij SCS `G` volgt de Python-port de Fortran-code, niet de
Fortran-comment. De mean-formule gebruikt geen `/2` op de kwadratische term;
de SD-formule wel.

Unit tests voor de curve-port staan in:

```text
python/tests/test_curves.py
python/tests/test_covariance.py
python/tests/test_adjustments.py
python/tests/test_age.py
python/tests/test_prediction.py
```

Validatie vanaf `python/`:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

## Praktische waarschuwingen

- Draai `./bestpred` liever niet direct in de repo als je alleen wilt testen. Het overschrijft `results_v2.dcr`, `lctcurve.dat` en kan `cowcurve.*`/`cowdata.*` genereren.
- Gebruik de harness met `--out-dir /tmp/...`.
- De checked-in `bestpred` binary was oorspronkelijk macOS Mach-O; die is vervangen door Linux ELF tijdens deze sessie.
- De originele macOS binary staat tijdelijk als `/tmp/bestpred-macrun/bestpred-macos` en is via Darling reproduceerbaar gedraaid.
- Objectfiles (`*.o`) zijn build-artifacts en horen niet in git.
- `DCRexample.results.dcr` niet blind gebruiken als enige waarheid voor de Python-port. Gebruik hem als legacy referentie, maar current Fortran output als praktische oracle.
- `standard_curves.py` is nuttig als leesbare Python context, maar de parameters lijken niet gegarandeerd exact dezelfde als de Fortran-kern.

## Huidige repo-status na deze fase

Belangrijke nieuwe/gewijzigde artefacten:

- `bestpred` is nu een Linux executable.
- `bestpred_fortran_analysis.html` bevat een visueel rapport.
- `scripts/run_bestpred_profiles.py` bevat de profielharness.
- Dit document is de centrale AI wiki.
- `docs/fortran-kernel-trace.md` traceert `interpolate()` en vormt de plek om
  `vary()`/`covary()` bij te schrijven.
- `python/src/bestpred/core/curves.py` bevat de eerste numerieke Python-port.
- `python/src/bestpred/core/covariance.py` bevat de eerste port van `vary()` en
  de table/lookup-laag van `covary()`.
- `python/src/bestpred/core/adjustments.py` bevat de eerste port van `ymean()`
  en `adjust3X()`.
- `python/src/bestpred/core/age.py` bevat de pure Python port van `aiplage.c`;
  de grote coefficient arrays worden uit de gebundelde kopie van `aiplage.h`
  geparsed.
- `python/src/bestpred/data/` bevat de runtime resources voor de Python-port:
  `aiplage.h`, `adjust.scs` en `bestpred.f90`. De loaders gebruiken
  `importlib.resources`, zodat de port niet meer afhankelijk is van relatieve
  repo-root paden.
- `python/src/bestpred/core/prediction.py` bevat de eerste matrix-solve helper
  zonder expliciete inverse.
- `python/src/bestpred/core/kernel.py` bouwt de source-11 partial kernelrun
  met MT M/F/P en ST SCS, inclusief 305/365/laclen/LTD yields, DCR, `RELyld`,
  `PERSvec`, `RELpers`, `Yvec`, `herd305` en `bump`.
- `python/tests/test_curves.py` test de eerste Fortran-formules.
- `python/tests/test_covariance.py` test covariance basisgevallen.
- `python/tests/test_adjustments.py` test ymean en 3X basisgevallen.
- `python/tests/test_age.py` test `aiplage` tegen C-probe waarden.
- `python/tests/test_prediction.py` test de core algebra tegen de expliciete
  Fortran-vorm.
- `python/tests/test_kernel.py` test de minimale source-11 milk-kernelstate.
- `python/src/bestpred/adapters/farm_data_definitions.py` bevat de eerste
  typed adapter van FDD `Cow`/`Herd` plus BESTPRED lactatie/testdag DTO's naar
  `Format4Record`. De Bovi/FDD analyse staat in
  `python/docs/bovi_fdd_alignment.md`.
- `AGENTS.md` verwijst toekomstige agents naar dit document.

## Open vragen

1. Willen we de Linux `bestpred` binary committen, of liever alleen build-instructies en de binary negeren?
2. Willen we `DCRexample.results.dcr` behouden als legacy expected, en daarnaast een nieuwe `tests/golden/current_fortran/source11/results_v2.dcr` toevoegen?
3. Willen we de originele macOS binary als artefact bewaren onder een expliciete naam, of alleen via git history herstellen wanneer nodig?
4. Welke source-11 intermediate covariance cases willen we als eerste debug fixture vastleggen?

Aanbevolen antwoord op 2: ja. Maak expliciet twee golden tracks:

- `legacy_aipl_manual`
- `current_fortran_linux`

Dan kunnen we verschillen bewust documenteren in plaats van ze te verwarren.
