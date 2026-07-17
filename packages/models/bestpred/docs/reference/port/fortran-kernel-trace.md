# Fortran Kernel Trace

Dit document traceert de numerieke BESTPRED-kern naar port-units voor de Python
implementatie. De current Fortran/Linux output blijft de oracle.

## Port Volgorde

| Unit | Fortran locatie | Python target | Status |
| --- | --- | --- | --- |
| Curve interpolatie | `bestpred.f90:2461` (`interpolate`) | `python/src/bestpred/core/curves.py` | geport |
| Mean/deviation op testdagen | `bestpred.f90:2301` (`ymean`) | `python/src/bestpred/core/adjustments.py` | gestart |
| 3X correctie | `bestpred.f90:2169` (`adjust3X`) | `python/src/bestpred/core/adjustments.py` | gestart |
| Age/PDO correctie MFP | `aiplage.c`, `aiplage.h` | `python/src/bestpred/core/age.py` | gestart |
| Lactatievarianties | `bestpred.f90:1912` (`vary`) | `python/src/bestpred/core/covariance.py` | gestart |
| Covarianties | `bestpred.f90:2110` (`covary`) | `python/src/bestpred/core/covariance.py` | gestart |
| Matrix solve | `bestpred.f90:1049`, `bestpred.f90:1317` | `python/src/bestpred/core/prediction.py` | gestart |
| DCR/reliability | `bestpred.f90:1133`, `bestpred.f90:1383`, `bestpred.f90:1502` | `python/src/bestpred/core/reliability.py` | getraceerd |
| Persistency | `bestpred.f90:1247`, `bestpred.f90:1358`, `bestpred.f90:1415` | `python/src/bestpred/core/kernel.py` | single-trait geport |
| Output assembly | `bestpred.f90` source/output blokken | `python/src/bestpred/io/` | deels fase 1 |

## `interpolate`

Fortran locatie: `bestpred.f90:2461-3275`.

Doel: dagcurve, cumulatieve curve, dagelijkse standaarddeviatie en de
persistency-teller (`meanp`) berekenen per trait, lactatiegroep en ras.

Belangrijke inputs:

| Naam | Shape/conventie | Betekenis |
| --- | --- | --- |
| `trait` | integer `1..4` | 1 melk, 2 vet, 3 eiwit, 4 SCS |
| `lacn` | integer `1..2` | 1 eerste lactatie, 2 latere lactaties |
| `breed` | integer `1..6` | 1 AY, 2 BS, 3 GU, 4 HO, 5 JE, 6 MS |
| `method` | character | `L/W/R/C/T` voor MFP, `L/G/S/D/U` voor SCS |
| `maxlen` | integer | aantal dagen waarvoor arrays gevuld worden |
| `region` | integer `1..7` | regio-index voor `R/S/T/U` curves |
| `season` | integer `1..4` | calving-season index voor `C/D/T/U` curves |
| `dyld` | `(12,4,2)` | maandelijkse gemiddelde dagwaarden voor lineair |
| `dsd` | `(12,4,2)` | maandelijkse standaarddeviaties voor lineair |

Belangrijke outputs:

| Naam | Shape/conventie | Betekenis |
| --- | --- | --- |
| `dyield` | `(maxlen,4,2)` | dagwaarde per DIM |
| `meanyld` | `(4,maxlen,2)` | cumulatieve dagwaarde |
| `sd` | `(4,maxlen,2)` | standaarddeviatie per DIM |
| `meanp` | `(2,4)` | som van `dyield(i)*i` |

### Defaults en validatie

- Ongeldige MFP methodes vallen terug op `L`: `bestpred.f90:3016-3021`.
- Ongeldige SCS methodes vallen terug op `L`: `bestpred.f90:3022-3027`.
- Ongeldig ras valt terug op Holstein (`4`): `bestpred.f90:3030-3035`.
- In source 11 zet de Fortran de defaults op `W` voor melk/vet/eiwit en `G`
  voor SCS via `bestpred.par`.
- Python port nu alle methodes: `L/W/R/C/T` voor melk/vet/eiwit en
  `L/G/S/D/U` voor SCS.
- De grote regio/seizoen-tabellen worden uit de originele Fortran `data`
  blocks in `bestpred.f90` geparsed en met Fortran-order gereshaped.
- Omdat `INTmethod='W'` en `INTmethodSCS='G'`, zet de main flow `herdstate`
  op `"00"` en gebruikt hij `region=2`, `season=1`; regio/seizoen-curves zijn
  dus niet nodig voor de current source-11 oracle.

### Methode `L`

Fortran: `bestpred.f90:3041-3058`.

Lineaire interpolatie tussen maandelijkse waarden. De maandindex gebruikt
integerdeling:

```text
month = min(max(1, (i + 15) / 30), 11)
```

Na dag 365 wordt de dagwaarde constant op maand 12 gezet. `meanp` telt alleen
dagen `<=305`.

### Methode `W`

Fortran: `bestpred.f90:3062-3078`.

Wood-curve voor melk, vet en eiwit:

```text
daily = a * dim**b * exp(-dim * c)
sd    = a * dim**b * exp(-dim * c)
```

De parameters komen uit `woods_means` en `woods_sd` (`bestpred.f90:2486-2559`).
Let op: `meanp` telt tot `maxlen`, niet tot 305, omdat de Fortran-conditie
`if (i <= maxlen)` binnen de `1..maxlen` loop altijd waar is.

### Methode `G`

Fortran: `bestpred.f90:3152-3178`.

Morant en Gnanasakthy SCS-curve met een 10-dagen shift:

```text
dim10 = dim + 10
daily = a - b*dim10 + c*dim10**2 + d/dim10
sd    = a - b*dim10 + (c*dim10**2)/2 + d/dim10
```

De comment in Fortran zegt dat ook de mean een `/2` op de kwadratische term
heeft, maar de code doet dat niet. De Python-port volgt de code, niet de
comment.

### Nog Niet Geport

Deze methodes staan nog open omdat ze niet nodig zijn voor source-11 current
defaults:

- `R`: regio-specifieke Wood MFP curves.
- `C`: seizoen-van-afkalven Wood MFP curves.
- `T`: regio plus seizoen Wood MFP curves.
- `S`: regio-specifieke Morant/Gnanasakthy SCS curves.
- `D`: seizoen-specifieke Morant/Gnanasakthy SCS curves.
- `U`: regio plus seizoen Morant/Gnanasakthy SCS curves.

## `vary`

Fortran locatie: `bestpred.f90:1912-2108`.

Python target:

```text
python/src/bestpred/core/covariance.py::observation_covariance
```

Doel: covariance van twee testdagobservaties berekenen. Dit is de bouwsteen
voor de observation covariance matrix (`var`) en voor de lactatie-covariance
tabellen.

Belangrijke inputs:

| Naam | Shape/conventie | Betekenis |
| --- | --- | --- |
| `dim1`, `dim2` | integer DIM | testdag of einddag van LER-segment |
| `trait1`, `trait2` | integer `1..4` | melk, vet, eiwit, SCS |
| `super1`, `super2` | integer `0..9` | supervision/DCR code |
| `Xmilk1`, `Xmilk2` | integer | aantal melkingen per dag |
| `sample1`, `sample2` | integer | sample/weigh frequentie |
| `MRD1`, `MRD2` | integer | multi-day record lengte |
| `sd` | `(4,maxlen,last)` | standaarddeviatiecurve uit `interpolate()` |
| `lacn` | integer `1..2` | pariteitsgroep |

Belangrijke output:

| Naam | Betekenis |
| --- | --- |
| return value | gemiddelde covariance over het meetbereik |

Gedrag dat exact gevolgd moet worden:

- Voor melk met `MRD > 1` wordt gemiddeld over alle dagen
  `dim-MRD+1..dim`.
- Voor vet/eiwit/SCS met `MRD > 1` gebruikt Fortran alleen de middelste dag:
  `dim - (MRD - 1) / 2` met integerdeling.
- Correlaties worden opgebouwd uit:
  - trait correlatie matrix `mfpcorr`.
  - pariteitsgroep-specifieke DIM-correlation formules.
  - aparte formules voor MFP, SCS en MFP-vs-SCS.
  - diagonal override: zelfde trait en zelfde DIM krijgt correlation `1.0`.
  - AM/PM verhoging op dezelfde dag:
    `.3 * mfpcorr * (Xmilk/max(sample1,sample2) - 1)`.
  - unusable supervision code met DCR `0` geeft zeer grote diagonal
    correlation `1000000`.
  - owner-sampler error via DCR codes.
- De uiteindelijke covariance is:

```text
corr * sd(trait1, day_i, lacn) * sd(trait2, day_j, lacn)
```

en wordt gemiddeld over beide meetbereiken.

## `covary`

Fortran locatie: `bestpred.f90:2110-2166`.

Python targets:

```text
python/src/bestpred/core/covariance.py::build_covariance_tables
python/src/bestpred/core/covariance.py::lactation_covariance
```

Doel: covariance tussen een testdagobservatie en lactatie-yield of
persistency lookupen. De Fortran-functie heeft twee modes:

1. `trait == 0`: bouw cached tabellen `covari` en `covd`.
2. `trait != 0`: lookup gemiddelde covariance uit die tabellen.

Belangrijke inputs voor table build:

| Naam | Shape/conventie | Betekenis |
| --- | --- | --- |
| `trt305` | integer `1..4` | target trait waarvoor lactatietotaal wordt opgebouwd |
| `maxlen` | integer | maximale lactatielengte |
| `precise` | integer | stapgrootte; default in Fortran is `1` |
| `sd` | `(4,maxlen,last)` | standaarddeviatiecurve |

Belangrijke tabellen:

| Naam | Fortran shape | Python shape | Betekenis |
| --- | --- | --- | --- |
| `covari` | `(4,4,maxlen,maxlen,last)` | `(4,4,maxlen,maxlen)` per parity | cumulatieve covariance naar lactatie-yield |
| `covd` | `(4,4,maxlen,maxlen,last)` | `(4,4,maxlen,maxlen)` per parity | cumulatieve DIM-gewogen covariance voor persistency |

De table build gebruikt voor de lactatiezijde vaste pseudo-observatie
parameters:

```text
supervision = 1
Xmilk      = 2
sample     = 2
MRD        = 1
```

De lookup-mode gebruikt opnieuw de MRD-meetrange van de werkelijke observatie
en middelt over die range. `stat == 1` kiest `covari`; anders kiest Fortran
`covd`.

### Single-Trait Persistency

Fortran locaties:

- `bestpred.f90:640-664`: bouwt `stdvar`, `dV1`, `dVd`, `varp` en normaliseert
  `covd`.
- `bestpred.f90:1247-1249`: vult single-trait `dcov`.
- `bestpred.f90:1358-1363`: berekent `qCVC1`, `dvari` en `pers`.
- `bestpred.f90:1415-1416`: schrijft `PERSvec` en `RELpers`.

De Python-port in `python/src/bestpred/core/kernel.py` volgt de actieve
single-trait route:

```text
dcov = covary(trait, trait, DIM, 305, ..., stat=2) * varfac**2 / agefac
pers = dcov * inv(var) * dev
RELpers = (dcov * inv(var) * dcov') / varfac**2
```

Belangrijk: `covd` is niet alleen een DIM-gewogen covariance. Tijdens setup
wordt die eerst gecentreerd rond `dim0` en geschaald met `sqrt(varp)`:

```text
covd = (raw_dim_covariance - raw_yield_covariance * dim0) / sqrt(varp)
```

Bij `dim0flag=0` komen de tipping points uit `bestpred.par`. Bij
`dim0flag=1` berekent Fortran nieuwe tipping points; de Python-port ondersteunt
die berekening lokaal, maar muteert de parameterconfig niet.

## `adjust3X`

Fortran locatie: `bestpred.f90:2169-2296`.

Python target:

```text
python/src/bestpred/core/adjustments.py::adjust_3x
```

Doel: factoren berekenen om testdagen, 305-d lactatietotaal en partial yield
te corrigeren voor 3X melken.

Belangrijke inputs:

| Naam | Shape/conventie | Betekenis |
| --- | --- | --- |
| `ntd` | integer | aantal testdagen |
| `dim` | `(maxtd)` | DIM per testdag |
| `length` | integer | lactatielengte/partial length |
| `yearfr` | integer | afkalfjaar voor phase-in |
| `parity` | integer | lactatie/pariteit |
| `Xmilk` | `(maxtd)` | aantal melkingen per dag |
| `meanyld` | `(4,maxlen,last)` | cumulatieve standaardcurve |
| `use3X` | integer `0..3` | geen/oud/nieuw/phase-in |

Belangrijke outputs:

| Naam | Shape/conventie | Betekenis |
| --- | --- | --- |
| `test3X` | `(4,maxtd)` | factor per trait/testdag |
| `fact3X` | `(4)` | factor voor 305-d yield |
| `part3X` | `(4)` | factor voor partial yield |

Gedrag:

- `use3X=0`: geen adjustment.
- `use3X=1`: oude Kendrick 1953 factoren.
- `use3X=2`: nieuwe Karaca 1998 factoren.
- `use3X=3`: lineaire phase-in tussen 1996 en 1999.
- Raw factoren worden omgezet naar multiplicatieve factor:

```text
factor = 1 / (1 + raw_adjustment)
```

- Testdagen met `Xmilk > 2` krijgen de 3X factor; andere testdagen blijven
  `1.0`.
- De 305- en partial-factoren worden afgeleid uit cumulatieve expected yield
  per segment tussen gesorteerde testdagen.

Python nuance: de lokale Fortran array `freq(maxlen)` wordt niet zichtbaar
geinitialiseerd in `adjust3X`. De Python-port gebruikt een expliciete mapping
van DIM naar milking frequency, wat deterministisch is en overeenkomt met de
bedoelde flow.

## `ymean`

Fortran locatie: `bestpred.f90:2301-2332`.

Python target:

```text
python/src/bestpred/core/adjustments.py::expected_daily_yield
```

Doel: expected daily yield voor een observatie berekenen vanuit de standaard
dagcurve en herd ratio.

Belangrijke inputs:

| Naam | Shape/conventie | Betekenis |
| --- | --- | --- |
| `trait` | integer `1..4` | melk, vet, eiwit, SCS |
| `dim` | integer | testdag DIM |
| `MRD` | integer | multi-day record lengte |
| `dyield` | `(maxlen,4,last)` | dagelijkse standaardcurve |
| `hratio` | `(4)` | herd 305 / standaard 305 ratio |

Gedrag:

- Melk met `MRD > 1`: gemiddelde over `dim-MRD+1..dim`.
- Vet/eiwit/SCS met `MRD > 1`: alleen middelste dag
  `dim - (MRD - 1) / 2`.
- Daarna multiplicatieve herd-ratio:

```text
ymean = average_daily_curve_value * hratio(trait)
```

## `aiplage`

C locatie:

- Formula: `aiplage.c`.
- Coefficient data: `aiplage.h`.
- Fortran call site: `bestpred_fmt4.f90:298-299`.

Python target:

```text
python/src/bestpred/core/age.py
```

Doel: age, season en previous-days-open factors voor melk, vet en eiwit
berekenen. BESTPRED gebruikt deze factoren als `agefac(1:3)`.

Portstrategie:

- De formule is geport naar `aiplage_factors()`.
- De grote coefficient arrays blijven in `aiplage.h` staan en worden
  structureel geparsed door `load_aiplage_data()`.
- De Python-port bundelt `aiplage.h` onder `python/src/bestpred/data/` en laadt
  die standaard via `importlib.resources`; expliciete paden zijn alleen nog
  nodig voor debug/probe fixtures.
- `perpetual_day()` port `bestpred_fmt4.f90::pday`.
- `format4_age_factors()` reproduceert de `bestpred_fmt4.f90` record-flow:
  - leeftijd via `pday(fresh) - pday(birth)`;
  - state uit `herd_id[0:2]`;
  - previous-days-open fallback naar 140, en 0 voor eerste lactatie.

Geteste C-parity:

- Holstein source-11 eerste record:
  - age 71 maanden;
  - fresh year 1999;
  - fresh month 4;
  - parity 5;
  - state 12;
  - effective previous days open 140.
    `bestpred_main.f90` zet voor source 11 tijdelijk `doprev = 200`, maar
    `bestpred_fmt4.f90` leest `doprev` opnieuw uit Format-4 bytes 246-248.
    Source 11 vult die bytes niet, dus `doprev < 1` en de fallback wordt 140.
- `agebase=36` tweede-pass pad is ook tegen C-output gepind.

## SCS Age/Stage/Month Adjustment

Fortran locatie:

- `bestpred_fmt4.f90:306-307`:
  `agefac(4) = adjscs4(brd,lacno,frmo,state,age,i305,scs305)/(scs305*1.d0)`

C locatie:

- De Linux build linkt `ageadjs.c`.
- `ageadjs.c::adjscs4()` converteert Fortran `short` waarden naar `char` en
  roept `ageadjs.c::adjscs()` aan.
- `ageadjs.c::adjscs()` laadt bij de eerste call `adjust.scs`.

Python target:

```text
python/src/bestpred/core/scs.py::adjusted_scs
python/src/bestpred/core/scs.py::scs_age_factor
python/src/bestpred/core/scs.py::format4_scs_age_factor
```

Belangrijke details:

- Inputs en output zijn `100 * SCS`.
- Lactatie-index is `0` voor parity 1, anders `1`.
- Breed-index is `1` voor Jersey/Guernsey, anders `0`.
- Leeftijd wordt begrensd op 18..120 maanden.
- DIM wordt begrensd op 15..305.
- Regio komt alleen uit state ranges: `<=23`, `<=48`, `<=74`, anders.
- De actieve `ageadjs.c` formule is additive:

```text
rint(scs - dimfac[dim,lact,breed]
         - monfac[fresh_month,region,breed]
         + (agefac[age,breed] - 1) * 300)
```

Geteste C-parity:

- Source-11 eerste record: `H, parity 5, month 4, state 12, age 71, dim 305,
  scs 329 -> 277`, dus `agefac(4) = 277 / 329`.
- Clamp-cases voor leeftijd, DIM en outputgrenzen staan in
  `python/tests/test_scs.py`.

## Prediction Matrix Solve

Fortran locaties:

- Multi-trait loop: `bestpred.f90:881-1112`.
- Single-trait loop: `bestpred.f90:1164-1364`.
- Matrix inverse: `bestpred.f90:1049-1050` en `bestpred.f90:1317-1318`.
- Matrix multiply helper: `bestpred.f90:2334-2384` (`MULT`).

Python target:

```text
python/src/bestpred/core/prediction.py::solve_prediction_system
```

Fortran algebra:

```text
call invrt2(var, ...)
call mult(covar305, cov305, var, ...)
call mult(vari305, covar305, covp305, ...)
call mult(multi305, covar305, dev, ...)
```

Na `invrt2` bevat `var` de inverse van de observation covariance matrix.
Daarmee is de kern:

```text
weights     = cov305 * inv(var_original)
prediction  = weights * dev
reliability = weights * covp305
```

Python gebruikt hiervoor `np.linalg.solve` in plaats van een expliciete inverse:

```text
prediction = covariance_to_targets * solve(observation_covariance, deviations)
weights    = solve(observation_covariance.T, covariance_to_targets.T).T
```

Dat is dezelfde algebra, maar numeriek stabieler en beter testbaar.

### Multi-trait Loop

Fortran locatie: `bestpred.f90:881-1112`.

Belangrijke stappen:

1. Filter testdagen op geldige DIM en duplicate/overlap status.
2. Voor elk testdag/trait paar tot `mtrait`:
   - zero onbruikbare weigh/sample combinaties.
   - zet vet/eiwitpercentage om naar pounds.
   - bouw `q = [dim, trait, super, Xmilk, sample_or_weigh, MRD]`.
   - bereken deviation:

```text
dev = observed_yield * test3X - ymean(...) / agefac
```

3. Bouw observation covariance matrix `var` met `vary()`, geschaald door
   `varfac` en `agefac`.
4. Bouw covariance vectors/matrices naar 305, 365, custom `laclen`,
   persistency en partial yield.
5. Inverteer `var`.
6. Bereken MT predictions via matrixproducten.

### Single-trait Loop

Fortran locatie: `bestpred.f90:1164-1364`.

De ST-loop herhaalt dezelfde structuur per trait, maar gebruikt een
`1 x size` covariance target in plaats van `4 x size`. ST vult later de fallback
waarden waar MT niet beschikbaar of niet gevraagd is.

## Output Assembly, Reliability En DCR

Fortran locaties:

- MT output assembly: `bestpred.f90:1114-1144`.
- ST output assembly: `bestpred.f90:1366-1440`.
- Age/3X de-adjustment en units: `bestpred.f90:1444-1489`.
- DCR summary en expanded yield/persistency: `bestpred.f90:1502-1529`.

### `YLDvec`

Shape/conventie:

```text
YLDvec(2,16)
row 1 = ME/internal adjusted scale
row 2 = actual/de-adjusted scale

columns:
1..4   = 305 m,f,p,scs
5..8   = 365 m,f,p,scs
9..12  = laclen/maxlen m,f,p,scs
13..16 = partial m,f,p,scs
```

MT vult `YLDvec(1,*)` met:

```text
predicted_deviation + herd_total
```

ST doet hetzelfde per trait. Partial yield gebruikt:

```text
partrec + meanyld(trait, length, lacn) * hratio(trait)
```

Daarna wordt row 2 afgeleid door age en 3X terug te delen:

```text
actual 305/365/laclen = ME / (agefac * fact3X)
actual partial        = ME / (agefac * part3X)
```

Als `UNITSout == 'P'`, worden melk/vet/eiwit terug naar pounds vermenigvuldigd
met `lb=2.205`. SCS blijft op score-schaal en wordt gedeeld door lengte:

```text
305 SCS = sum / 305
365 SCS = sum / 365
laclen SCS = sum / laclen
partial SCS = sum / length
```

### Reliability

MT:

```text
RELyld(trait)  = vari305(trait,trait) / (stdvar(trait,lacn) * varfac(trait)**2)
RELpers(trait) = dvari(trait,trait) / varfac(trait)**2
```

ST:

```text
RELyld(trait)  = vari305(1,1) / (stdvar(trait,lacn) * varfac(trait)**2)
RELpers(trait) = dvari(1,1) / varfac(trait)**2
```

`nbump` is a compile-time parameter set to `0` in this source, so bumpiness does
not currently reduce reliability. The ST bumpiness code is still relevant for
reporting and future parity but is inactive for DCR reduction.

Python status:

- Source-11 single-trait `bump` is geport in
  `python/src/bestpred/core/kernel.py`.
- Omdat `nbump=0` in de Fortran source staat, gebruikt de Python-port bumpiness
  nu alleen als outputveld en niet als reliability/DCR reductie.
- Voor de current `mtrait=3` oracle blijven M/F/P bumpvelden in Fortran via de
  nog-niet-geporte MT-route anders dan de ST debug-route; SCS komt direct uit
  de ST-route.

### DCR

Per trait:

```text
DCRvec(trait) = 100 * RELyld(trait) / rmonth(trait,lacn)
```

Summary outputs:

```text
DCRm = DCRvec(1)
DCRc = mean(DCRvec(2), DCRvec(3))
DCRs = DCRvec(4)
if no protein tests: DCRc = DCRvec(2)
```

### Partial Records

For 305/365/laclen MT targets, Fortran uses `covary()` lookup tables. Partial
records are different: `bestpred.f90:992-997` builds `covsum` by summing
`vary()` over each lactation day `1..length` with the actual observation
metadata on the test-day side:

```text
covsum(target_trait, obs) += vary(actual_obs_dim, actual_obs_trait,
                                 actual_super, actual_xmilk,
                                 actual_sample_or_weigh, actual_mrd,
                                 lactation_day, target_trait,
                                 1, 2, 2, 1, ...)
```

That matters for source-11 row 2: milk LER records use `q(5)=weigh=Xmilk`
inside the MT loop, so Python must not reuse the 305/365 `covary` path for
partial M/F/P output.

### Length Greater Than `maxlen`

`bestpred.f90:736-738` clamp't `length > maxlen` intern naar `maxlen` voordat
3X, herd curves en outputvectors worden berekend. Testdagen met `dim > maxlen`
worden daarna niet in `dimvec` gezet en vallen ook uit de MT/ST loops. Source
11 cow 0025 heeft `length=366` en een enige testdag op DIM 366; Fortran
schrijft de originele output-DIM `366`, maar rekent intern alsof
`length=365` en `size == 0`. Python volgt die combinatie nu expliciet.

### Expanded Yield/Persistency

For MT:

```text
Yvec = herd305 + (YLDvec305 - herd305) / regrel
```

For ST or SCS fallback:

```text
Yvec = PERSvec / RELpers
```

with a guard for `RELpers == 0` in the all-ST path.

Python status:

- Source-11 partial output vult `Yvec` voor alle vier traits.
- M/F/P gebruiken de actieve expanded-yield formule op interne 305-d schaal en
  worden daarna naar output units geconverteerd.
- SCS gebruikt in de current `mtrait=3` route de Fortran fallback
  `PERSvec(4) / RELpers(4)`, en wordt daarna als score per dag geschreven door
  deling met `305`. Voor het eerste source-11 record geeft dit
  `-0.002662399`, afgerond/geformatteerd als `-0.00`.
- `herd305` output vult nu ook SCS met de breed mean fallback uit `brdyld`
  (`3.08` voor Holstein later-lactation in de source-11 fixture).

## Volgende Trace Units

De eerstvolgende implementatiestap is niet nog een losse formule, maar een
minimale orchestration voor één source-11 record:

1. curves bouwen voor M/F/P/SCS;
2. covariance tabellen bouwen;
3. herd means, `hratio`, `varfac`, 3X factors berekenen;
4. observaties naar deviations en matrices omzetten;
5. `solve_prediction_system()` aanroepen;
6. eerste 305-d milk output vergelijken met Fortran intermediate/debug output.

Status: stap 1, 3, 4 en 5 bestaan nu voor ST milk 305 in
`predict_source11_milk_305_debug`. De covariance-to-305 vector wordt direct
berekend in plaats van via de volledige `covari` table, omdat dat voor één
trait/record sneller en traceerbaarder is. De milk age-factor gebruikt nu de
geporteerde `aiplage` via `format4_age_factors()`, maar kan nog expliciet worden
overridden voor debugexperimenten.
