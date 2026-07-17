"""Typed domain models for BESTPRED inputs and outputs."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BestpredModel(BaseModel):
    """Base Pydantic model used by the public package interfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BestpredSource(IntEnum):
    """Standalone BESTPRED input sources."""

    FORMAT4 = 10
    DCR_EXAMPLE = 11
    USDA_MASTER = 12
    DRMS = 14
    FORMAT4_WITH_MEANS = 15
    DRMS_FILE_LIST = 24


class Trait(StrEnum):
    """Traits supported by BESTPRED."""

    MILK = "milk"
    FAT = "fat"
    PROTEIN = "protein"
    SCS = "scs"


class BreedCode(StrEnum):
    """BESTPRED breed codes used by source 11 and Format 4 records."""

    AYRSHIRE = "AY"
    BROWN_SWISS = "BS"
    GUERNSEY = "GU"
    HOLSTEIN = "HO"
    JERSEY = "JE"
    MILKING_SHORTHORN = "MS"

    @classmethod
    def from_source11_index(cls, value: int) -> BreedCode:
        mapping = {
            1: cls.AYRSHIRE,
            2: cls.BROWN_SWISS,
            3: cls.GUERNSEY,
            4: cls.HOLSTEIN,
            5: cls.JERSEY,
            6: cls.MILKING_SHORTHORN,
        }
        try:
            return mapping[value]
        except KeyError as exc:
            raise ValueError(f"Unsupported source-11 breed index: {value}") from exc

    @property
    def fortran_trait_prefix(self) -> str:
        return self.value[0]


class BestpredParameters(BestpredModel):
    """Subset of `bestpred.par` parameters used by the Python port."""

    laclen: int = 305
    maxlen: int = 365
    dailyfreq: int = 6
    plotfreq: int = 6
    use3x: int = Field(default=3, alias="use3X")
    mtrait: int = 3
    global_mtrait: int = Field(default=3, alias="GLOBALmtrait")
    source: BestpredSource = BestpredSource.DCR_EXAMPLE
    breed11: int = 4
    breed_unk: int = Field(default=4, alias="breedUNK")
    write_curve: int = Field(default=0, alias="WRITEcurve")
    write_data: int = Field(default=0, alias="WRITEdata")
    curve_single: int = Field(default=0, alias="CURVEsingle")
    curve_small: int = Field(default=0, alias="CURVEsmall")
    curve_file: str = Field(default="cowcurve", alias="CURVEfile")
    data_file: str = Field(default="cowdata", alias="DATAfile")
    in_file: str = Field(default="pcdart.bpi", alias="INfile")
    out_file: str = Field(default="pcdart.bpo", alias="OUTfile")
    maxprnt: int = 0
    onscreen: int = Field(default=0, alias="ONscreen")
    obs: int = 99_999_999
    maxshow: int = 0
    maxtd: int = 50
    int_method: str = Field(default="W", alias="INTmethod")
    int_method_scs: str = Field(default="G", alias="INTmethodSCS")
    debug_msgs: int = Field(default=0, alias="DEBUGmsgs")
    debug_parms: int = Field(default=0, alias="DEBUGparms")
    dim0: tuple[int, int, int, int, int, int, int, int] = (
        115,
        115,
        150,
        155,
        161,
        152,
        159,
        148,
    )
    dim0flag: int = 0
    log_on: int = Field(default=0, alias="LOGon")
    log_file: str = Field(default="example", alias="LOGfile")
    log_freq: int = Field(default=0, alias="LOGfreq")
    region: int = 1
    season: int = 1
    units_in: str = Field(default="P", alias="UNITSin")
    units_out: str = Field(default="P", alias="UNITSout")
    pers_floor: float = Field(default=-9.99, alias="PERSfloor")
    pers_ceiling: float = Field(default=9.99, alias="PERSceiling")
    grafplot: tuple[int, int, int, int] = Field(default=(0, 0, 0, 0), alias="GRAFplot")

    @field_validator("int_method", "int_method_scs", "units_in", "units_out", mode="before")
    @classmethod
    def normalize_single_character(cls, value: object) -> str:
        return str(value).strip().strip("'\"")

    @property
    def source11_breed(self) -> BreedCode:
        return BreedCode.from_source11_index(self.breed11)


class Source11PlanLine(BestpredModel):
    """One non-comment row from `DCRexample.txt`."""

    supervised: int
    times_milked: int
    times_weighed: int
    times_sampled: int
    ler_days: int
    test_interval: int
    first_test: int
    last_test: int
    parity: int
    name: str = ""


class Source11Example(BestpredModel):
    """A source-11 example record, which can contain one or more plan lines."""

    number: int
    plan_lines: tuple[Source11PlanLine, ...]


class TestDaySegment(BestpredModel):
    """Format 4-style test-day segment constructed for BESTPRED."""

    dim: int
    supervised: int
    status: int
    times_milked: int
    times_weighed: int
    times_sampled: int
    ler_days: int
    percent_shipped: int
    milk_yield: int
    fat_percent: int
    protein_percent: int
    scs: int

    def to_fortran_segment(self) -> str:
        """Render the 23-character segment produced by Fortran format 145."""

        return (
            f"{self.dim:3d}"
            f"{self.supervised:1d}"
            f"{self.status:1d}"
            f"{self.times_milked:1d}"
            f"{self.times_weighed:1d}"
            f"{self.times_sampled:1d}"
            f"{self.ler_days:2d}"
            f"{self.percent_shipped:3d}"
            f"{self.milk_yield:4d}"
            f"{self.fat_percent:2d}"
            f"{self.protein_percent:2d}"
            f"{self.scs:2d}"
        )


class Format4Record(BestpredModel):
    """Minimal Format 4-style record produced by source 11."""

    cow_id: str
    birth_date: str
    herd_id: str
    fresh_date: str
    parity: int
    length: int
    previous_days_open: int
    herd_me_milk: int
    herd_me_fat: int
    herd_me_protein: int
    herd_me_scs: float | None = None
    herd_deviation_milk: int = 0
    herd_deviation_fat: int = 0
    herd_deviation_protein: int = 0
    compatibility_tag: str | None = None
    segments: tuple[TestDaySegment, ...]


class Format4MeansRecord(BestpredModel):
    """One source-15 `format4.means` row."""

    cow_id: str
    fresh_date: str
    herd_me_milk: int
    herd_me_fat: int
    herd_me_protein: int
    herd_me_scs: float


class DcrResultRow(BestpredModel):
    """One row from `results_v2.dcr` or `DCRexample.results.dcr`."""

    animal_id: str
    fresh_date: str
    dim: int
    numeric_values: tuple[float, ...]
    raw_line: str


class BestpredRunRequest(BestpredModel):
    """CLI/run request."""

    source: BestpredSource
    input_path: Path
    output_path: Path
    parameter_path: Path
    oracle_output_path: Path | None = None
    pcdart_output_path: Path | None = None
