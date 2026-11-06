"""What a number means, and the one way it is shown to a person.

Unit definitions and registry rules live in the native core; Python exposes values and caches.

Historically units lived here, in the bottom layer, because a unit is a fact about data --
what the number in this column is -- and not about drawing it.  They used to
live in ``zlc_plot``, which meant the layer that compiles a pulse could not
see them and kept a second, smaller table of its own; the two disagreed about
the base (seconds against nanoseconds), about spelling (this one accepts
``µs``, that one did not), and about arithmetic.

**A prefix is not a unit.**  ``kHz`` was once a registry entry beside ``Hz``,
``MHz`` and ``GHz``, which made "show this in the next unit up" mean "look up
a different object" -- so nothing ever did it, and the table had holes wherever
nobody had typed the row (``kHz`` existed, ``mHz`` did not; ``um`` existed,
``pm`` did not).  Here the registry holds BASE units, one per dimension, plus
the handful that are genuinely their own thing, and every prefixed spelling is
derived on resolution.  Changing scale is then arithmetic on an exponent,
which is what lets one rule format ``120000000.0 Hz`` as ``120.0000000 MHz``
and read ``1.05M`` back.

**Showing a number never changes it.**  The stored value is always in the
unit its owner declared; a prefix is chosen when the string is built and
forgotten immediately after.  This is the same discipline that ``base`` and
``canonical`` had to be separated to get right, and it is why formatting here
shifts a decimal point instead of multiplying a float: shifting is exact, so
what is displayed is what is held, digit for digit.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from threading import RLock

import numpy as np
from numpy.typing import ArrayLike, NDArray


class UnitError(ValueError):
    """A unit was not understood, or two units could not meet."""


# --------------------------------------------------------------------- prefix


@dataclass(frozen=True, slots=True)
class Prefix:
    """One decimal step, in the spelling shown and the spellings accepted.

    ``symbol`` is what a person READS and ``accepts`` what they may TYPE: the
    micro prefix prints as ``µ`` because that is what it is, and reads ``u``
    because that is what a keyboard has.  One table serves both directions, so
    a spelling that can be shown can always be typed back.
    """

    symbol: str
    exponent: int
    accepts: tuple[str, ...] = ()

    @property
    def spellings(self) -> tuple[str, ...]:
        return (self.symbol, *self.accepts)


#: Every prefix some unit's ladder uses, largest first.  This table says how
#: a prefix is spelled and what it multiplies by; WHICH of them a unit is
#: read in is the unit's own declaration (``Unit.prefixes``), because that is
#: a fact about the instrument and the quantity -- a frequency here goes up
#: to gigahertz and a temperature down to nanokelvin, and neither is read in
#: teraseconds.  The identity step is a member like any other so that "no
#: prefix" needs no special case anywhere below.
PREFIXES: tuple[Prefix, ...] = (
    Prefix("G", 9),
    Prefix("M", 6),
    Prefix("k", 3),
    Prefix("", 0),
    Prefix("m", -3),
    Prefix("µ", -6, ("u", "μ")),
    Prefix("n", -9),
)

_PREFIX_BY_EXPONENT = {prefix.exponent: prefix for prefix in PREFIXES}
#: The identity step, for a caller that wants a number in EXACTLY the unit it
#: names -- a reading converted into the hertz the operator chose for a row
#: stays in hertz however large it is, because that row is the one place the
#: operator said which scale they read.
NO_PREFIX = _PREFIX_BY_EXPONENT[0]
# ----------------------------------------------------------------- conversion


@dataclass(frozen=True, slots=True)
class Scaled:
    """``base = value * factor`` -- every ordinary unit."""

    factor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", float(self.factor))
        _conversion_facts(self)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=False)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=True)


@dataclass(frozen=True, slots=True)
class Offset:
    """``base = value + offset`` -- a scale with its zero somewhere else.

    A Celsius reading is a kelvin reading with 273.15 taken off, so it is
    not ``value * factor`` and never was.  A unit read this way takes no
    prefix: a thousandth of a degree Celsius is not a unit, and the
    arithmetic that moves a decimal point would move the zero with it.
    """

    offset: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", float(self.offset))
        _conversion_facts(self)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=False)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=True)


@dataclass(frozen=True, slots=True)
class Decibel:
    """A level: ``base = reference * 10 ** (value / factor)``.

    dBm is a power written logarithmically, so it is not ``value * factor``
    and never was: no scale exists that turns -3 dBm into watts.  Giving it
    one anyway is what an affine-only registry forces, and refusing to admit
    it is why ``dBm`` was simply absent -- the RF driver declared it, the plot
    contract had never heard of it, and the first scan of a signal generator's
    power died on ``unknown unit 'dBm'``.

    ``factor`` is 10 for a power ratio.  Amplitude ratios (20) exist and are
    not registered here, because nothing in this project measures one.
    """

    reference: float
    factor: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference", float(self.reference))
        object.__setattr__(self, "factor", float(self.factor))
        _conversion_facts(self)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=False)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=True)


@dataclass(frozen=True, slots=True)
class VoltageIntoLoad:
    """Voltage expressed as the power it delivers into a stated load.

    ``base = value**2 / (mean_square_divisor * load_ohms)``. The divisor
    is 8 for a sine's peak-to-peak amplitude, 1 for RMS. The registry's
    default is 50 ohms; a device with another load owns its conversion,
    rather than treating that convention as a measurement of its load.
    """

    load_ohms: float
    mean_square_divisor: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "load_ohms", float(self.load_ohms))
        object.__setattr__(self, "mean_square_divisor", float(self.mean_square_divisor))
        _conversion_facts(self)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=False)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=True)


@dataclass(frozen=True, slots=True)
class Prefixed:
    """A prefix in front of a unit whose conversion is not a plain scale.

    A prefix multiplies the NUMBER written in the unit -- a millivolt
    peak-to-peak is a thousandth of a volt peak-to-peak -- so it is applied
    before the unit's own conversion and undone after it.  A linear unit
    never needs this: its prefixed spelling is one Scaled with an exact
    decade, which is what the compiler reads.
    """

    inner: object
    factor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", float(self.factor))
        _conversion_facts(self)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=False)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return _apply_conversion(self, values, inverse=True)


Conversion = Scaled | Offset | Decibel | VoltageIntoLoad | Prefixed


# ------------------------------------------------------------------- the unit


@dataclass(frozen=True, slots=True)
class Unit:
    """One way of writing a quantity of one dimension.

    A unit is either registered (``s``, ``dBm``, ``pixel``) or derived from a
    registered base by a prefix (``ms``, ``MHz``).  Derived units are built on
    resolution and compare equal by value, so two resolutions of one spelling
    are one unit.  A dataset never holds a Unit: an axis and a value schema
    record the SPELLING (``AxisSpec.unit`` and ``ValueSchema.value_unit`` are
    text, and the codec writes that text), and whoever reads the dataset
    resolves it against a registry that knows the same spellings.
    """

    symbol: str
    dimension: str
    conversion: Conversion = field(default_factory=lambda: Scaled(1.0))
    #: The prefixes this unit is read in, besides its bare symbol: ``("m",
    #: "µ", "n")`` for the second, nothing for a unit that takes none.  What
    #: an instrument here is read in is a fact about the quantity, and only
    #: the reference of a spelling family carries them -- ``mms`` is not a
    #: unit, and neither is ``mdBm``.
    prefixes: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    #: The dimension this one becomes when inverted, for the axes that can be
    #: read either way round (a time is a frequency and back).
    inverse_dimension: str | None = None
    #: The EXACT power of ten this unit is of its base, when it is one.
    #: DERIVED, never authored: it is a fact about the conversion, and a
    #: second place to state it is a second place to state it wrongly.
    #:
    #: ``scale`` is this number's float shadow and is not good enough
    #: everywhere: a microsecond is 1e-6 s, and 1e-6/1e-9 is not exactly a
    #: thousand in binary.  Whoever must be exact -- the compiler turning a
    #: duration into device ticks -- reads the integer and builds its own
    #: exact ratio from it.  ``None`` for a unit that is not a power of ten
    #: (a degree) and for a level (dBm).
    decade: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if isinstance(self.prefixes, str) or isinstance(self.aliases, str):
            raise UnitError("unit prefixes and aliases must be sequences, not strings")
        answer = _unit_native("units.validate", unit=_unit_tree(self))
        object.__setattr__(self, "prefixes", tuple(answer["prefixes"]))
        object.__setattr__(self, "aliases", tuple(answer["aliases"]))
        object.__setattr__(self, "decade", answer["decade"])

    @property
    def ladder(self) -> tuple[Prefix, ...]:
        """The rungs this unit is shown and read in, largest first.

        The bare symbol is always one; the others are the prefixes the unit
        declares.  A unit that declares none has a ladder of one rung, and
        every ladder is read off the one prefix table so the rungs of every
        unit are spelled and ordered the same way.
        """

        offered = _unit_facts(self)["ladder"]
        return tuple(prefix for symbol in offered for prefix in PREFIXES if prefix.symbol == symbol)

    @property
    def is_linear(self) -> bool:
        return bool(_unit_facts(self)["linear"])

    @property
    def coordinate_scale(self) -> tuple[str, float] | None:
        result = _unit_facts(self)["coordinate_scale"]
        return None if result is None else tuple(result)

    @property
    def scale(self) -> float:
        """How many base units one of this unit is.

        Only linear units have one.  Asking a level for its scale is the
        mistake this raises on rather than answering with a number that would
        be wrong everywhere it was used.
        """

        if not isinstance(self.conversion, Scaled):
            raise UnitError(f"{self.symbol!r} is not a linear unit and has no scale")
        return self.conversion.factor

    @property
    def is_base(self) -> bool:
        """Whether this is its dimension's reference unit (seconds, not ms).

        The BASE is the unit system's reference, and only that.  It is not the
        "canonical" unit: canonical, everywhere in this project, is the unit a
        dataset is written in, and a dataset written in microseconds is
        canonical in microseconds.  These used to share the word, and a
        conversion to the dataset's unit was written as a conversion to the
        base -- exact for metres and volts, a million times off for a
        microsecond axis.
        """

        return bool(_unit_facts(self)["base"])

    def compatible_with(self, other: "Unit") -> bool:
        return self.dimension == other.dimension

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        """Values in this unit, in the dimension's base unit."""

        return self.conversion.to_base(values)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        """Values in the dimension's base unit, in this unit."""

        return self.conversion.from_base(values)

    def convert_value_to(self, values: ArrayLike, target: "Unit") -> NDArray[np.generic]:
        """Values in this unit, in ``target`` -- the one conversion callers use."""

        if self == target:
            return np.asarray(values)
        facts = _conversion_facts(self.conversion)
        array = np.asarray(values, dtype=float if facts["initial_real_to"] else None)
        return _conversion_answer(_unit_native("units.convert", values=array,
                            source=_unit_tree(self), target=_unit_tree(target)))



UnitLike = str | Unit


def _conversion_tree(conversion):
    if isinstance(conversion, Scaled):
        return {"kind": "scaled", "factor": conversion.factor}
    if isinstance(conversion, Offset):
        return {"kind": "offset", "offset": conversion.offset}
    if isinstance(conversion, Decibel):
        return {"kind": "decibel", "reference": conversion.reference, "factor": conversion.factor}
    if isinstance(conversion, VoltageIntoLoad):
        return {"kind": "voltage", "load_ohms": conversion.load_ohms, "divisor": conversion.mean_square_divisor}
    if isinstance(conversion, Prefixed):
        return {"kind": "prefixed", "inner": _conversion_tree(conversion.inner), "factor": conversion.factor}
    raise UnitError("unknown unit conversion")


def _conversion_from_tree(value):
    # The native answer is already validated; adopting it is a value operation,
    # not a second native request during catalogue construction.
    kind = value["kind"]
    classes = {"scaled": Scaled, "offset": Offset, "decibel": Decibel, "voltage": VoltageIntoLoad, "prefixed": Prefixed}
    if kind not in classes:
        raise UnitError("unknown native unit conversion")
    result = object.__new__(classes[kind])
    for name, item in value.items():
        if name == "kind":
            continue
        if name == "inner":
            item = _conversion_from_tree(item)
        object.__setattr__(result, "mean_square_divisor" if name == "divisor" else name, item)
    return result


@lru_cache(maxsize=512)
def _conversion_facts(conversion):
    return _unit_native("units.conversion_facts", conversion=_conversion_tree(conversion))


def _conversion_answer(answer):
    if answer["discarded_imaginary"]:
        import warnings
        warnings.warn("Casting complex values to real discards the imaginary part", np.exceptions.ComplexWarning, stacklevel=3)
    values = answer["values"]
    return values[()] if values.ndim == 0 else values


def _apply_conversion(conversion, values, *, inverse):
    facts = _conversion_facts(conversion)
    array = np.asarray(values, dtype=float if facts["initial_real_from" if inverse else "initial_real_to"] else None)
    return _conversion_answer(_unit_native("units.apply", conversion=_conversion_tree(conversion), values=array, inverse=inverse))


def _unit_tree(unit):
    return {"symbol": unit.symbol, "dimension": unit.dimension,
            "conversion": _conversion_tree(unit.conversion), "prefixes": list(unit.prefixes),
            "aliases": list(unit.aliases), "inverse_dimension": unit.inverse_dimension}


def _unit_from_tree(value):
    # Native definitions are already validated. Reconstructing a Python value
    # must not recursively ask the backend to validate the backend's own answer.
    unit = object.__new__(Unit)
    for name in ("symbol", "dimension", "inverse_dimension", "decade"):
        object.__setattr__(unit, name, value[name])
    for name in ("prefixes", "aliases"):
        object.__setattr__(unit, name, tuple(value[name]))
    object.__setattr__(unit, "conversion", _conversion_from_tree(value["conversion"]))
    return unit


def _unit_native(operation, **params):
    from zlc_native import call
    try:
        return call(operation, **params)
    except ValueError as error:
        raise UnitError(str(error)) from None


@lru_cache(maxsize=512)
def _unit_facts(unit):
    return _unit_native("units.facts", unit=_unit_tree(unit))


@lru_cache(maxsize=512)
def _prefixed(base: Unit, prefix: Prefix) -> Unit:
    if prefix.exponent == 0:
        return base
    return _unit_from_tree(_unit_native("units.resolve", units=[_unit_tree(base)], unit=prefix.symbol + base.symbol))


class UnitRegistry:
    """Python values/cache over the core's canonical registry rules."""

    def __init__(self, units: Iterable[Unit] = (), *, _defaults: bool = False) -> None:
        self._lock = RLock()
        self._revision = 0
        self._definitions = []
        self._registered = {}
        self._cache = OrderedDict()
        self._metadata_cache = OrderedDict()
        self._defaults = _defaults
        for unit in units:
            self.register(unit)

    def _ensure(self):
        if self._defaults:
            self._adopt(_unit_native("units.builtins")["units"])
            self._revision = len(self._definitions)
            self._defaults = False

    def _adopt(self, definitions):
        self._definitions = definitions
        self._registered = {}
        self._cache.clear()
        self._metadata_cache.clear()
        for definition in definitions:
            unit = _unit_from_tree(definition)
            for name in (unit.symbol, *unit.aliases):
                self._registered[name] = unit

    def _call(self, operation, **params):
        self._ensure()
        return _unit_native(operation, units=self._definitions, **params)

    def _metadata(self, operation, unit):
        """Bounded metadata cache; a registry replacement invalidates every answer."""
        key = (operation, unit)
        if key in self._metadata_cache:
            self._metadata_cache.move_to_end(key)
            return self._metadata_cache[key]
        answer = self._call(operation, unit=_unit_tree(unit) if isinstance(unit, Unit) else unit)
        self._metadata_cache[key] = answer
        if len(self._metadata_cache) > 512:
            self._metadata_cache.popitem(last=False)
        return answer

    @property
    def revision(self) -> int:
        with self._lock:
            self._ensure()
            return self._revision

    def register(self, unit: Unit, *, replace: bool = False) -> Unit:
        if not isinstance(unit, Unit):
            raise TypeError("unit must be a Unit")
        with self._lock:
            answer = self._call("units.register", unit=_unit_tree(unit), replace=bool(replace))
            self._adopt(answer["units"])
            self._revision += 1
            return self._registered[unit.symbol]

    def resolve(self, unit: UnitLike) -> Unit:
        if isinstance(unit, Unit):
            return unit
        if not isinstance(unit, str):
            raise TypeError("unit must be a Unit or registered symbol")
        with self._lock:
            self._ensure()
            if unit in self._registered:
                return self._registered[unit]
            if unit not in self._cache:
                self._cache[unit] = _unit_from_tree(self._call("units.resolve", unit=unit))
                if len(self._cache) > 512:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(unit)
            return self._cache[unit]

    def base_for(self, unit_or_dimension: UnitLike) -> Unit:
        with self._lock:
            return _unit_from_tree(self._metadata("units.base", unit_or_dimension))

    def family_of(self, unit: UnitLike) -> Unit:
        return self.family_and_prefix(unit)[0]

    def family_and_prefix(self, unit: UnitLike) -> tuple[Unit, Prefix]:
        with self._lock:
            answer = self._metadata("units.family", self.resolve(unit))
            return _unit_from_tree(answer["unit"]), _PREFIX_BY_EXPONENT[int(answer["prefix"])]

    def convert(self, values: ArrayLike, source: UnitLike, target: UnitLike) -> NDArray[np.generic]:
        return self.resolve(source).convert_value_to(values, self.resolve(target))

    def convert_decimal(self, value: object, source: UnitLike, target: UnitLike) -> Decimal:
        number = value if isinstance(value, Decimal) else decimal_of(value)
        if number is None or not number.is_finite():
            raise UnitError("unit conversion requires a finite scalar")
        with self._lock:
            return Decimal(self._call("units.decimal", value=str(number),
                source=_unit_tree(self.resolve(source)), target=_unit_tree(self.resolve(target))))

    def inverse_for(self, unit: UnitLike) -> Unit | None:
        with self._lock:
            answer = self._metadata("units.inverse", self.resolve(unit))
            return None if answer is None else _unit_from_tree(answer)

    def inverse_symbol(self, unit: UnitLike) -> str:
        with self._lock:
            return self._metadata("units.inverse_symbol", self.resolve(unit))

    def distinct_symbols(self) -> tuple[str, ...]:
        with self._lock:
            self._ensure()
            return tuple(sorted(definition["symbol"] for definition in self._definitions))

    def display_choices(self, unit: UnitLike) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._metadata("units.choices", self.resolve(unit)))


#: The load an RF amplitude is read into, in ohms.  The convention every
#: signal generator's front panel assumes when it shows dBm beside Vpp.
RF_LOAD_OHMS = 50.0


def _builtin_units() -> tuple[Unit, ...]:
    return tuple(_unit_from_tree(value) for value in _unit_native("units.builtins")["units"])


# Lazy: importing data types never spawns a process or requires an active host.
DEFAULT_UNITS = UnitRegistry(_defaults=True)


def resolve_unit(value: UnitLike, registry: UnitRegistry | None = None) -> Unit:
    return (registry or DEFAULT_UNITS).resolve(value)


# ------------------------------------------------------------- showing a value


def decimal_of(value: object) -> Decimal | None:
    """The exact decimal a float IS, or None when it is not a finite number.

    ``repr`` of a float is the shortest string that reads back as the same
    float, which is the only defensible answer to "how many digits does this
    number have": anything shorter is a different number and anything longer
    is invention.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        # A whole number has no decimals, and inventing "512.0" for a pixel
        # count says the value is less certain than it is.  A NumPy integer
        # is read as the integer it is, never through a float: an int64 has
        # digits a double cannot hold, and 9007199254740993 shown as
        # 9007199254740992 is a count the dataset never held.
        return Decimal(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    try:
        return Decimal(repr(number))
    except InvalidOperation:  # pragma: no cover - repr of a float always parses
        return None


def prefix_for(values: ArrayLike, unit: UnitLike, registry: UnitRegistry | None = None) -> Prefix:
    units = registry or DEFAULT_UNITS
    resolved = units.resolve(unit)
    numbers = [str(decimal) for decimal in (decimal_of(value) for value in np.atleast_1d(np.asarray(values, dtype=object)).ravel())
               if decimal is not None]
    with units._lock:
        exponent = units._call("units.prefix", values=numbers, unit=_unit_tree(resolved))
    return _PREFIX_BY_EXPONENT[int(exponent)]


def figure_padded(value: object, widest: object) -> str:
    """``value`` written in as many characters as ``widest``, padded on the left
    with figure spaces (U+2007, a digit's width in a font with tabular digits).

    For a count that climbs while it is read -- shot 9 of 200, then 10 -- so
    whatever follows it on the line stands still.
    """

    return str(value).rjust(len(str(widest)), "\u2007")


def format_quantity(value: object, unit: UnitLike = "1", *, prefix: Prefix | None = None,
                    registry: UnitRegistry | None = None) -> str:
    units = registry or DEFAULT_UNITS
    resolved = units.resolve(unit)
    number = decimal_of(value)
    if number is None:
        return f"{value} {resolved.symbol}".strip()
    step = None if prefix is None else {"symbol": prefix.symbol, "exponent": prefix.exponent, "accepts": list(prefix.accepts)}
    with units._lock:
        return units._call("units.format", value=str(number), unit=_unit_tree(resolved), prefix=step)


def parse_quantity(text: str, unit: UnitLike = "1", *, registry: UnitRegistry | None = None) -> float:
    units = registry or DEFAULT_UNITS
    resolved = units.resolve(unit)
    if not isinstance(text, str):
        raise UnitError("a typed quantity must be text")
    with units._lock:
        return float(units._call("units.parse", text=text, unit=_unit_tree(resolved)))


__all__ = [
    "DEFAULT_UNITS",
    "Decibel",
    "Offset",
    "PREFIXES",
    "Prefix",
    "Scaled",
    "Unit",
    "UnitError",
    "UnitLike",
    "UnitRegistry",
    "figure_padded",
    "format_quantity",
    "parse_quantity",
    "prefix_for",
    "resolve_unit",
]
