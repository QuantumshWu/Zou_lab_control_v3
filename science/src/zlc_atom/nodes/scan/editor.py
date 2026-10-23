"""What the scan plan editor shows, as data (the node's ``editor_facts``).

v2 drew the plan as a Qt table inside this package; v3's editor is the UI's
(apps/desktop/ui/src/views/node_editors/scan_plan.ts) and holds no scan rule:
which knobs there are and under what they hang, what the plan amounts to, and
what the pulse's API values say are all answered here, from the draft, by the
same functions the build binds with.  The editor writes back exactly two
fields, ``plan`` (its axes, as the JSON document) and ``api_values`` (one
``<field_id> = <number>`` a line, only what differs from the pulse).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict

from zlc_data.units import DEFAULT_UNITS, UnitError, format_quantity
from zlc_plugin.tunable import convert_tunable_value

from .plan import (
    API_PARAM_FAMILY,
    DEVICE_PARAM_FAMILY,
    MANUAL_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    ScanPlan,
    ScanPort,
    ambiguous_api_value,
    api_overrides_from_authored,
    api_overrides_to_authored,
    api_scan_ports_for,
    device_port_parts,
    hardware_scan_ports_for,
    host_advanced_port,
    parse_scan_values,
    plan_input_rows,
    port_group,
    port_leaf,
    unscannable_api_ports,
)

#: The fields the editor writes; the Edit page's plain form leaves them out.
SCAN_EDITOR_FIELDS = ("plan", "api_values")
_FAMILIES = {"pulse": PULSE_PARAM_FAMILY, "api": API_PARAM_FAMILY, "device": DEVICE_PARAM_FAMILY,
             "manual": MANUAL_PARAM_FAMILY}


def _counted(count: int, one: str, many: str | None = None) -> str:
    return f"{count} {one if count == 1 else many or one + 's'}"


def _spoken_list(words: Sequence[str]) -> str:
    """``words`` as a sentence lists them: "power", "power and angle"."""

    words = [str(word) for word in words]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _families(symbols) -> dict:
    """The family each spelling belongs to (v2 unit_choice_tree's trunks: W, dBm, Vpp)."""

    families = {}
    for symbol in symbols:
        try:
            families[symbol] = DEFAULT_UNITS.family_of(symbol).symbol
        except UnitError:
            families[symbol] = symbol
    return families


def _ladder(unit: str) -> list:
    """The spellings an axis on a port may be authored in that are exact
    decimal shifts of the port's unit (its prefix rungs: s, ms, µs, ns), each
    with its decade against that unit; empty when it has only its own.  A
    value converts as a shift of its decimal digits, never a binary round
    trip (UnitRegistry.convert_decimal)."""

    try:
        family, prefix = DEFAULT_UNITS.family_and_prefix(unit)
        rungs = [[symbol, rung.exponent - prefix.exponent] for symbol in DEFAULT_UNITS.display_choices(unit)
                 for rung_family, rung in (DEFAULT_UNITS.family_and_prefix(symbol),) if rung_family == family]
    except UnitError:
        return []
    return rungs if len(rungs) > 1 else []


def _rounded(value: float) -> float:
    """A converted number as a box can show it: twelve significant digits."""

    return float(f"{float(value):.12g}")


class _Spellings:
    """A device knob's spellings past its unit's decimal shifts (v2 converted
    a device axis's unit: dBm and mW, Vpp, Vrms), converted through the
    device's own law when it has one -- an RF amplitude into its channel's
    load -- else by the quantity's own; one conversion a unit family, its
    prefix rungs shifted from it.  A slot or an API parameter has none."""

    def __init__(self, tunables: Mapping[str, object] | None) -> None:
        self._tunables = dict(tunables or {})
        self._memo: dict[tuple, tuple] = {}
        self._limits: dict[str, dict] = {}

    def convert(self, port: ScanPort, values: Sequence[float], source: str, target: str) -> list[float]:
        family, _prefix = DEFAULT_UNITS.family_and_prefix(target)
        key, field = device_port_parts(port.port)
        token = (port.port, tuple(float(value) for value in values), source, family.symbol)
        if token not in self._memo:
            self._memo[token] = tuple(convert_tunable_value(self._tunables.get(key), field, token[1], source, family.symbol))
        return [_rounded(DEFAULT_UNITS.convert_decimal(value, family.symbol, target)) for value in self._memo[token]]

    def limits(self, port: ScanPort) -> dict[str, list[float]]:
        """The port's limits in each such spelling the device converts into
        (one its policy refuses, or that fails to convert, is not offered)."""

        if port.port in self._limits:
            return self._limits[port.port]
        limits: dict[str, list[float]] = {}
        if port.port.startswith(DEVICE_PARAM_FAMILY):
            try:
                choices = DEFAULT_UNITS.display_choices(port.unit)
            except UnitError:
                choices = ()
            shifts = {symbol for symbol, _decade in _ladder(port.unit)} | {port.unit}
            for symbol in (symbol for symbol in choices if symbol not in shifts):
                try:
                    limits[symbol] = sorted(self.convert(port, (port.lo, port.hi), port.unit, symbol))
                except Exception:  # noqa: BLE001 -- the device's refusal: that spelling is not offered
                    continue
        self._limits[port.port] = limits
        return limits


def _port(port: ScanPort, spellings: _Spellings) -> dict:
    """One knob with the branch it hangs under (a device's under the device),
    its name within it and the spellings its axis may be authored in: each
    with its decade against the port's unit, or None and the port's limits in
    it where no decimal shift reaches it."""

    limits = spellings.limits(port)
    ladder = _ladder(port.unit) or ([[port.unit, 0]] if limits else [])
    units = [*ladder, *([symbol, None] for symbol in limits)]
    group = port.label.removesuffix(f".{port_leaf(port.port)}") if port.port.startswith(DEVICE_PARAM_FAMILY) else port_group(port.port)
    return {**asdict(port), "group": group, "leaf": port_leaf(port.port),
            "units": units, "families": _families(symbol for symbol, _decade in units), "limits": limits}


def _conversions(rows: Sequence[Mapping], ports: Sequence[ScanPort], spellings: _Spellings) -> dict:
    """Each device axis's values in every other spelling its knob offers past
    the decimal shifts, by port (v2 ``_show_converted``): both banks, the
    Values list kept inside the knob's limits in that spelling."""

    by_port = {port.port: port for port in ports}
    converted: dict[str, dict] = {}
    for row in rows:
        port = by_port.get(row["port"])
        limits = {} if port is None else spellings.limits(port)
        if not limits:
            continue
        try:
            listed = parse_scan_values(row["value_text"]) if str(row["value_text"]).strip() else ()
            table = {}
            for symbol in [*(symbol for symbol, _decade in _ladder(port.unit) or [[port.unit, 0]]), *limits]:
                if symbol == row["unit"]:
                    continue
                low, high = limits.get(symbol) or sorted(spellings.convert(port, (port.lo, port.hi), port.unit, symbol))
                table[symbol] = {"values": spellings.convert(port, row["values"], row["unit"], symbol),
                                 "explicit": [min(high, max(low, value)) for value in spellings.convert(port, listed, row["unit"], symbol)] if listed else []}
        except Exception:  # noqa: BLE001 -- the row's own problem is said under it; a refused conversion offers nothing
            continue
        converted[row["port"]] = table
    return converted


def _rows(plan: object) -> tuple[dict, ...]:
    try:
        return plan_input_rows(plan)
    except (TypeError, ValueError):
        return ()


def _summary(plan: object) -> tuple[str, str]:
    """What the plan is, or why it cannot be read (v2 ``_refresh_summary``)."""

    try:
        rows = plan_input_rows(plan) if str(plan or "").strip() else ()
        parsed = ScanPlan.from_tree(plan) if rows else None
    except (TypeError, ValueError) as error:
        return "", str(error)
    if parsed is None:
        return ("No axes yet. Each row is one axis, outermost first; every point plays the pulse and "
                "captures one measurement."), ""
    manual = [axis for axis in parsed.axes if axis.port.startswith(MANUAL_PARAM_FAMILY)]
    device = [axis for axis in parsed.axes if axis.port.startswith(DEVICE_PARAM_FAMILY)]
    # The run stops once per setting of the hand's axes, not once per point.
    stops = 1
    for axis in manual:
        stops *= len(axis.values)
    by_hand = "" if not manual else (
        f" The run stops {'once' if stops == 1 else _counted(stops, 'time')} for you to set "
        f"{_spoken_list([axis.port[len(MANUAL_PARAM_FAMILY):] for axis in manual])} by hand.")
    calls = 1
    for axis in device:
        calls *= len(axis.values)
    by_device = "" if not device else (
        f" {_counted(calls, 'device setting')} {'is' if calls == 1 else 'are'} applied and read back between fires.")
    shape = " × ".join(str(count) for count in parsed.shape)
    return (f"{_counted(len(parsed.axes), 'axis', 'axes')}, {shape} = {_counted(parsed.point_count, 'point')}, "
            "outermost first; each point resolves the template, plays it, and captures one measurement."
            f"{by_hand}{by_device}"), ""


def _api(pulse: Mapping[str, object], rows: Sequence[Mapping], text: object) -> dict:
    """The pulse's API values this run sets (v2 ``_reconcile_values``): one
    row per field the board does not sweep -- locked, with the reason, while
    an ``api:`` axis walks it -- each with the spellings its unit can be
    read in, and every value the draft keeps that no row shows, as a line of
    its own (a name this pulse does not declare, one a board axis sweeps, or
    a label two of its fields answer to).

    ``tidy`` is the text re-read against THIS pulse (v2 ``_rescope_values``)
    when that changes it, else None: a value equal to the pulse's own is no
    override and is dropped, a label is written as the field id it names, and
    everything else is kept.  Text that does not read decides nothing."""

    fields = list(pulse.get("api_fields") or ())
    scanned = {row["port"][len(PULSE_PARAM_FAMILY):] for row in rows if row["port"].startswith(PULSE_PARAM_FAMILY)}
    walked = {row["port"][len(API_PARAM_FAMILY):] for row in rows if row["port"].startswith(API_PARAM_FAMILY)}
    authored = {str(field["field_id"]): (float(field["value"]), str(field["unit"])) for field in fields}
    labels = {str(field["field_id"]): str(field["label"]) for field in fields}
    named: dict[str, list[str]] = {}
    for field_id, label in labels.items():
        named.setdefault(label, []).append(field_id)
    try:
        written = api_overrides_from_authored(text)
        readable = True
    except ValueError:
        written, readable = {}, False  # the draft's issue says why; nothing to show yet
    # A label one field answers to is that field.
    overrides = {named[name][0] if name not in authored and len(named.get(name, ())) == 1 else name: value
                 for name, value in written.items()}
    shown = []
    for field in fields:
        name = str(field["field_id"])
        if name in scanned:
            continue
        code = bool(field.get("is_dac"))
        low = field.get("limit_lo") if field.get("limit_lo") is not None else field.get("lo")
        high = field.get("limit_hi") if field.get("limit_hi") is not None else field.get("hi")
        number = overrides.get(name, authored[name][0])
        shown.append({
            "name": name, "label": labels[name], "kind": "integer" if code else "number",
            "unit": "" if code else authored[name][1], "units": [] if code else _ladder(authored[name][1]),
            "families": {} if code else _families(symbol for symbol, _decade in _ladder(authored[name][1])),
            "description": name,
            "minimum": None if low is None else (int(low) if code else float(low)),
            "maximum": None if high is None else (int(high) if code else float(high)),
            "authored": int(authored[name][0]) if code else authored[name][0],
            "value": int(number) if code else number,
            "unavailable_reason": (
                "swept by the scan plan; the value shown is kept for when its axis is taken out of the plan"
                if name in walked and name in overrides
                else "swept by the scan plan; take its axis out of the plan to set a value for the run"
                if name in walked else ""),
        })
    keys = {row["name"] for row in shown}
    parked = []
    for name, value in overrides.items():
        if name in keys:
            continue
        if name not in authored and len(named.get(name, ())) > 1:
            parked.append({"name": name, "text": ambiguous_api_value(name, named[name]), "problem": True})
            continue
        number = format_quantity(value, "1")
        if name not in authored:
            text_line = f"not in this pulse: {name} = {number} (kept, not applied)"
        else:
            unit = authored[name][1]
            quantity = number if unit in ("value", "count") else f"{number} {unit}"
            text_line = (f"swept by the scan plan: {labels.get(name, name)} = {quantity} "
                         "(kept for when its axis is taken out of the plan)")
        parked.append({"name": name, "text": text_line, "problem": False})
    parts = []
    free = [row["name"] for row in shown if row["name"] not in walked]
    if free:
        set_here = sum(1 for name in free if name in overrides and overrides[name] != authored[name][0])
        parts.append(f"{set_here} of {len(free)} set for this run"
                     + ("; the rest run what the pulse carries." if set_here < len(free) else "."))
    swept = (scanned | walked) & set(authored)
    if swept:
        parts.append(f"Swept by the plan: {', '.join(sorted(labels.get(name, name) for name in swept))}.")
    kept = {name: value for name, value in overrides.items() if name not in authored or value != authored[name][0]}
    return {"fields": shown, "overrides": [[name, value] for name, value in overrides.items()],
            "parked": parked, "note": " ".join(parts),
            "tidy": api_overrides_to_authored(kept) if readable and kept != written else None}


def scan_editor_facts(values: Mapping[str, object], pulse: Mapping[str, object] | None,
                      device_ports: Sequence[ScanPort], tunables: Mapping[str, object] | None = None) -> dict:
    """The scan plan editor's model for one draft: the knobs (the pulse's
    slots and API parameters, then the devices' scannable fields), the API
    parameters the pulse withholds and why, what the plan amounts to, and the
    API values section (None until a pulse is chosen).  ``tunables`` are the lent
    devices: a knob's other spellings are converted through them."""

    spellings = _Spellings(tunables)
    ports = [*hardware_scan_ports_for(pulse), *api_scan_ports_for(pulse)] if pulse is not None else []
    ports.extend(device_ports)
    summary, problem = _summary(values.get("plan"))
    # An axis on a knob nobody offers now (a slot renamed, a binding taken off,
    # a device closed) keeps saying what was authored, under its owner.
    offered = {port.port for port in ports}
    unoffered = {}
    for row in _rows(values.get("plan")):
        name = row["port"]
        if name in offered or name.startswith(MANUAL_PARAM_FAMILY):
            continue
        try:
            unoffered[name] = {"group": port_group(name), "leaf": port_leaf(name)}
        except ValueError:
            unoffered[name] = {"group": "unavailable", "leaf": name}
    return {
        "editor": "scan_plan",
        "manages": list(SCAN_EDITOR_FIELDS),
        "families": dict(_FAMILIES),
        # The families the HOST walks between fires: their rows stand outside the board's.
        "host_advanced": [prefix for prefix in _FAMILIES.values() if host_advanced_port(prefix + "x")],
        "ports": [_port(port, spellings) for port in ports],
        "conversions": _conversions(_rows(values.get("plan")), ports, spellings),
        "withheld": {} if pulse is None else unscannable_api_ports(pulse),
        "unoffered": unoffered,
        "summary": summary,
        "problem": problem,
        "api": None if pulse is None else _api(pulse, _rows(values.get("plan")), values.get("api_values")),
    }


__all__ = ["SCAN_EDITOR_FIELDS", "scan_editor_facts"]
