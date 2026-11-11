"""Small, headless authoring-schema contracts shared by devices and nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Any, Callable, Mapping

from zlc_data.units import resolve_unit


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and bool(left == right)


def condition_enabled(
    enabled_when: tuple[str, tuple[Any, ...]] | None,
    values: Mapping[str, Any],
) -> bool:
    """One condition for editable fields and their active run dependencies."""
    if enabled_when is None:
        return True
    controller, offered = enabled_when
    return any(_typed_equal(values.get(controller), value) for value in offered)


@dataclass(frozen=True, slots=True)
class AuthoringChoice:
    """One owner-declared value and the human label that explains it."""

    value: Any
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("authoring choice label must be non-empty")
        try:
            hash(self.value)
        except TypeError as error:
            raise TypeError("authoring choice value must be immutable") from error


@dataclass(frozen=True)
class AuthoringField:
    name: str
    value_type: str
    label: str
    default: Any = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[AuthoringChoice, ...] = ()
    #: ``(field name, values)``: this field is editable only while that field
    #: holds one of those values; otherwise it does not participate in run
    #: validation or build arguments. It is still SHOWN -- an option that appears
    #: and disappears as another is touched is a moving target, and an
    #: operator cannot see what a setting would offer before choosing it.
    enabled_when: tuple[str, tuple[Any, ...]] | None = None
    #: The node computes this field from the rest of the draft and its
    #: resources -- a calibration's camera exposure from its windows.  It is
    #: SHOWN, never edited, and is whatever the node's ``resolve_defaults``
    #: says at every finalization; the draft holds nothing for it.  The
    #: description is what the operator reads as the reason it is not theirs.
    derived: bool = False
    #: The unit this field's value is expressed in, or None for a bare
    #: number.  Declared by the field's owner because it is a fact about the
    #: KNOB -- an RF frequency is hertz whoever reads it -- and a scan axis
    #: built over the field publishes this unit into its dataset column.
    unit: str | None = None
    #: What the field wants written, for the operator: the syntax of an
    #: expression, the shape of a pair.  Shown where the value is typed.
    description: str = ""
    #: What leaving an optional field blank DOES -- "camera's current", "full
    #: sensor" -- shown in the empty box in place of a bare "(optional)".
    blank_label: str = ""
    #: For a ``rows`` field: the fields of one row.  Its value is a tuple
    #: of rows, each a mapping of these names, built one row at a time --
    #: a derive's variables -- and every row is projected by these fields
    #: exactly as a whole draft is by its schema, a column of rows included
    #: (a variable's terms), its problems located row by row.
    columns: tuple["AuthoringField", ...] = ()
    #: Show ``choices`` side by side as one segmented control rather than a
    #: popup.  The owner's call, for a closed set it declares with labels
    #: short enough to sit on one line (it declares those labels too); a
    #: form never infers it from how many choices there are.
    segmented: bool = False
    #: How far one notch of the wheel, an arrow key or the stepper moves the
    #: number, in the field's unit, or None for the control's own 1.  A fact
    #: about the KNOB, like its unit and bounds: a camera exposure of 0.02 s
    #: stepped by 1 s became fifty times longer in one notch.
    step: float | None = None
    #: The section of the node's form this field stands in ("Imaging",
    #: "Movement"): a form puts the title where the section changes.  Empty
    #: for a field that needs none.
    group: str = ""
    #: A file path edited with Browse; the stored value remains a string.
    file_filter: str = ""
    #: The unit family a view shows this number in by default, when its owner
    #: prefers one to ``unit``'s own -- a function generator's amplitude, a
    #: power in dBm, read in Vpp -- or None for ``unit``'s.  Display only:
    #: commands, bounds and records stay in ``unit``, and a view converts
    #: through the owner (an RF level into its channel's load).
    display_unit: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.value_type or not self.label:
            raise ValueError("authoring fields require name, value_type, and label")
        if self.step is not None:
            if str(self.value_type) not in {"int", "float"} or self.choices:
                raise ValueError(f"only a number field can declare a step, not {self.name!r}")
            if (
                not isinstance(self.step, (int, float))
                or isinstance(self.step, bool)
                or not math.isfinite(self.step)
                or self.step <= 0
            ):
                raise ValueError(f"field {self.name!r} step must be a positive finite number")
            if str(self.value_type) == "int" and not isinstance(self.step, int):
                raise TypeError(f"int field {self.name!r} step must be a whole number")
        if not isinstance(self.segmented, bool):
            raise TypeError("authoring field segmented must be bool")
        if self.segmented and not self.choices:
            raise ValueError(f"segmented field {self.name!r} needs its choices")
        if not isinstance(self.description, str) or not isinstance(self.blank_label, str) or not isinstance(self.group, str):
            raise TypeError("authoring field description, blank_label and group must be text")
        if self.derived and (self.required or self.enabled_when is not None):
            raise ValueError(
                f"derived field {self.name!r} is the node's to fill: it is neither "
                "required of the operator nor conditionally editable"
            )
        columns = tuple(self.columns)
        if str(self.value_type) == "rows":
            if not columns:
                raise ValueError(f"rows field {self.name!r} needs the columns of a row")
            if any(not isinstance(column, AuthoringField) for column in columns):
                raise TypeError("authoring field columns must contain AuthoringField values")
            names = tuple(column.name for column in columns)
            if len(set(names)) != len(names):
                raise ValueError(f"rows field {self.name!r} column names must be unique")
            if (
                self.choices
                or self.unit is not None
                or self.minimum is not None
                or self.maximum is not None
            ):
                raise ValueError(
                    f"rows field {self.name!r} takes neither choices, bounds nor a unit"
                )
            default = () if self.default is None else self.default
            if isinstance(default, (str, bytes, Mapping)) or not isinstance(
                default, (list, tuple)
            ) or any(not isinstance(row, Mapping) for row in default):
                raise TypeError(f"rows field {self.name!r} default must be rows")
            object.__setattr__(self, "default", tuple(dict(row) for row in default))
        elif columns:
            raise ValueError(f"field {self.name!r} is not rows and declares columns")
        object.__setattr__(self, "columns", columns)
        if self.unit is not None:
            if not isinstance(self.unit, str) or not self.unit.strip():
                raise ValueError("authoring field unit must be non-empty text or None")
            # HERE, where it is written, and not where it is first drawn.  A
            # unit nobody had registered used to travel all the way to the
            # plot before anything noticed: rf.rigol_dg4000 declared dBm, the
            # plot contract had never heard of it, and the first scan of a
            # signal generator's power died with "unknown unit 'dBm'" on a
            # panel, hours from the line that wrote it.
            resolve_unit(self.unit.strip())
        if self.display_unit is not None:
            # A family the value can be read in: the same quantity, said another way.
            if self.unit is None or resolve_unit(str(self.display_unit).strip()).dimension != resolve_unit(self.unit.strip()).dimension:
                raise ValueError(f"field {self.name!r} display unit {self.display_unit!r} is not a spelling of its unit {self.unit!r}")
        if self.enabled_when is not None:
            controller, values = self.enabled_when
            if not str(controller).strip():
                raise ValueError("enabled_when needs the name of a field")
            if not tuple(values):
                raise ValueError("enabled_when needs at least one enabling value")
            object.__setattr__(
                self,
                "enabled_when",
                (str(controller), tuple(values)),
            )
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("authoring field minimum exceeds maximum")
        # A default that violates the field's own window poisons every draft
        # built from it (a "no value yet" encoded as an out-of-range number).
        # Vacancy is None; a numeric default must obey the declared bounds.
        if isinstance(self.default, (int, float)) and not isinstance(self.default, bool):
            if self.minimum is not None and self.default < self.minimum:
                raise ValueError(
                    f"authoring field {self.name!r} default {self.default!r} "
                    "is below its own minimum; a vacancy is None, not a number"
                )
            if self.maximum is not None and self.default > self.maximum:
                raise ValueError(
                    f"authoring field {self.name!r} default {self.default!r} "
                    "is above its own maximum; a vacancy is None, not a number"
                )
        choices = tuple(self.choices)
        if any(not isinstance(choice, AuthoringChoice) for choice in choices):
            raise TypeError("authoring field choices must contain AuthoringChoice values")
        values = tuple(choice.value for choice in choices)
        if any(
            any(_typed_equal(value, prior) for prior in values[:index])
            for index, value in enumerate(values)
        ):
            raise ValueError("authoring choice values must be unique")
        if choices and self.default is not None and not any(
            _typed_equal(self.default, value) for value in values
        ):
            raise ValueError("authoring field default is not one of its choices")
        object.__setattr__(self, "choices", choices)


class TuneRefused(ValueError):
    """The instrument would not take this value, and nothing was written.

    A grid step it does not land on, a number that is not finite, a name
    this firmware does not have: the knob is still where it was, so the
    reading this session holds is still true.  That is what tells this
    apart from a write that REACHED the instrument and then lost its
    readback, where the value really is unknown and the honest answer is to
    have none.

    It lives here, beside the field a device declares and the call that
    moves it, because a refusal is a fact about TUNING and not about any
    one device family.  It began in the RF contract, where the first
    quantized instrument needed it, and a waveform device that wanted to
    say the same thing could not reach it across the family boundary --
    which is how a second, identical exception gets written.
    """


@dataclass(frozen=True)
class TunableField:
    """One runtime setting: stable form metadata beside current device truth."""

    metadata: AuthoringField
    current: Any
    live_write: bool
    dependency_group: tuple[str, ...]
    #: The instrument's own range for this knob, as the device reports it,
    #: or None when the device states none (a switch, a policy edge).
    #: ``metadata.minimum``/``maximum`` say what may be COMMANDED -- the
    #: tighter, side by side, of these limits and any bench policy window --
    #: and Device Control shows the limits beside them, read-only, so an
    #: operator can see which of the two is fencing a knob.
    device_limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, AuthoringField):
            raise TypeError("tunable metadata must be an AuthoringField")
        if type(self.live_write) is not bool:
            raise TypeError("tunable live_write must be bool")
        if self.device_limits is not None:
            limits = tuple(self.device_limits)
            if len(limits) != 2 or any(isinstance(edge, bool) for edge in limits):
                raise TypeError("tunable device_limits must be a (low, high) pair")
            low, high = (float(edge) for edge in limits)
            if not (math.isfinite(low) and math.isfinite(high)) or low > high:
                raise ValueError("tunable device_limits must be a finite ordered pair")
            for bound in (self.metadata.minimum, self.metadata.maximum):
                if bound is not None and not low <= float(bound) <= high:
                    raise ValueError(
                        "tunable bounds must lie inside the device's own limits"
                    )
            object.__setattr__(self, "device_limits", (low, high))
        group = tuple(str(name).strip() for name in self.dependency_group)
        if (
            not group
            or any(not name for name in group)
            or len(set(group)) != len(group)
            or self.metadata.name not in group
        ):
            raise ValueError(
                "tunable dependency_group must uniquely include its own field"
            )
        # A READING IS NOT AN INPUT.  The window says what may be
        # commanded -- ``tune`` and every draft projection still refuse
        # outside it -- while ``current`` is what the device answered, and
        # a device idling outside bench policy is exactly the state its
        # operator has to be shown.  Judging the reading by the window
        # made that state unreportable: the field refused its own device,
        # so a driver's only way to open at all was to move the knob.  The
        # reading is still projected for TYPE, which is what makes it the
        # same kind of value the field speaks in.
        reading = replace(
            self.metadata, default=None, minimum=None, maximum=None
        )
        current = AuthoringSchema((reading,)).project_values(
            {reading.name: self.current}
        )[reading.name]
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "dependency_group", group)


def is_tunable(device: object) -> bool:
    """Whether ``device`` speaks the whole tunable quartet.

    ``tunable_fields`` declares its knobs, ``tune`` moves one,
    ``tunable_values`` reads them all and ``settings_provenance`` names the
    session and epoch a reading belongs to -- the surface ``RfSource``
    spells out in ``devices/rf/contract.py``.  A scan records the values,
    Device Control checks the provenance and the bench fabric serves all
    four, so a device with only part of the set is tunable nowhere: five
    separately spelled checks used to offer it as a scan axis, then raise
    on its Control page.
    """

    return all(
        callable(getattr(device, name, None))
        for name in ("tunable_fields", "tune", "tunable_values", "settings_provenance")
    )


# Reading and tuning a knob in an operator's unit is the SDK's (its `fields`
# and `tune` ops serve Device Control); the package speaks the same calls.
from zlc_plugin.tunable import (  # noqa: E402
    convert_tunable_value, read_tunable_in_unit, refresh_tunable_fields, tune_in_unit,
)


@dataclass(frozen=True)
class AuthoringSchema:
    fields: tuple[AuthoringField, ...] = ()
    validator: Callable[[Mapping[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if any(not isinstance(field, AuthoringField) for field in fields):
            raise TypeError("fields must contain AuthoringField values")
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise ValueError("authoring field names must be unique")
        object.__setattr__(self, "fields", fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def is_enabled(self, name: str, values: Mapping[str, Any]) -> bool:
        field = next((field for field in self.fields if field.name == name), None)
        if field is None:
            return False
        if field.enabled_when is None:
            return True
        effective = {field.name: values.get(field.name, field.default) for field in self.fields}
        return condition_enabled(field.enabled_when, effective)

    def project_values(
        self,
        values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert one raw draft and validate it through its sole domain owner."""

        return self._project(values, require_complete=True)

    def draft_values(
        self,
        values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Project an EDITABLE draft: everything checks except completeness.

        Coercion, grids, bounds and choices apply exactly as at build time,
        but a required field may stand empty -- the form exists to fill it,
        and a type whose required field has no sensible default (a VISA
        resource) must still be addable.  The cross-field validator waits
        until every required field is present, because it is written
        against complete value sets.  ``project_values`` at Init keeps
        refusing anything incomplete, per device, by name.
        """

        return self._project(values, require_complete=False)

    def _project(
        self,
        values: Mapping[str, Any] | None,
        *,
        require_complete: bool,
    ) -> dict[str, Any]:
        from zlc_plugin.declare import field_json
        from zlc_plugin.services import call

        answer = call("fields.project", fields=[field_json(field) for field in self.fields],
                      values=dict(values or {}), complete=require_complete)
        _native_projection_error(answer)
        result = {field.name: (_projection_shape(field, answer["values"][field.name])
                              if answer["enabled"][field.name] else answer["values"][field.name]) for field in self.fields}
        if self.validator is not None and answer["complete"]:
            self.validator(result)
        return result


def _native_projection_error(answer):
    error = answer.get("error")
    if error is not None:
        kind = TypeError if error["type"] == "TypeError" else ValueError
        raise kind(error["message"])


def _projection_shape(field, value):
    # Python collection conventions are an adapter detail; rules live in Rust.
    if value is None:
        return None
    if field.value_type in ("numeric_tuple", "text_tuple"):
        return tuple(value)
    if field.value_type == "rows":
        return tuple({column.name: (_projection_shape(column, row[column.name])
                                    if condition_enabled(column.enabled_when, row) else row[column.name]) for column in field.columns}
                     for row in value)
    return value


def project_field(
    field: AuthoringField, value: object, *, require_complete: bool = False
) -> object:
    """``value`` as ``field`` keeps it: its type's coercion, its choices, its bounds.

    One field's own rule, and nothing about completeness or across fields:
    a draft may stand incomplete or fail the schema's validator, but never
    holds a value its field refuses.  A refusal raises one sentence naming
    the field by its label.  ``require_complete`` reaches only the rows of a
    rows field.
    """

    from zlc_plugin.declare import field_json
    from zlc_plugin.services import call

    answer = call("fields.value", field=field_json(field), value=value, complete=require_complete)
    _native_projection_error(answer)
    return _projection_shape(field, answer["value"])


__all__ = ["AuthoringChoice", "AuthoringField", "AuthoringSchema", "TunableField",
           "TuneRefused", "is_tunable", "refresh_tunable_fields", "read_tunable_in_unit",
           "tune_in_unit", "convert_tunable_value", "condition_enabled", "project_field"]
