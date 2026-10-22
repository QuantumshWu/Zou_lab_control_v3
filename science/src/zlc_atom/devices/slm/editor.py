"""The SLM Editor (v2's "<role> SLM Editor" window) as the SLM's own commands.

One open phase-only SLM's authoring draft -- its target, the pattern phase
solved for it, the input pupil and the operator wavefront -- and every
action v2's window had on it: presets, painting, the pupil and wavefront
layers, solve, Send, Adopt, and the target and Science Context files.  It
lives in the SLM's own plugin process beside the adapter it sends to, so a
Send is the device's own lane, not a second owner.  The window and an MCP
client are two clients of the same ``editor_*`` commands; neither holds a
rule of the editor's.

The core draws the three images (target, science phase, wavefront phase):
whenever the draft changes the editor sends one ``device.notice`` named
``slm_editor`` with its state and the planes that changed (a solve ends
long after the command that asked for it).  The solve is v2's latest-only
worker: a newer request stops the one under way.  The device is read every
100 ms while the editor is open, as v2's window polled it, and only a
change is told.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np

from zlc_atom.authoring import AuthoringChoice, AuthoringField
from zlc_plugin import Command, FileParam, notifier

from .phase import canonical_phase
from .solver import (
    compose_science_phase, default_grid_spacing, freeze_pattern_phase, imported_target, load_science_context, load_target,
    preset_checkerboard, preset_flat_top, preset_gaussian, preset_grid, preset_text, save_science_context,
    save_target, science_operator_wavefront, science_pupil_fields, solve_phase, validate_target,
)

_ZERNIKE = (
    ("defocus", "Noll Z4 · Defocus", "√3(2r²−1)"),
    ("astig_oblique", "Noll Z5 · Oblique astig", "2√6xy"),
    ("astig_vertical", "Noll Z6 · Vertical astig", "√6(x²−y²)"),
    ("coma_y", "Noll Z7 · Y coma", "√8y(3r²−2)"),
    ("coma_x", "Noll Z8 · X coma", "√8x(3r²−2)"),
    ("trefoil_y", "Noll Z9 · Y trefoil", "√8y(3x²−y²)"),
    ("trefoil_x", "Noll Z10 · X trefoil", "√8x(x²−3y²)"),
    ("spherical", "Noll Z11 · Spherical", "√5(6r⁴−6r²+1)"),
)
PRESETS = ("Grid", "Checkerboard", "Gaussian", "Flat Top", "Text")
#: The Grid an Editor opens on (v2: at preset_grid's own spacing) and its Preset offers.
_FIRST_GRID = (5, 7)
_PRESET_SHOWN = {
    "Grid": {"rows", "columns", "spacing_y", "spacing_x", "intensity"},
    "Checkerboard": {"rows", "columns", "spacing_y", "spacing_x", "intensity"},
    "Gaussian": {"radius_y", "radius_x", "intensity"},
    "Flat Top": {"radius_y", "radius_x", "edge", "intensity"},
    "Text": {"text_spacing", "atom_budget", "intensity"},
}
_PAINT_MODES = (("toggle", "Toggle"), ("brush", "Brush"), ("erase", "Erase"))
#: One plot is half of the Editor's row: the 490 x 357 logical viewport of
#: the 2x2 preset; settings live on their own tabs instead of shrinking it.
PLOT_SIZE = "2x2"
_NOTE = ("Steering X/Y adds a full-raster carrier. A physical Fourier-plane stop/aperture is still required to "
         "reject zero order. Coefficients are Noll-normalized waves RMS on the configured unit disk.")
#: The images the core draws, read by their shape, not off a colour scale:
#: none draws a side distribution or colorbar beside it.
_PLOTS = (("target", "Target intensity"), ("phase", "Science phase (pre-correction)"), ("wavefront", "Wavefront phase (rad)"))
#: The window's file actions (v2 _choose): save or open, the offered name, the filter, the
#: caption (v2 name_words of its action).
_FILES = {
    "load_target": (False, "target.json", "SLM target (*.json)", "Load target"),
    "save_target": (True, "target.json", "SLM target (*.json)", "Save target"),
    "import_target": (False, "", "Array or image (*.npy *.png *.tif *.tiff *.bmp)", "Import"),
    "load_context": (False, "science-context.npz", "SLM Science Context (*.npz)", "Load context"),
    "save_context": (True, "science-context.npz", "SLM Science Context (*.npz)", "Save context"),
}


def _number(name: str, label: str, value: float, minimum: float, maximum: float, *, integral: bool = False,
            decimals: int = 3, step: float = 1.0, description: str = "") -> dict[str, Any]:
    return {"name": name, "label": label, "type": "int" if integral else "float", "value": int(value) if integral else float(value),
            "minimum": int(minimum) if integral else float(minimum), "maximum": int(maximum) if integral else float(maximum),
            "decimals": 0 if integral else decimals, "step": step, "description": description}


def _preset_fields(shape: tuple[int, int]) -> list[dict[str, Any]]:
    """v2 _build_preset_popup: sites start 25 target pixels apart; only a
    target too small to hold the default grid that far apart gets the widest
    spacing that still holds it."""

    rows, columns = _FIRST_GRID
    spacing_y, spacing_x = default_grid_spacing(shape, _FIRST_GRID)
    fields = [
        _number("rows", "N rows", rows, 1, shape[0], integral=True),
        _number("columns", "M columns", columns, 1, shape[1], integral=True),
        _number("spacing_y", "Spacing Y", spacing_y, 1, shape[0], integral=True),
        _number("spacing_x", "Spacing X", spacing_x, 1, shape[1], integral=True),
        _number("intensity", "Intensity", 1.0, 0.0, 1000.0),
        _number("edge", "Edge", max(2, min(shape) // 32), 0.0, min(shape)),
        _number("radius_y", "Radius Y", max(1, shape[0] // 8), 1, shape[0]),
        _number("radius_x", "Radius X", max(1, shape[1] // 8), 1, shape[1]),
        _number("text_spacing", "Minimum atom spacing (target px)", 4, 1, min(shape), integral=True),
        _number("atom_budget", "Atom budget", 256, 1, min(4096, int(np.prod(shape))), integral=True),
    ]
    checker = "Base checker-cell spacing; long-row sites are 2× this distance."
    for field in fields:
        field["kinds"] = [kind for kind in PRESETS if field["name"] in _PRESET_SHOWN[kind]]
        # v2 _sync_preset_popup: what a field is called under each pattern.
        labels = {kind: field["label"] for kind in PRESETS}
        if field["name"] == "columns":
            labels["Checkerboard"] = "Sites in long row"
        if field["name"] == "spacing_x":
            labels["Checkerboard"] = "Checker cell spacing X"
            field["descriptions"] = {"Checkerboard": checker}
        if field["name"] in ("radius_y", "radius_x"):
            axis = field["label"][-1]
            labels = {kind: f"{'1/e² ' if kind == 'Gaussian' else ''}radius {axis}" for kind in PRESETS}
        field["labels"] = labels
    return fields


def _read_imported_target(path: object) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        values = np.load(source, allow_pickle=False)
    else:
        from PIL import Image
        with Image.open(source) as image:
            values = np.asarray(image.convert("F"), dtype=np.float32)
    return imported_target(values)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


class SlmEditor:
    """One SLM's draft and its latest-only solve (v2 SlmEditorControl without Qt)."""

    def __init__(self, adapter: Any, notify: Callable[[str, Any], bool], instance: str) -> None:
        self.adapter, self._notify, self.instance = adapter, notify, instance
        self.shape = tuple(int(value) for value in adapter.shape_yx)
        height, width = self.shape
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._target = preset_grid(self.shape, _FIRST_GRID)
        self._phase = canonical_phase(np.zeros(self.shape), self.shape)
        self._pattern_phase = self._phase
        self._pattern_metadata: dict[str, object] = {"source": "deterministic-seed"}
        self._phase_metadata: dict[str, object] = {"source": "authoring-draft"}
        self._system_correction: dict[str, object] | None = None
        self._phase_match: tuple[np.ndarray, np.ndarray, bool] | None = None
        self._device_state = self._read_device_state()
        command_revision, mapping_revision, _phase, receipt = self._device_state
        self._context_command_receipt = dict(receipt)
        self._draft_command_revision, self._draft_mapping_revision = command_revision, mapping_revision
        self._device_diverged = False
        self._device_error: str | None = None
        self._objective_kind = "spots"
        self._pupil_enabled = True
        self._pupil_center_xy = (0.5 * (width - 1), 0.5 * (height - 1))
        self._pupil_diameter_xy = (0.70 * height, 0.70 * height)
        self._pupil_support, self._pupil_amplitude = science_pupil_fields(self.shape, self._pupil_settings())
        self._pupil_text = self._pupil_description(True)
        #: A loaded Science Context's operator exactly as its file states it,
        #: until a wavefront control is edited: a Save right after a Load
        #: writes that value back, not a reading of it.
        self._frozen_operator: dict[str, object] | None = None
        self._operator: dict[str, Any] = {"enabled": False, "carrier_x": 0.0, "carrier_y": 0.0,
                                          **{key: 0.0 for key, *_ in _ZERNIKE}}
        self._operator_text = "Off"
        self._wavefront_phase = science_operator_wavefront(self.shape, self._pupil_settings(), self._operator_settings())
        self._preset_fields = _preset_fields(self.shape)
        self._preset = {"kind": "Grid", "text": "USTC", "values": {field["name"]: field["value"] for field in self._preset_fields}}
        self._plot_size = PLOT_SIZE
        self._request_revision = 0
        self._phase_request_revision: int | None = None
        self._spot_optimizer_state: dict[str, object] | None = None
        self._pending: tuple[int, np.ndarray, str, np.ndarray, dict[str, object] | None] | None = None
        self._running = self._closed = self._command_active = False
        self._last_error: str | None = None
        self._status = ("", "task")
        #: Each image's revision, and the one last told to the core.
        self._revisions = {"target": 1, "phase": 1, "wavefront": 1}
        self._told: dict[str, int] = {}
        self._told_state: dict[str, Any] | None = None
        self._catalog_told = False
        self._stop = threading.Event()
        threading.Thread(target=self._solving, name=f"slm-solve-{instance}", daemon=True).start()
        with self._lock:
            self._render_device_state()
            self._queue_solve()
        threading.Thread(target=self._polling, name=f"slm-poll-{instance}", daemon=True).start()

    # ------------------------------------------------------------------ facts
    def _pupil_settings(self) -> dict[str, object]:
        return {"enabled": self._pupil_enabled, "center_xy": list(self._pupil_center_xy), "diameter_xy": list(self._pupil_diameter_xy)}

    def _operator_settings(self) -> dict[str, object]:
        if self._frozen_operator is not None:
            return deepcopy(self._frozen_operator)
        return {"enabled": self._operator["enabled"], "carrier_waves_xy": [self._operator["carrier_x"], self._operator["carrier_y"]],
                "zernike_noll_waves_rms": {key: self._operator[key] for key, *_ in _ZERNIKE}}

    def _pupil_description(self, enabled: bool) -> str:
        if not enabled:
            return "Off · uniform full raster"
        dx, dy = self._pupil_diameter_xy
        return f"On · Gaussian · 1/e² intensity diameter {dx:g} × {dy:g} px"

    @property
    def solver_idle(self) -> bool:
        return not self._running and self._pending is None

    def _solved(self) -> bool:
        return self._phase_request_revision == self._request_revision

    def _send_enabled(self) -> bool:
        return not self._closed and not self._command_active and not self._device_diverged and self._solved()

    def catalog(self) -> dict[str, Any]:
        """What the editor offers and how it is spelled (v2 control_catalog), for the window and an AI."""

        height, width = self.shape
        pupil = [
            _number("center_x", "Center X", self._pupil_center_xy[0], 0.0, width - 1, decimals=2),
            _number("center_y", "Center Y", self._pupil_center_xy[1], 0.0, height - 1, decimals=2),
            _number("diameter_x", "1/e² intensity diameter X", self._pupil_diameter_xy[0], 1.0, 2.0 * width, decimals=2),
            _number("diameter_y", "1/e² intensity diameter Y", self._pupil_diameter_xy[1], 1.0, 2.0 * height, decimals=2),
        ]
        operator = [
            _number("carrier_x", "Steering X", 0.0, -1000.0, 1000.0, decimals=5, step=0.01,
                    description="Full-raster edge-to-edge waves across device width."),
            _number("carrier_y", "Steering Y", 0.0, -1000.0, 1000.0, decimals=5, step=0.01,
                    description="Full-raster edge-to-edge waves across device height."),
            *(_number(key, label, 0.0, -1000.0, 1000.0, decimals=5, step=0.01,
                      description=f"{label}: {polynomial}; Noll-normalized waves RMS on the configured unit disk.")
              for key, label, polynomial in _ZERNIKE),
        ]
        return {
            "editor_kind": "slm", "shape_yx": list(self.shape), "presets": list(PRESETS), "paint_modes": [list(item) for item in _PAINT_MODES],
            "intensity": _number("intensity", "Intensity", 1.0, 0.0, 1000.0),
            "fields": {"preset": self._preset_fields, "pupil": pupil, "operator": operator,
                       "text": {"name": "text", "label": "Text", "placeholder": "Chinese and English text", "kinds": ["Text"]}},
            "note": _NOTE, "plot_size": PLOT_SIZE, "plots": [list(item) for item in _PLOTS],
            "files": {action: {"save": save, "name": name, "filter": filter_, "title": title}
                      for action, (save, name, filter_, title) in _FILES.items()},
            "units": {"target": "zero-based raster x/y pixels, intensity weights",
                      "pupil": "SLM pixels; diameter is 1/e^2 intensity diameter",
                      "operator": "carrier: edge-to-edge waves; Zernike: Noll-normalized waves RMS"},
        }

    def state(self, *, details: bool = False) -> dict[str, Any]:
        """The draft and device facts the window shows (v2 control_state)."""

        command, mapping, applied, receipt = self._device_state
        status, word = self._status
        result = {
            "editor_kind": "slm", "open": not self._closed, "device": self.instance, "shape_yx": list(self.shape),
            "revision": self._request_revision + self._revisions["phase"], "request_revision": self._request_revision,
            "solving": not self.solver_idle, "command_pending": self._command_active,
            "pending": not self.solver_idle or self._command_active, "solved": self._solved(),
            "send_enabled": self._send_enabled(), "adopt_enabled": not self._command_active and applied is not None,
            "status": status, "status_word": word, "error": self._last_error,
            "device_status": self._device_text, "device_status_word": self._device_word,
            "device_diverged": self._device_diverged, "command_revision": command, "mapping_revision": mapping,
            "command_known": applied is not None and receipt.get("outcome") != "unknown", "command_receipt": deepcopy(receipt),
            "objective_kind": self._objective_kind, "pupil": self._pupil_settings(), "pupil_text": self._pupil_text,
            "operator": self._operator_settings(), "operator_values": dict(self._operator), "operator_text": self._operator_text,
            "preset": deepcopy(self._preset), "plot_size": self._plot_size, "plots": dict(self._revisions),
        }
        if details:
            y, x = np.nonzero(self._target)
            result["target"] = {"nonzero_count": int(len(x)), "truncated": len(x) > 512,
                                "spots": [{"x": int(cx), "y": int(cy), "intensity": float(self._target[cy, cx])}
                                          for cy, cx in zip(y[:512], x[:512])]}
            result["phase"] = {"minimum_rad": float(np.min(self._phase)), "maximum_rad": float(np.max(self._phase)),
                               "dtype": str(self._phase.dtype)}
            result["pattern_metadata"] = deepcopy(self._pattern_metadata)
            result["context_command_receipt"] = deepcopy(self._context_command_receipt)
        return result

    # --------------------------------------------------------------- telling
    def _tell(self, *, everything: bool = False) -> None:
        """One notice with the state and every image changed since the last (all of them when asked)."""

        from zlc_atom.data import snapshot_from_array
        from zlc_data import SPATIAL_X, SPATIAL_Y
        from zlc_plugin.figures import dataset_to_wire

        with self._lock:
            if everything:
                self._told.clear()
                self._catalog_told = False
            state = self.state()
            planes = {"target": self._target, "phase": self._phase, "wavefront": self._wavefront_phase}
            plots = {}
            for name, title in _PLOTS:
                revision = self._revisions[name]
                if self._told.get(name) == revision:
                    continue
                snapshot = snapshot_from_array(np.asarray(planes[name])[None], producer="slm_editor", signal=name,
                                               cell_axes=(SPATIAL_Y, SPATIAL_X), generation=f"control.{self.instance}", revision=revision)
                plots[name] = {"title": title, "x": "x", "y": "y", "revision": revision, "dataset": dataset_to_wire(snapshot)}
            if not plots and state == self._told_state and self._catalog_told:
                return
            value: dict[str, Any] = {"state": state, "plots": plots}
            if not self._catalog_told:
                value["catalog"] = self.catalog()
            if self._notify("slm_editor", value):
                self._told.update({name: plot["revision"] for name, plot in plots.items()})
                self._told_state = state
                self._catalog_told = True

    def _show_status(self, text: str, word: str = "task") -> None:
        self._status = (text, word)

    def _report_error(self, error: object) -> None:
        self._last_error = str(error)
        self._show_status(self._last_error, "error")

    # ----------------------------------------------------------------- device
    def _read_device_state(self) -> tuple[int, int, np.ndarray | None, dict[str, object]]:
        device = self.adapter
        return (int(device.command_revision), int(device.mapping_revision), device.last_commanded_phase,
                dict(device.last_command_receipt))

    def _render_device_state(self) -> None:
        command_revision, mapping_revision, phase, receipt = self._device_state
        self._device_diverged = (command_revision != self._draft_command_revision or mapping_revision != self._draft_mapping_revision)
        if self._device_error is not None:
            self._device_text, self._device_word = self._device_error, "error"
            return
        prefix = f"Device {receipt['transport']} · {receipt['outcome']} · command r{command_revision} · mapping r{mapping_revision}"
        if self._device_diverged:
            self._phase_match = None
            text = f"{prefix} · changed externally; Adopt or Load before Send"
        elif phase is None:
            self._phase_match = None
            text = f"{prefix} · command is unknown; draft is unsent"
        else:
            compared = self._phase_match
            if compared is None or compared[0] is not phase or compared[1] is not self._phase:
                compared = (phase, self._phase, bool(np.array_equal(phase, self._phase)))
                self._phase_match = compared
            text = f"{prefix} · " + ("draft matches device" if compared[2] else "authoring draft differs from device")
        if self._phase_metadata.get("source") == "science-context":
            text += f" · context provenance {self._context_command_receipt['identity']}"
        self._device_text, self._device_word = text, "warning" if self._device_diverged else "idle"

    def _observe(self, observed: tuple[int, int, np.ndarray | None, dict[str, object]]) -> None:
        # A poll begun before a command may answer after it: the revisions, not the order, say which is new.
        if observed[0] < self._device_state[0] or observed[1] < self._device_state[1]:
            return
        self._device_state = observed
        if self._last_error and self._last_error.startswith("Device state unavailable:"):
            self._last_error = None
        self._device_error = None
        self._render_device_state()

    def _polling(self) -> None:
        """v2's 100 ms device poll: the adapter's cached state, read between this editor's own commands."""

        while not self._stop.wait(0.1):
            if not getattr(self._notify, "alive", True):
                # The device closed under the editor: it goes with it.
                with _EDITORS_LOCK:
                    if _EDITORS.get(self.instance) is self:
                        del _EDITORS[self.instance]
                self.close()
                return
            with self._lock:
                if self._command_active:
                    continue
            try:
                observed = self._read_device_state()
            except Exception as error:  # noqa: BLE001 -- a transport hiccup is a status line, never the end of the editor
                with self._lock:
                    self._last_error = self._device_error = f"Device state unavailable: {error}"
                    self._render_device_state()
                    self._tell()
                continue
            with self._lock:
                if self._closed:
                    return
                self._observe(observed)
                self._tell()

    # ------------------------------------------------------------------ solve
    def _queue_solve(self) -> None:
        self._last_error = None
        optimizer_state = dict(self._spot_optimizer_state or {}) if self._objective_kind == "spots" else None
        self._pending = (self._request_revision, np.array(self._target, copy=True), self._objective_kind,
                         np.array(self._pupil_amplitude, copy=True), optimizer_state)
        self._show_status("Solving latest target…")
        self._wake.notify_all()

    def _solving(self) -> None:
        while True:
            with self._lock:
                while self._pending is None and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                revision, target, objective_kind, pupil_amplitude, optimizer_state = self._pending
                self._pending, self._running = None, True
                initial = np.array(self._pattern_phase, copy=True)
            outcome: tuple[np.ndarray, dict[str, object]] | BaseException | None
            try:
                outcome = solve_phase(target, initial_phase=initial, objective_kind=objective_kind, pupil_amplitude=pupil_amplitude,
                                      spot_optimizer_state=optimizer_state,
                                      stop_requested=lambda: self._stop.is_set() or revision != self._request_revision)
            except InterruptedError:
                outcome = None
            except Exception as error:  # noqa: BLE001 -- the editor's error line, as v2's solve future reported it
                outcome = error
            with self._lock:
                self._running = False
                current = revision == self._request_revision and not self._closed
                if isinstance(outcome, BaseException):
                    if current:
                        self._report_error(outcome)
                elif outcome is not None and current:
                    phase, metadata = outcome
                    self._last_error = None
                    self._pattern_phase = freeze_pattern_phase(phase, self.shape)
                    self._pattern_metadata = dict(metadata)
                    self._spot_optimizer_state = None if optimizer_state is None else dict(optimizer_state)
                    self._phase_request_revision = revision
                    self._compose_phase(wavefront=False)
                    self._show_status(f"Solved with {metadata['method']}; hardware unchanged")
                self._render_device_state()
                self._tell()

    def _compose_phase(self, *, wavefront: bool = True) -> None:
        """The science phase from the pattern and the wavefront; `wavefront`: its inputs
        (pupil, operator) changed, so it is made again (a solve's new pattern keeps it)."""

        operator = self._operator_settings()
        if wavefront:
            self._wavefront_phase = science_operator_wavefront(self.shape, self._pupil_settings(), operator)
            self._revisions["wavefront"] += 1
        self._phase = compose_science_phase(self._pattern_phase, self._wavefront_phase)
        self._phase_metadata = {
            "source": "composite", "hardware_correction": "excluded", "pattern": dict(self._pattern_metadata),
            "input_pupil": self._pupil_text, "carrier_waves_xy": operator["carrier_waves_xy"],
            "zernike_enabled": operator["enabled"], "zernike_noll_waves_rms": operator["zernike_noll_waves_rms"],
        }
        self._revisions["phase"] += 1

    # ----------------------------------------------------------------- target
    def set_target(self, values: object, *, objective_kind: str = "auto") -> None:
        target = validate_target(values)
        if target.shape != self.shape:
            raise ValueError(f"target shape must be {self.shape!r}")
        if objective_kind not in {"auto", "spots", "image"}:
            raise ValueError("objective_kind must be 'auto', 'spots', or 'image'")
        keep_optimizer = (self._objective_kind == objective_kind == "spots" and np.any(target > 0.0)
                          and np.array_equal(target > 0.0, self._target > 0.0))
        if not keep_optimizer:
            self._spot_optimizer_state = None
        self._objective_kind = objective_kind
        self._target, self._request_revision = target, self._request_revision + 1
        self._revisions["target"] += 1
        if np.any(target > 0.0):
            self._queue_solve()
        else:
            self._pending = None
            self._show_status("Empty target; hardware unchanged")

    def preset(self, kind: str | None, values: dict[str, Any], text: str | None) -> None:
        """v2 target.preset: the popup's fields as stated (the rest as they stand), then Apply."""

        fields = {field["name"]: field for field in self._preset_fields}
        for name, value in values.items():
            field = fields[name]
            _finite(value, name)
            if field["type"] == "int":
                if value != int(value):
                    raise ValueError(f"{name} must be an integer")
                value = int(value)
            if not field["minimum"] <= value <= field["maximum"]:
                raise ValueError(f"{name} must be between {field['minimum']} and {field['maximum']}")
            self._preset["values"][name] = value
        if kind is not None:
            if kind not in PRESETS:
                raise ValueError(f"unknown SLM preset {kind!r}")
            self._preset["kind"] = kind
        if text is not None:
            self._preset["text"] = text
        value = self._preset["values"]
        kind = self._preset["kind"]
        grid, spacing = (value["rows"], value["columns"]), (value["spacing_y"], value["spacing_x"])
        if kind == "Grid":
            objective, target = "spots", preset_grid(self.shape, grid, spacing_yx=spacing, intensity=value["intensity"])
        elif kind == "Checkerboard":
            objective, target = "spots", preset_checkerboard(self.shape, grid, spacing_yx=spacing, intensity=value["intensity"])
        elif kind == "Gaussian":
            objective, target = "image", preset_gaussian(self.shape, (value["radius_y"], value["radius_x"]), intensity=value["intensity"])
        elif kind == "Flat Top":
            objective, target = "image", preset_flat_top(self.shape, (value["radius_y"], value["radius_x"]), edge=value["edge"],
                                                         intensity=value["intensity"])
        else:
            objective, target = "spots", preset_text(self.shape, self._preset["text"], spacing=value["text_spacing"],
                                                     atom_budget=value["atom_budget"], intensity=value["intensity"])
        self.set_target(target, objective_kind=objective)

    def spots(self, spots: list[dict[str, Any]]) -> None:
        """v2 target.spots: the target is exactly these pixels (none: Clear)."""

        target = np.zeros(self.shape, dtype=np.float32)
        used = set()
        for spot in spots:
            x, y, intensity = spot.get("x"), spot.get("y"), spot.get("intensity")
            if type(x) is not int or type(y) is not int or not (0 <= x < self.shape[1] and 0 <= y < self.shape[0]):
                raise ValueError("spot x/y must be integer pixels inside the target raster")
            if _finite(intensity, "spot intensity") < 0:
                raise ValueError("spot intensity must be finite and nonnegative")
            if (x, y) in used:
                raise ValueError("each target pixel may occur only once")
            used.add((x, y))
            target[y, x] = intensity
        self.set_target(target, objective_kind="spots")

    def paint(self, mode: str, intensity: float, stroke: str, points: list[tuple[float, float]]) -> None:
        """v2's paint on the target: a press toggles the pixel under it (Toggle) or starts a
        radius-2 brush (Brush, Erase) that the drag goes on with; points are canonical x, y."""

        pixels = []
        for x, y in points:
            column, row = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
            if 0 <= row < self.shape[0] and 0 <= column < self.shape[1]:
                pixels.append((row, column))
        if not pixels or (mode == "toggle" and stroke != "press"):
            return
        target = np.array(self._target, copy=True)
        if mode == "toggle":
            row, column = pixels[0]
            target[row, column] = 0.0 if target[row, column] > 0.0 else 1.0
            self.set_target(target, objective_kind="spots")
            return
        value = 0.0 if mode == "erase" else _finite(intensity, "intensity")
        for row, column in pixels:
            # The radius-2 brush touches a 5 x 5 window, so only that window is masked.
            top, left = max(0, row - 2), max(0, column - 2)
            bottom, right = min(self.shape[0], row + 3), min(self.shape[1], column + 3)
            yy, xx = np.ogrid[top:bottom, left:right]
            window = target[top:bottom, left:right]
            window[(yy - row) ** 2 + (xx - column) ** 2 <= 4] = value
        self.set_target(target, objective_kind=self._objective_kind if mode == "erase" else "spots")

    # ----------------------------------------------------------------- layers
    def pupil(self, values: dict[str, Any]) -> None:
        """v2 pupil.update / Apply pupil and the Input pupil switch: re-solves with the new pupil."""

        height, width = self.shape
        limits = {"center_x": (0.0, width - 1), "center_y": (0.0, height - 1), "diameter_x": (1.0, 2.0 * width),
                  "diameter_y": (1.0, 2.0 * height)}
        if "enabled" in values and type(values["enabled"]) is not bool:
            raise ValueError("enabled must be true or false")
        for name, (low, high) in limits.items():
            if name in values and not low <= _finite(values[name], name) <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        self._pupil_enabled = values.get("enabled", self._pupil_enabled)
        cx, cy = self._pupil_center_xy
        dx, dy = self._pupil_diameter_xy
        self._pupil_center_xy = (float(values.get("center_x", cx)), float(values.get("center_y", cy)))
        self._pupil_diameter_xy = (float(values.get("diameter_x", dx)), float(values.get("diameter_y", dy)))
        self._pupil_support, self._pupil_amplitude = science_pupil_fields(self.shape, self._pupil_settings())
        self._spot_optimizer_state = None
        self._pupil_text = self._pupil_description(self._pupil_enabled)
        self._request_revision += 1
        self._phase_request_revision = None
        self._compose_phase()
        if np.any(self._target > 0.0):
            self._queue_solve()
        else:
            self._pending = None
            self._show_status("Empty target; hardware unchanged")

    def operator(self, values: dict[str, Any], reset: bool) -> None:
        """v2 operator.update, the Zernike switch and Reset wavefront: recomposes, never solves."""

        if "enabled" in values and type(values["enabled"]) is not bool:
            raise ValueError("enabled must be true or false")
        for name, value in values.items():
            if name != "enabled" and abs(_finite(value, name)) > 1000.0:
                raise ValueError(f"{name} must be between -1000.0 and 1000.0")
        if self._frozen_operator is not None:
            # An explicit edit: the wavefront is the controls' from here on (they show the loaded one).
            self._frozen_operator = None
        if reset:
            self._operator.update({name: 0.0 for name in self._operator if name != "enabled"})
        self._operator.update({name: (value if name == "enabled" else float(value)) for name, value in values.items()})
        self._last_error = None
        active = sum(bool(self._operator[name]) for name in self._operator if name != "enabled")
        self._operator_text = f"On · {active} active" if self._operator["enabled"] else "Off"
        self._compose_phase()
        self._show_status("Solving latest target…" if not self.solver_idle else "Science phase updated; hardware unchanged")

    def set_phase(self, values: object, metadata: object = None) -> None:
        """A frozen phase as the draft (Adopt): the wavefront back to zero."""

        self._last_error = None
        self._frozen_operator = None
        self._request_revision += 1
        self._pending = None
        self._phase = freeze_pattern_phase(values, self.shape)
        self._spot_optimizer_state = None
        self._pattern_phase = self._phase
        self._phase_request_revision = self._request_revision
        self._phase_metadata = dict({"source": "loaded"} if metadata is None else metadata)
        self._pattern_metadata = dict(self._phase_metadata)
        self._operator.update({name: (False if name == "enabled" else 0.0) for name in self._operator})
        self._wavefront_phase = canonical_phase(np.zeros(self.shape, dtype=np.float32), self.shape)
        self._revisions["phase"] += 1
        self._revisions["wavefront"] += 1
        self._system_correction = None
        command_revision, mapping_revision, _phase, receipt = self._device_state
        self._context_command_receipt = dict(receipt)
        self._draft_command_revision, self._draft_mapping_revision = command_revision, mapping_revision
        self._operator_text = "Off"
        self._show_status("Science phase loaded; wavefront reset; hardware unchanged")

    # ---------------------------------------------------------------- context
    def _set_context(self, context: dict[str, object], revisions: tuple[int, int]) -> None:
        pupil, operator = context["pupil"], context["operator_metadata"]
        frozen_target = validate_target(context["target_intensity"])
        carrier = operator.get("carrier_waves_xy", [0.0, 0.0])
        coefficients = operator.get("zernike_noll_waves_rms", {})
        arrays = (context["phase"], context["pattern_phase"], context["operator_wavefront"], context["pupil_amplitude"], context["pupil_support"])
        if any(np.asarray(array).shape != self.shape for array in arrays):
            raise ValueError(f"Science Context shape must be {self.shape!r}")
        if frozen_target.shape != self.shape:
            raise ValueError(f"Science Context Target shape must be {self.shape!r}")
        if (type(operator.get("enabled", False)) is not bool or not isinstance(carrier, list) or len(carrier) != 2
                or any(type(value) not in (int, float) or not np.isfinite(value) or abs(value) > 1000.0 for value in carrier)
                or not isinstance(coefficients, dict) or set(coefficients) - {key for key, *_ in _ZERNIKE}
                or any(type(value) not in (int, float) or not np.isfinite(value) or abs(value) > 1000.0 for value in coefficients.values())):
            raise ValueError("Science Context operator metadata is invalid")
        center, diameter = pupil["center_xy"], pupil["diameter_xy"]
        if not (0.0 <= center[0] <= self.shape[1] - 1 and 0.0 <= center[1] <= self.shape[0] - 1
                and diameter[0] <= 2.0 * self.shape[1] and diameter[1] <= 2.0 * self.shape[0]):
            raise ValueError("Science Context pupil lies outside Editor limits")
        receipt = context["command_receipt"]
        if (type(receipt["command_revision"]) is not int or type(receipt["mapping_revision"]) is not int
                or receipt["outcome"] not in {"known-old", "known-new", "unknown"}):
            raise ValueError("Science Context command receipt is invalid")
        self._pupil_enabled = bool(pupil["enabled"])
        self._operator.update({"enabled": bool(operator.get("enabled", False)), "carrier_x": float(carrier[0]),
                               "carrier_y": float(carrier[1]), **{key: float(coefficients.get(key, 0.0)) for key, *_ in _ZERNIKE}})
        self._frozen_operator = {"enabled": bool(operator.get("enabled", False)), "carrier_waves_xy": [float(value) for value in carrier],
                                 "zernike_noll_waves_rms": {key: float(value) for key, value in coefficients.items()}}
        self._request_revision += 1
        self._pending, self._spot_optimizer_state = None, None
        self._target = frozen_target
        self._phase = context["phase"]
        self._pattern_phase = context["pattern_phase"]
        self._wavefront_phase = context["operator_wavefront"]
        self._pupil_amplitude = context["pupil_amplitude"]
        self._pupil_support = context["pupil_support"]
        self._pupil_center_xy = tuple(float(value) for value in pupil["center_xy"])
        self._pupil_diameter_xy = tuple(float(value) for value in pupil["diameter_xy"])
        self._pupil_text = self._pupil_description(bool(pupil["enabled"]))
        self._operator_text = "On · frozen context" if operator.get("enabled") else "Off"
        self._objective_kind = str(context["objective_kind"])
        self._pattern_metadata = dict(context["pattern_metadata"])
        self._phase_metadata = {"source": "science-context"}
        self._system_correction = None if context["system_correction"] is None else dict(context["system_correction"])
        self._context_command_receipt = dict(context["command_receipt"])
        self._draft_command_revision, self._draft_mapping_revision = revisions
        self._phase_request_revision = self._request_revision
        for name in self._revisions:
            self._revisions[name] += 1
        self._last_error = None
        self._show_status("Frozen Science Context loaded; hardware unchanged")

    # --------------------------------------------------------------- commands
    def work(self, label: str, operation: Callable[[], Any], *, claim: bool = False,
             completion: Callable[[Any], None] | None = None) -> None:
        """v2 _queue_work: one device or file operation, "Working…" while it runs; the
        lane is the device's, so nothing else reaches the device meanwhile."""

        with self._lock:
            if self._closed or self._command_active:
                raise RuntimeError("SLM Editor is closing" if self._closed else "SLM command already in progress")
            draft = (self._draft_command_revision, self._draft_mapping_revision)
            self._command_active, self._last_error = True, None
            self._show_status("Working…")
            self._tell()
        try:
            if claim and (self.adapter.command_revision, self.adapter.mapping_revision) != draft:
                raise RuntimeError("Device command changed before ownership was acquired; reconcile first")
            result = operation()
            # The device this command left behind, read on the same lane: the
            # answer from before it is never shown as an external change after it.
            delivered = self._read_device_state() if claim else None
        except Exception as error:
            with self._lock:
                self._command_active = False
                self._report_error(error)
                self._render_device_state()
                self._tell()
            raise
        with self._lock:
            self._command_active = False
            try:
                if delivered is not None:
                    self._device_state = delivered
                    self._draft_mapping_revision = delivered[1]
                    if label.startswith("Phase"):
                        self._draft_command_revision = delivered[0]
                        self._context_command_receipt = delivered[3]
                if completion is not None:
                    completion(result)
                else:
                    self._show_status(label)
            except Exception as error:
                self._report_error(error)
                raise
            finally:
                self._render_device_state()
                self._tell()

    def send(self) -> None:
        with self._lock:
            if not self._solved():
                raise RuntimeError("Wait for the latest target solve before Send")
            self._render_device_state()
            if self._device_diverged:
                raise RuntimeError("Device command changed externally; Adopt or Load before Send")
            expected = canonical_phase(self._phase, self.shape)
        self.work("Phase sent to SLM", lambda: self._apply_phase(expected), claim=True)

    def _apply_phase(self, expected: np.ndarray) -> None:
        device = self.adapter
        applied = device.apply_phase(expected)
        receipt = dict(device.last_command_receipt)
        if (not np.array_equal(applied, expected) or not np.array_equal(device.last_commanded_phase, expected)
                or receipt.get("outcome") != "known-new" or receipt.get("identity") != device.identity
                or receipt.get("command_revision") != device.command_revision
                or receipt.get("mapping_revision") != device.mapping_revision):
            raise RuntimeError("SLM did not confirm a known-new command receipt")

    def adopt(self) -> None:
        """The device's known command as a 16-bit frozen Pattern draft."""

        observed = self._read_device_state()
        with self._lock:
            self._device_state = observed
            if observed[2] is None:
                self._render_device_state()
                raise RuntimeError("Device command outcome is unknown; nothing can be adopted")
            self.set_phase(observed[2], {"source": "explicit-device-adopt"})
            self._show_status("Device command adopted as a 16-bit frozen Pattern draft")
            self._render_device_state()

    def refresh(self) -> None:
        observed = self._read_device_state()
        with self._lock:
            self._observe(observed)

    def file(self, action: str, path: str) -> None:
        """v2 file_action: one explicit path through the editor's worker."""

        if not str(path).strip():
            raise ValueError("Name the file")
        if action == "load_target":
            self.work("Target loaded", lambda: load_target(path), completion=lambda result: self.set_target(result[0], objective_kind=result[1]))
        elif action == "import_target":
            self.work("Target imported", lambda: _read_imported_target(path), completion=lambda target: self.set_target(target))
        elif action == "load_context":
            def loaded() -> tuple[dict[str, object], tuple[int, int]]:
                context = load_science_context(path)
                return context, (int(self.adapter.command_revision), int(self.adapter.mapping_revision))
            self.work("Science Context loaded", loaded, completion=lambda result: self._set_context(*result))
        elif action == "save_target":
            with self._lock:
                target, objective = np.array(self._target, copy=True), self._objective_kind
            self.work("Target saved", lambda: self._write(path, lambda: save_target(path, target, objective_kind=objective)))
        elif action == "save_context":
            with self._lock:
                if not self._solved():
                    raise RuntimeError("wait for the latest target solve before saving context")
                arguments = dict(target_intensity=np.array(self._target, copy=True), objective_kind=self._objective_kind,
                                 pupil=self._pupil_settings(), system_correction=deepcopy(self._system_correction),
                                 command_receipt=deepcopy(self._context_command_receipt), pattern_metadata=deepcopy(self._pattern_metadata),
                                 operator_metadata=self._operator_settings())
                pattern = np.array(self._pattern_phase, copy=True)
            self.work("Science Context saved", lambda: self._write(path, lambda: save_science_context(path, pattern, **arguments)))
        else:
            raise ValueError(f"unknown SLM file action {action!r}")

    @staticmethod
    def _write(path: str, write: Callable[[], Any]) -> Any:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return write()

    def set_plot_size(self, size: str) -> None:
        self._plot_size = size
        self._show_status("Solving latest target…" if not self.solver_idle else f"Plot size {size}; hardware unchanged")

    def close(self) -> None:
        with self._lock:
            self._closed, self._pending = True, None
            self._spot_optimizer_state = None
            self._stop.set()
            self._wake.notify_all()


# ------------------------------------------------------------------ commands
_EDITORS: dict[str, SlmEditor] = {}
_EDITORS_LOCK = threading.Lock()


def _editor(adapter: Any, *, opening: bool = False) -> SlmEditor:
    notify = notifier()
    with _EDITORS_LOCK:
        editor = _EDITORS.get(notify.instance)
        if editor is not None and (editor.adapter is not adapter or editor._closed):
            editor = None
        if editor is None:
            if not opening:
                raise RuntimeError("The SLM Editor is not open; editor_open opens it")
            editor = _EDITORS[notify.instance] = SlmEditor(adapter, notify, notify.instance)
        return editor


def _edit(adapter: Any, change: Callable[[SlmEditor], None]) -> dict[str, Any]:
    """One draft edit under the editor's lock; the window and the core are told, the state answered."""

    editor = _editor(adapter)
    with editor._lock:
        try:
            change(editor)
        except Exception as error:
            editor._report_error(error)
            editor._tell()
            raise
        editor._render_device_state()
        editor._tell()
        return editor.state()


def _given(values: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {name: values[name] for name in names if values.get(name) is not None}


def _open(adapter: Any) -> dict[str, Any]:
    editor = _editor(adapter, opening=True)
    editor._tell(everything=True)
    with editor._lock:
        return {**editor.state(), "catalog": editor.catalog()}


def _state(adapter: Any, details: bool | None = None) -> dict[str, Any]:
    with _EDITORS_LOCK:
        editor = _EDITORS.get(notifier().instance)
    if editor is None or editor.adapter is not adapter or editor._closed:
        return {"editor_kind": "slm", "open": False}
    with editor._lock:
        return {**editor.state(details=bool(details)), "catalog": editor.catalog()}


def _close(adapter: Any) -> dict[str, Any]:
    with _EDITORS_LOCK:
        editor = _EDITORS.pop(notifier().instance, None)
    if editor is not None:
        editor.close()
    return {"editor_kind": "slm", "open": False}


_PRESET_NAMES = ["rows", "columns", "spacing_y", "spacing_x", "intensity", "edge", "radius_y", "radius_x", "text_spacing", "atom_budget"]
_PUPIL_NAMES = ["enabled", "center_x", "center_y", "diameter_x", "diameter_y"]
_OPERATOR_NAMES = ["enabled", "carrier_x", "carrier_y", *(key for key, *_ in _ZERNIKE)]


def _preset(adapter: Any, kind: str | None = None, text: str | None = None, **values: Any) -> dict[str, Any]:
    return _edit(adapter, lambda editor: editor.preset(kind, _given(values, _PRESET_NAMES), text or None))


def _spots(adapter: Any, spots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return _edit(adapter, lambda editor: editor.spots(list(spots or ())))


def _paint(adapter: Any, mode: str, stroke: str, points: list[dict[str, Any]] | None = None, intensity: float | None = None) -> dict[str, Any]:
    return _edit(adapter, lambda editor: editor.paint(mode, 1.0 if intensity is None else intensity, stroke,
                                                      [(float(point["x"]), float(point["y"])) for point in points or ()]))


def _pupil(adapter: Any, **values: Any) -> dict[str, Any]:
    return _edit(adapter, lambda editor: editor.pupil(_given(values, _PUPIL_NAMES)))


def _operator(adapter: Any, reset: bool | None = None, **values: Any) -> dict[str, Any]:
    return _edit(adapter, lambda editor: editor.operator(_given(values, _OPERATOR_NAMES), bool(reset)))


def _solve(adapter: Any) -> dict[str, Any]:
    return _edit(adapter, lambda editor: editor._queue_solve())


def _size(adapter: Any, size: str) -> dict[str, Any]:
    return _edit(adapter, lambda editor: editor.set_plot_size(size))


def _refresh(adapter: Any) -> dict[str, Any]:
    return _edit(adapter, lambda editor: editor.refresh())


def _commanded(adapter: Any, action: Callable[[SlmEditor], None]) -> dict[str, Any]:
    """A command that runs as v2's worker did (it tells its own progress)."""

    editor = _editor(adapter)
    try:
        action(editor)
    except Exception as error:
        with editor._lock:
            if editor._last_error != str(error):
                editor._report_error(error)
                editor._tell()
        raise
    with editor._lock:
        return editor.state()


def _send(adapter: Any) -> dict[str, Any]:
    return _commanded(adapter, lambda editor: editor.send())


def _adopt(adapter: Any) -> dict[str, Any]:
    return _commanded(adapter, lambda editor: (editor.adopt(), editor._tell()))


def _file(action: str) -> Callable[..., dict[str, Any]]:
    return lambda adapter, path: _commanded(adapter, lambda editor: editor.file(action, path))


def _float(name: str, label: str, description: str = "") -> AuthoringField:
    return AuthoringField(name, "float", label, description=description)


def _int(name: str, label: str, description: str = "") -> AuthoringField:
    return AuthoringField(name, "int", label, description=description)


_PATH = (AuthoringField("path", "str", "File", required=True,
                        description="a workspace-relative or absolute path on the operator's computer"),)
#: The file is the operator's: the core reads or writes it, wherever this SLM is served (v2).
_READ, _WRITE = FileParam("path", "read"), FileParam("path", "write")
_POINT = (AuthoringField("x", "float", "x", required=True), AuthoringField("y", "float", "y", required=True))
_SPOT = (AuthoringField("x", "int", "x", required=True, description="target raster column"),
         AuthoringField("y", "int", "y", required=True, description="target raster row"),
         AuthoringField("intensity", "float", "Intensity", required=True, minimum=0.0))

#: The editor's commands, one per v2 window action (and v2 control_action), each with its schema.
EDITOR_COMMANDS = (
    Command("editor_open", "Open this SLM's Editor draft (v2 SLM Editor): the default 5x7 grid, solved at once; "
            "an open one is answered as it stands. Returns its state and catalog (fields, bounds, presets, files).", _open),
    Command("editor_state", "The SLM Editor's draft and device facts: status, solved, send_enabled, device_status, pupil, "
            "operator, preset; details adds the target spots and phase range.", _state,
            (AuthoringField("details", "bool", "Details", default=False),), read_only=True),
    Command("editor_close", "Close the SLM Editor: its draft and any solve under way are dropped (the SLM keeps its phase).", _close),
    Command("editor_preset", "Replace the target with a preset (Grid, Checkerboard, Gaussian, Flat Top, Text) and solve. "
            "Fields left out keep their current values; editor_state's catalog gives their bounds.", _preset,
            (AuthoringField("kind", "choice", "Pattern", choices=tuple(AuthoringChoice(kind, kind) for kind in PRESETS)),
             _int("rows", "N rows"), _int("columns", "M columns (Checkerboard: sites in long row)"), _int("spacing_y", "Spacing Y"),
             _int("spacing_x", "Spacing X"), _float("intensity", "Intensity"), _float("edge", "Edge"), _float("radius_y", "Radius Y"),
             _float("radius_x", "Radius X"), _int("text_spacing", "Minimum atom spacing (target px)"), _int("atom_budget", "Atom budget"),
             AuthoringField("text", "str", "Text", description="Chinese and English text"))),
    Command("editor_spots", "Replace the target with exactly these pixels and solve; no spots empties it (Clear: the SLM "
            "keeps its phase).", _spots, (AuthoringField("spots", "rows", "Spots", columns=_SPOT),)),
    Command("editor_paint", "Paint the target at canonical x, y points as the window's Paint does: Toggle flips the pixel "
            "under a press; Brush (at intensity) and Erase paint a radius-2 disc at each point of a press or drag; then solve.",
            _paint, (AuthoringField("mode", "choice", "Paint", required=True, choices=tuple(AuthoringChoice(value, label) for value, label in _PAINT_MODES)),
                     AuthoringField("stroke", "choice", "Stroke", required=True, choices=(AuthoringChoice("press", "Press"), AuthoringChoice("drag", "Drag"))),
                     AuthoringField("points", "rows", "Points", columns=_POINT), _float("intensity", "Intensity"))),
    Command("editor_pupil", "Set the input pupil (enabled: Gaussian illumination, else uniform; center and 1/e² intensity "
            "diameter in SLM pixels; left out = unchanged) and solve again.", _pupil,
            (AuthoringField("enabled", "bool", "Input pupil"), _float("center_x", "Center X"), _float("center_y", "Center Y"),
             _float("diameter_x", "1/e² intensity diameter X"), _float("diameter_y", "1/e² intensity diameter Y"))),
    Command("editor_operator", "Set the operator wavefront (Zernike switch, steering carrier waves, Noll-normalized waves RMS; "
            "left out = unchanged; reset: all zero first) and recompose the science phase without solving.", _operator,
            (AuthoringField("enabled", "bool", "Zernike"), _float("carrier_x", "Steering X"), _float("carrier_y", "Steering Y"),
             *(_float(key, label) for key, label, _ in _ZERNIKE), AuthoringField("reset", "bool", "Reset wavefront", default=False))),
    Command("editor_solve", "Solve the current target again (nothing is sent).", _solve),
    Command("editor_size", "The size preset the Editor's plots are drawn at (2x2 as opened).", _size,
            (AuthoringField("size", "str", "Size", required=True),)),
    Command("editor_send", "Send the solved science phase to the SLM (refused until the latest solve is in, or when the "
            "device changed externally: Adopt or Load first).", _send, drives=True),
    Command("editor_adopt", "Adopt the device's known command as a 16-bit frozen Pattern draft.", _adopt, drives=True),
    Command("editor_refresh", "Read the device's command state again.", _refresh),
    Command("editor_load_target", "Load a zlc.slm.target JSON file as the target and solve.", _file("load_target"), _PATH, file=_READ),
    Command("editor_save_target", "Save the target as a zlc.slm.target JSON file.", _file("save_target"), _PATH, file=_WRITE),
    Command("editor_import_target", "Import an intensity array (.npy) or image as the target and solve.", _file("import_target"), _PATH, file=_READ),
    Command("editor_load_context", "Load a zlc.slm.science-context NPZ: its exact frozen phase, pupil and operator, without "
            "solving or sending.", _file("load_context"), _PATH, drives=True, file=_READ),
    Command("editor_save_context", "Save the solved draft as a zlc.slm.science-context NPZ (needs the latest solve).",
            _file("save_context"), _PATH, file=_WRITE),
)

#: The window an SLM's Control opens (v2 control_factory=open_slm_control).
CONTROL_WINDOW = "slm-editor"

__all__ = ["CONTROL_WINDOW", "EDITOR_COMMANDS", "PLOT_SIZE", "PRESETS", "SlmEditor"]
