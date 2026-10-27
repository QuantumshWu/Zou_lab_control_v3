# Desktop UI design: the zlc_ui Fluent family in CSS

The v3 window must look and behave like the v2 PyQt windows built from
`zlc_ui`. The source of truth is the zlc_ui package that v2 runs:
`Zou_lab_control_v2/packages/zlc_ui` (`fluent/style.py` tokens, `fluent/fluent.py`
controls, `console/panel_card_view.py`, `console/task_console_view.py`,
`form/qt_form.py`). The standalone `Github/zlc_ui` checkout is an older
snapshot (August 2026). It has no `SELECTED`/`ACCENT_STRONG` scale, no
segmented control and no two-click remover, so it is not used.

`tokens.css` holds every colour and length. `fluent.css` (the controls) and
`style.css` (the shell and the views) read only tokens. The controls are
`src/fluent/*` (README.md lists the source layout).
`src/scale.ts` writes the runtime inputs before the body is parsed.

## Units: why a v3 control has the v2 control's device pixels

zlc_ui sizes a control in Qt logical px with `scaled_px(v, minimum) =
max(minimum, round(v × fluent))`. Qt 5 (`AA_EnableHighDpiScaling`, Round
policy) paints one logical px as `round-half-up(Windows scale)` device px.
WebView2 paints one CSS px as `devicePixelRatio` device px. `scale.ts`
therefore sets:

| input | rule (v2 source) | this machine (3840×2400, 250 %) |
|---|---|---|
| `--qt` | `round-half-up(dpr) / dpr` CSS px per Qt px | 3 / 2.5 = 1.2 |
| `--fluent` | `resolve_fluent_auto_scale()` on the Qt-logical work area, clamped to [0.72, 1.25] | (760 − 88) / 790 = 0.8506 |
| `--cap-lift` | 2 × ((ascent − descent − capHeight) / 2), measured from Segoe UI, snapped to device px; the bottom padding that centres capitals on a box (`centred_line_top`) | 2.4 px |
| `--font-line` | ascent + descent, rounded up to a whole Qt px (QTextDocument line) | 21.6 px |
| `--number-text` | advance of `qt_form._NUMBER_SAMPLE` (number-box cap) | measured |
| `--chevron-drop`, `--chevron-step-up/down`, `--chevron-disclosure-down/up`, `--info-chevron-mask`, `--tick` | `paint_chevron` / `InfoTree.drawBranches` / `_paint_tick` polylines as SVG masks in device px, each as large as the part it sits in, at v2's snapped centre | — |
| `--tab-close-top` | `_tab_cap_centre`: a closable tab's disk on its label's capitals, on a device px | 16 / 2.5 px |

Every length token is written `calc(max(min, round(v × fluent)) × qt)`. An
integer Qt length is therefore an integer number of device px, and a v3
control covers the same device pixels as its v2 counterpart. On this monitor
v2 control height = 26 Qt px = 78 device px, and v3 = 31.2 CSS px = 78 device
px. The window frame is 3456×1959 device px in both.

## Token table

Colours (`fluent/style.py`). Derived faces were computed with Qt's own
`QColor.darker` and zlc_ui's `_mixed`.

| v2 token | CSS | value | used for |
|---|---|---|---|
| ACCENT | `--accent` | #77AADD | on switch, tick, field focus edge, combo hover edge, scroll handle hover |
| ACCENT_HOVER | `--accent-hover` | #6DA2D5 | on switch / ticked box under pointer |
| ACCENT_TINT | `--accent-tint` | #E4EFF9 | hover wash: popup row, segment, stepper cell, quiet ghost |
| SELECTED | `--selected` | #AACAEA | chosen/pressed: current segment, picked popup row, pressed standard command, active toggle |
| SELECTED_HOVER | `--selected-hover` | #9DC2E7 | chosen under pointer |
| BUTTON_TINT | `--button-tint` | #D4E5F5 | standard command face |
| `_mixed(BUTTON_TINT, ACCENT, .25)` | `--button-hover` | #BDD6EF | standard command hover |
| ACCENT_STRONG | `--accent-strong` | #2F6FB0 | the one primary command (Start camera) |
| `.darker(110)` / `.darker(122)` | `--strong-hover` / `--strong-pressed` | #2B65A0 / #275B90 | primary hover / pressed |
| HOVER | `--ink-accent` | #004578 | standard command ink |
| BG | `--bg` | #F3F3F3 | window face, title band, disabled field, caption |
| SURFACE | `--surface` | #FFFFFF | cards, fields, popups |
| TEXT | `--text` | #323130 | body ink |
| MUTED_INK | `--muted-ink` | #6A6866 | quiet information, placeholders, panel id |
| DISABLED_INK | `--disabled-ink` | #8A8886 | disabled labels/text |
| PLACEHOLDER | `--placeholder` | #A19F9D | field edge, off switch track, dimmed chevron |
| DIVIDER | `--divider` | #E1DFDD | card/popup edge, disabled face, segment track |
| NEUTRAL_FILL | `--neutral-fill` | #C8C6C4 | unticked box, scroll handle |
| `_mixed(NEUTRAL_FILL, ACCENT, .4)` | `--unticked-hover` | #A8BBCE | unticked box under pointer |
| `_mixed(PLACEHOLDER, TEXT, .2)` | `--switch-off-hover` | #8B8987 | off switch under pointer |
| RED / RED_DARK / RED_TINT | `--red` / `--red-dark` / `--red-tint` | #CD7380 / #9E3B4A / #F7E3E6 | error dot / destructive ink and armed face / destructive face |
| `_mixed(RED_TINT, RED, .19)` / `(.3)` | `--red-hover` / `--red-pressed` | #EFCED3 / #EAC1C7 | destructive hover / pressed |
| RED_DARK `.darker(112)` / `(126)` | `--red-strong-hover` / `--red-strong-pressed` | #8D3542 / #7D2F3B | armed hover / pressed |
| GREEN, ORANGE, ORANGE_DARK, GREY, GRAPHITE | same names | #7FC2AD, #D69A6E, #8A4B1F, #A2A2A2, #808080 | status dots (STATUS_CUES), warning ink, chevron |
| AXIS_GROUP_COLORS | `--axis-group-0..2` | #5A4A8A, #8A4B1F, #004578 | panel title structure groups |

Metrics (Qt px before `× --qt`; the value is at fluent 0.8506):

| v2 source | CSS | rule | value |
|---|---|---|---|
| FONT, fluent_font_size() | `--font-family`, `--font-size` | Segoe UI, max(8, round(12f)) pt in the whole px Qt asks for, qRound(pt × 4/3) | 10 pt = 13 Qt px = 15.6 CSS px |
| FluentButton glyph, tab ✕ and ⋯, step chip, badge letters | `fluent/ink.ts` `glyphFace` | round(pt × 1.35) pt; bold max(9, int(disk × 0.62)) px; bold max(11, int(26f × 0.72)) px; 0.8 pt; bold max(6, pt − 5) pt — each in the whole px Qt asks for (qRound(pt × 4/3)), Segoe UI then MS UI Gothic (Qt's fallback for ⋯ and ✕) | 19 / 9 / 15 / 11 / 8 px |
| HEADING_CSS_WEIGHT | `--heading-weight` | Semibold | 600 |
| FIELD_BORDER_PX | `--edge` | 1 Qt px | 3 device px |
| RADIUS | `--radius` / `--radius-inner` | 4 / radius − edge | 3 |
| control_height() | `--control` | 30, min 22 | 26 |
| row_gap() | `--row-gap` | 6, min 4 | 5 |
| window_pad(1), window_pad(.5) | `--window-pad`, `--window-pad-half` | 14 / 7 | 12 / 6 |
| CARD_PAD, CARD_TITLE_PAD | `--card-pad`, `--card-title-pad` | 10 / 4 | 9 / 3 |
| PADDING_V, EDIT_PADDING_H, QLineEdit margin | `--pad-v`, `--edit-pad-h`, `--qline-margin` | 1 / 4 / 2 (unscaled) | 1 / 3 / 2 |
| BUTTON_PADDING_H | `--button-pad-h` | 10 min 6 | 9 |
| COMBO_WIDTH, fluent_stepper_extent() | `--combo-width`, `--stepper-extent` | 16 / 16 + 2 edges | 14 / 16 |
| COMBO_TRI_SIZE, chevron pen | `--chevron-half`, `--chevron-stroke` | round(8f)/2, max(1, round(3f)/2) | 3.5 / 1.5 |
| _switch_track_size(), thumb inset, label gap | `--switch-w/-h`, `--switch-inset`, `--switch-label-gap` | 40 min 32 × even(20 min 16), 4 min 3, 8 | 34 × 18, 3, 7 |
| FluentCheckBox indicator, spacing | `--check-size`, `--check-gap` | even(16) + 2 edges (QSS width excludes the border), 6 | 16, 5 |
| FluentSegmentedButtons rim, pad | `--segment-rim`, `--segment-pad` | 2 min 2, 10 min 6 | 2, 9 |
| fluent_scrollbar_thickness(), handle margin, min | `--scrollbar`, `--scroll-handle-margin`, `--scroll-handle-min` | 12 min 10, 2, 28 | 10, 2, 24 |
| FluentStatusDot, status_dot_gap() | `--dot`, `--dot-gap` | even(12), 8 | 10, 7 |
| popup_gap(), popup item padding | `--popup-gap`, `--popup-item-pad-v/h` | 4 min 3, 3 min 2 × 6 min 4 | 3, 3 × 5 |
| TaskConsole header frame | `--header-h`, `--header-pad-v` | 48 min 38, 6 | 41, 5 |
| FluentStatusStrip | `--strip-h`, `--strip-pad-v` | 38 min 30, 4 | 32, 3 |
| setting_label_width() | `--label-min`, `--label-trail` | max(72, widest + 10) | 61, 9 |
| TaskConsole kind combo | `--kind-combo-w` | 170 min 130 | 145 |
| qt_form._number_cap | `--number-cap` | sample advance + chrome + stepper | measured |
| FluentSwitch animation | `--anim-switch` | 150 ms | 150 ms |

v3-only layout lengths have no v2 counterpart. They are scaled the same way:
`--signal-combo-w` (260), `--exposure-w` (96), `--panel-min-w` (470, an empty
card), `--surface-min-h` (240), `--settings-min-w` (340) and `--body-top` (2,
the card body inset of PanelCardView).

## Alignment

Every view stands on the same few edges; a view with its own number for one of them is a bug.

| edge | the one rule |
|---|---|
| label column | `labelColumnWidth` (fluent/dom.ts, setting_label_width): forms (`shareLabelColumn`), a popup's form, Edit data, the info panes |
| control column | the label column's end: a form's grid, a selection grid's first card, a row of [label, controls] |
| a bordered card's content | `--card-inset` inside its edge (a card pad from its outer edge): cards, popups, the Setting frame, the SLM frames |
| text in a field column | `--field-text-inset`: a field's text, a read-only value beside fields, a column heading over fields |
| a check box's words | `--check-gap` after the box; a row without a box keeps the box's place (Installation's rows, Force rebuild in the state column) |

## Control mapping

| v2 (zlc_ui) | v3 element | notes |
|---|---|---|
| FluentWindow (frameless, BG title bar, ZLab icon) | Tao window made frameless by `crates/zlc-desktop/src/frame.rs`, sized by `chrome.rs`; `.title-bar` (title_bar.ts) | screen_fit_window_size + centre on the work area in v2's device px; the client is the whole window (WindowsFramelessWindow's WM_NCCALCSIZE), DWM shadow over it, rounded on Windows 11; the page draws StandardTitleBar: 32 Qt px band in BG, `zlc.ico` 20 Qt on the body's left column, the title in the UI font, TitleBarButton 46 x 32 with v2's glyph coordinates and faces (hover 26/255 black, pressed 51/255, close #E81123 / #F1707A, white glyph); hit windows over the page answer v2's 5 device px resize border and the band (HTMAXBUTTON for snap layouts, caption drag, double-click maximize, system menu) and tell the page the hover and press; WebView2 `--force-color-profile=srgb` so token and plot values reach the screen unconverted, as Qt's do |
| fluent_widget_stylesheet | `body`, `.window` | BG face; cards a half window pad apart, a window pad from the edge |
| FluentFrame(bordered=False) | `.card` | white rounded region, no edge |
| TaskConsole header row | `header.header` | dot + `task` + muted summary (panels · live/paused · nodes running), the one chooser (`Plot: …` / `Measurement: …`), Add panel/Add node in one fitted slot, Selectors switch as a group of its own, Pause/Resume (active while paused), Save…, Open…, Clear |
| FluentStatusStrip | `StatusStrip`, `.strip` | dot (STATUS_CUES) + one eliding line (full text in the tooltip) + an optional destructive action (Stop task); an empty message falls back to the last idle line |
| FluentButton roles | `button`, `.primary`, `.destructive`, `.quiet`, `.glyph`, `.armed`, `.active` | each face's edge is its own fill; no focus ring; label capitals centred by `--cap-lift`; a one-symbol label (×, ⋯) is a control-square face with the symbol centred on its ink (`paint_centred_ink`: fluent/ink.ts, an SVG in device px, the translation snapped to a device px) |
| FluentButton(confirm_twice) | `armable()` / `command(…, { confirm })` | only removing a plot panel (card ×, Setting's Remove) or a logic node (row ×, Edit page Remove); armed = RED_DARK face for 500 ms (arm_interval_ms) |
| FluentButton.set_dirty / fit_button_words | `Command.setDirty`, `words` | a dirty Save is primary with a `*`; a slot keeps its widest word (Start/Restart/Starting…, Pause/Resume, Add panel/Add node) |
| required_label | `Field` label | a row that cannot be left empty says ` *` (`allow_none` false), except the Panel section's own rows other than Signal, as in v2's Panel form |
| FluentLineEdit | `input` | PLACEHOLDER edge, ACCENT edge on focus only, BG when disabled, MUTED_INK placeholder |
| FluentComboBox + FluentPopup | `select` with `appearance: base-select` and `::picker(select)` | field like an edit (ACCENT edge on hover/open), chevron in the stepper column, popup = white card with DIVIDER edge and popup gap, picked row SELECTED, hover ACCENT_TINT, no checkmark |
| FluentCycleComboBox | `select` with the `Scope` choice, `.cycle-active` | the closed box says `Scope: <coordinate>`, the list's row only `Scope`; picking Scope, or a click on its words (not the arrow), arms the wheel, shown by the SELECTED fill. While it is armed and focused, a wheel notch down (or ↓) is the next coordinate, up (↑) the one before, stopping at the ends. The arrow, Esc, a press anywhere else and blur disarm it. The card's strip closes its axes' line with `axis=coordinate` per Scope |
| FluentDoubleSpinBox / fluent_integer_box | `SpinBox` (fluent/spin.ts), `.spin` | required numbers only (optional = blank-or-number edit, as qt_form). Two stepper cells tiled from the frame's inner edge, hover/pressed wash clipped to the inner radius, no divider, a direction that cannot move is dimmed. The wheel acts only with focus, one notch per step on the step grid (`gridStep` = `_grid_step_inside`) |
| FluentSwitch | `switchButton()`, `.switch` | solid track, white thumb, no outline; `Auto`/`Manual` words sized to the widest |
| Auto row (qt_form `_make_row`) | `Field` with `automatic` | the Auto switch is the first cell; while on, the editor shows `automatic_value`, disabled |
| FluentSegmentedButtons | `Segmented`, `.segmented` | only where the schema declares `segmented` (relim mode, presentation), as qt_form does: a DIVIDER track, a SELECTED pill (radius concentric with the track), ACCENT_TINT hover, no outline |
| FluentCheckBox | `checkBox()`, `.check` | solid NEUTRAL_FILL/ACCENT box with white tick; used for pick-lists; a boolean setting is a switch |
| FluentUnitPicker | `units` of a field, `select.unit-picker` | after its number box, as wide as its widest spelling; a picked spelling is sent to the owner, which re-projects the row; a row without a picker says its unit in its label (`Exposure (s)`) |
| FluentSectionLabel / card_title_row / FluentGroupBox | `.section-title`, `titleRow()`, `GroupBox` | semibold title a control high, its commands at the row's end; a bordered card with a card pad, selected = ACCENT_TINT fill |
| FluentFrame(rule=True) list | `ruledList()`, `.ruled` | a DIVIDER hairline in the middle of a half-section break between items, never under the last |
| FluentTabWidget | `TabWidget`, `.tabs` | white card, pivot tabs (MUTED_INK, TEXT + ACCENT underline when current), first underline on the page content edge, closable tabs with a quiet round ×, no bar over one permanent page, widest tabs capped and elided when they do not fit, ⋯ lists all |
| _FluentRoundedMenu | `Menu`, `.menu` (popover) | white card, DIVIDER edge, rows washed ACCENT_TINT, no tick gutter, light dismiss |
| FluentPopup + FluentSettingsPopupAnchor | `Popup`, `.popup` (popover) | a bordered white card of settings beside its command (below, else above, inside the window), at its owner's width and minimum height; the command wears the active face while open and its press closes it; light dismiss |
| FluentCardDialog, fluent_message/confirm, FluentInputDialog | `Dialog`, `message`, `confirm`, `prompt` | the same card, centred, dragged by its body, the page behind not dimmed; a warning's confirm is the white-on-red fill |
| FluentSelectionGrid | `SelectionGrid` | cells are cards: empty BG, hover ACCENT_TINT with the name muted, chosen SELECTED, derived ACCENT_TINT + dashed SELECTION_EDGE, error RED_TINT, kept ORANGE_TINT, missing red dashes; row actions as ghost ×; `edited` once per gesture |
| InfoPane / InfoTree | `InfoPane`, `InfoTree` | path header (Browse…, Refresh), one filter per tab, names and values as an open tree (guides, chevrons), matches ORANGE_TINT, one rounded wash per row; Copy value, Copy name (from the top), Expand all, Collapse all; the name column its widest name up to six tenths of the tree, a longer name showing its two ends (Qt ElideMiddle); the label column from declared names |
| FluentReadoutEdit / PathEdit / PublishedItemsLegend | `Readout`, `PathEdit`, `Legend` | selectable value elided in the middle; typed path + Browse…; BG chips (name TEXT, shape MUTED_INK) |
| FluentScrollArea scrollbars | `::-webkit-scrollbar*` | NEUTRAL_FILL handle with margin, ACCENT under pointer, no arrows |
| FluentStatusDot | `.dot[data-state]` | idle GREY, running GREEN, task ACCENT, warning ORANGE, error RED |
| ConsoleBoardView | `.panel-board` (`PanelBoard`: `#panels` the Task Console's live board, `#figure-panels` the Figure Viewer's) | cards placed under v2's north-west gravity (`views/board_layout.ts`, a port of board_layout.py), a card pad apart, each as large as its picture; a card carried by its title band (open hand) swaps with the card it is let go onto or takes the empty place, the others already standing where the drop leaves them, its slot an ACCENT_TINT card with a dashed edge; the places live per column count for the window, the reading order is committed (`order_panels`) |
| PanelCardView | `.panel`, `.panel-header` | DIVIDER edge; BG title band inside the edge with the inner radius; two lines (id in MUTED_INK, structure groups in AXIS_GROUP_COLORS); status dot with the message as tooltip; `Setting` (active while open) and the red-on-hover × |
| Setting frame (FluentOverlayFrame) | `.settings` | inside the board, right edge on Setting's, `Setting · panel` + quiet ×, sections Panel / Semantic / Display / Fit under FluentSectionLabel titles sharing one label column (Fit: Fit model, and Parameters / Cell fit value once a model is chosen, its Fit command on the title row), then `Edit … Remove` |
| PanelEditorView (Panel Edit) | `.panel-edit` (panel_edit.ts) | a Task Console Edit page named after the panel: title + `(kind · cell cells)` scrolling with it, the panel's Setting sections in place, each a card of its own (no Cell kind, no card commands), Frozen snapshot (the held picture, `Held at … · revision n · the panel has newer data`, Refresh snapshot), Direct producer (`Logic node: name`, its selector-linked draft fields read only -- one its own node editor manages drawn by that editor (`NodeEditors` with `READ_ONLY`: a seamless scan's plan as its axis rows and what the plan is, without Add axis or the API values), the rest as plain rows (a camera's ROI) -- Open producer); one label column down the page; the tab's × closes it; Save figure (v2's inline group: Directory with Browse…, the day's folder; Name with Auto-name, the panel's name stamped `_%Y%m%d_%H%M%S`; Format segmented PNG / PDF / SVG, PDF and SVG withheld until the native renderer writes them; File, the path it makes; Save figure, then `panel saved to <image> and <archive>`) in place of the card's Save… |
| LogicRowView / LogicEditorView | `LogicRowView`, `LogicEditorView` (views/logic.ts) | dot, name (middle-elided, fixed column), kind column, status in its word's ink, Plot switch, Start (Restart/Starting…), Stop, Edit, ×; outputs as chips; Edit page: title + kind, status line, one label column (Name, Signal, devices, files, parameters), action bar Remove … Revert, Plot, Start (Restart*), Stop. Panel publishers keep the node columns |
| DeviceManagerView / DeviceControlView | views/device_manager.ts, device_control.ts | Config: Devices + dot + document name; family cards of draft devices (Role, type, Remove, disclosure chevron, parameters) a section gap from Discovered hardware and Loaded session (Control, Log, Close); New… Open… Save Save as… … lifecycle. Control and Log open as closable pages |
| Window set (v2 separate windows) | `Shell` (shell.ts), host `windows.rs` | one native window per v2 window, its page showing its view alone; the title bar says the v2 window's title; Devices@ opens first, Init brings TaskConsole@ in front, Shut down puts Devices@ back, Open/Ctrl+O opens each Figure in a FigureViewer@ of its own (its File field replaces it; closing it drops its Figure), a sequencer's Control shows PulseGUI@ (closing hides it), another device's Control its `<role> control` window (snug: half the standard width or its content's, its content's height; opens once read; closing hides it), Log its `<role> log@Zou lab` window (0.45 screen fit); a closed device takes both; a run's point review is a dialog window of its own (titled with the request, a screen fit, the others take no input until it closes, closing it is Stop); each window opens centred on the screen as v2's launcher, the review centred on TaskConsole@ and kept on its screen (v2 FluentDialogWindow); closing Devices@ or TaskConsole@ closes the experiment |

## Behaviour rules

- One blue scale: hover ACCENT_TINT, chosen/pressed SELECTED, dark
  ACCENT_STRONG only on the primary command.
- No decorative borders. Fields keep their edge; cards and popups keep their
  DIVIDER edge (as v2). Switches, boxes, segments and buttons have no
  outline. No focus ring anywhere (`:focus { outline: none }`).
- Destructive and closing commands are red: Stop, Remove, ×
  on hover. Only the panel card is armed (two clicks). Everything else acts
  on one click.
- Segmented only where the owner declares a closed set; combos for data or
  long lists.
- A line longer than its box ends in an ellipsis, on the side v2 elides it,
  never cut mid-letter or run over its neighbours: the tail by CSS
  (`text-overflow` on a block, never a flex box, whose text the ellipsis
  cannot reach), the middle by `src/fluent/elide.ts` (values and paths, node
  names, form labels, info tree names, a panel title's producer and axes),
  measured once per text and refitted only when its box is resized.  The
  tooltip says the whole text as v2's does: always for a label or a readout,
  only while cut for a check box.
- Spin box wheel: focus-gated and on the step grid. With step 1, from 1 or
  0.1 a notch down gives 0, never 0.0001. Checked by
  `ui_review/tools/behaviour.mjs` in the real window.
- Inner parts derive from the frame. The stepper cells span `edge`..`50 %`
  and `50 %`..`100 % − edge` inside the edge, with corners of `radius − edge`.
  The title band sits inside the card edge with the inner radius. Chevrons
  and digits are centred on the frame's inner centre.

## Measured against v2

These were measured in device px on one monitor (3840×2400 at 250 %, so v2
runs at Qt DPR 3 and v3 at WebView2 DPR 2.5):
- Window frame: 3456×1959 at (192, 160) in both.
- Control height: 78 in both. The settings rows step at the same pitch, and
  the section gaps match.
- Field text inset: the first capital starts 22 px inside the field edge in
  both.
- Capital centring: in a button (`S` of Setting) and in a combo (`F` of
  Facet), v3 leaves T25/B25 and v2 leaves T26/B24. Qt rounds the line to a
  whole logical px (3 device px). v3 keeps the exact centre, one device px
  higher.
- Faces: colours match the tokens exactly once WebView2 outputs sRGB. Without
  `--force-color-profile=srgb`, BUTTON_TINT rendered as (216, 228, 243)
  instead of (212, 229, 245).

## Remaining differences

- Tooltips are WebView2's native tooltips, not Qt's.
- Text rasterisation: Chromium's DirectWrite glyphs against Qt's, at the
  same whole-px size. Shapes differ by at most a pixel; a word's advance is
  within about 0.1 % of Qt's (QFontMetricsF), so a width v2 rounds to a Qt
  px (horizontalAdvance rounds, size() rounds up; v3 does the same for
  labels, samples and buttons) can land one Qt px apart.
- Structure where the products differ. The per-panel Save sits in the Setting frame, as
  v2 keeps figure saving off the card face. A card is the size of its picture
  (the packer is given each card's own rectangle, as v2's is); the workbench
  keeps only the order, the places a drop made live with the window, as v2's.
  Not yet in v3: browser sharing and Remote publication.
- Headless captures (research/ui-mock) must force the device scale factor;
  an emulated one lays out in CSS px and drifts by fractions of a pixel.
