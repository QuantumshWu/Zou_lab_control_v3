// Installation@Zou lab (installation.exe): what this PC has of what the
// experiment needs, one card per part as Devices@ lays out its families --
// the program, the science environment, the instrument links, the SLM GPU, the FPGA tools
// and, under Advanced, a developer's toolchain -- each row its status and its
// one-line detail; then the log of the step under way, and the commands that
// install what is selected or only what is missing. The host (installer.rs)
// checks and runs everything; this page shows it and asks.
import { GroupBox, StatusLine, StatusStrip, TabWidget, checkBox, command, confirm, element, readoutText, ruledList, setChecked, setState, statusDot, switchButton, } from "../fluent/index.js";
import { queueStatus } from "../shell.js";
/** A row's state as a status word: complete; needed and missing; optional and absent; failed; being checked. */
const STATES = { ok: "running", missing: "warning", optional: "idle", error: "error", checking: "task" };
/** The FPGA steps; Program and Flash need the board's cable, and say so first as v2's build_and_program.bat did. */
const FPGA = [
    ["fpga_estimate", "Estimate resources", false], ["fpga_build", "Build", true], ["fpga_program", "Program", true], ["fpga_flash", "Flash", true], ["fpga_diagnose", "Diagnose", true],
];
const CABLE = {
    fpga_program: ["Program the FPGA", "Program the existing bitstream into the FPGA over the JTAG cable.\n\nThis loads the VOLATILE FPGA config (lost when the board powers off); Flash makes it survive a power cycle."],
    fpga_flash: ["Program the SPI flash", "Write the bitstream into the board's SPI configuration flash over the JTAG cable, so the FPGA auto-boots from it on every power-up -- run it once and the program persists across reboots.\n\nRequires the board MODE jumper on SPI/QSPI boot; set ZLC_PS_CFGMEM_PART if your flash differs from the default."],
};
const FPGA_FREE = new Set(["fpga_estimate"]);
/** A step's title opening a sentence (v2's FPGA words are lower case: "hardware diagnose"). */
const sentence = (title) => title.charAt(0).toUpperCase() + title.slice(1);
/** One component's row: its check box (an installable one), title and detail, and its state. */
class Row {
    id;
    root = element("div", "list-row install-row");
    check = null;
    title = element("span", "list-title elided");
    detail = element("span", "note list-detail");
    status = new StatusLine();
    constructor(id, selectable, ctx) {
        this.id = id;
        if (selectable) {
            this.check = checkBox("", on => ctx.transport.send({ action: "installation_select", component: id, selected: on }));
            this.root.append(this.check);
        }
        else
            this.root.append(element("span", "check-slot"));
        const identity = element("div", "list-identity");
        identity.append(this.title, this.detail);
        this.root.append(identity, this.status.root);
    }
    update(component, busy) {
        if (this.title.textContent !== component.title)
            this.title.textContent = component.title;
        const warmed = component.time ? `Warmed ${new Date(component.time).toLocaleString()} · ` : "";
        const detail = warmed + component.detail;
        if (this.detail.textContent !== detail) {
            this.detail.textContent = detail;
            this.detail.title = detail;
        }
        this.detail.classList.toggle("error", component.state === "error");
        this.status.show(component.words, STATES[component.state]);
        if (this.check) {
            setChecked(this.check, component.selected);
            this.check.disabled = busy;
        }
    }
}
class InstallationView {
    ctx;
    key = "installation";
    title = "Installation";
    windowTitle = "Installation@Zou lab";
    root = element("div", "window installation");
    tabs = new TabWidget();
    page = element("div", "config-page");
    dot = statusDot();
    logName = element("span", "document-name elided");
    left = element("div", "config-column left");
    right = element("div", "config-column right");
    cards = new Map();
    rows = new Map();
    advanced;
    disclosure = element("button", "quiet square disclosure");
    rewarm;
    fpga = [];
    force;
    logText = readoutText("No step has run in this window yet.");
    lines = 0;
    planOnly;
    check;
    missing;
    install;
    installCurrent;
    strip;
    asked = 0;
    restartAsked = false;
    constructor(ctx) {
        this.ctx = ctx;
        const send = (command, words) => ctx.transport.send(command, words);
        const header = element("div", "title-row config-header");
        header.append(element("span", "heading", "Installation"), element("span", "spacer"), this.dot, this.logName);
        this.rewarm = command("Re-warm", () => send({ action: "installation_step", step: "rewarm" }, "Starting the SLM GPU warmup…"));
        this.installCurrent = command("Install checked-out release", () => send({ action: "installation_step", step: "install_current" }, "Installing this checkout's release…"));
        this.installCurrent.title = "Install the release named by this checkout without pulling GitHub. Use this after updating with GitHub Desktop, or while the online check is unavailable.";
        this.force = checkBox("Force rebuild", on => send({ action: "installation_option", name: "force_build", on }));
        for (const [step, label, vivado] of FPGA) {
            this.fpga.push([step, command(label, () => void this.runFpga(step, label)), vivado]);
        }
        const cards = [
            ["program", "Program", [this.installCurrent.root]], ["python", "Science environment", []], ["instruments", "Instrument links", []], ["gpu", "SLM GPU", [this.rewarm.root]], ["fpga", "FPGA pulse board", []],
        ];
        for (const [key, title, trailing] of cards) {
            const card = new GroupBox(title, trailing);
            card.body.append(ruledList());
            this.cards.set(key, card);
            this.left.append(card.root);
        }
        const actions = element("div", "list-row install-actions"), steps = element("div", "install-steps");
        steps.append(...this.fpga.map(([, button]) => button.root));
        actions.append(steps, this.force);
        this.cards.get("fpga").body.append(actions);
        // A developer's toolchain: under Advanced, folded until asked for.
        this.disclosure.type = "button";
        this.disclosure.title = "Show the development tools";
        this.disclosure.addEventListener("click", () => this.fold(!this.advanced.body.hidden));
        this.advanced = new GroupBox("Advanced", [this.disclosure]);
        this.advanced.body.append(ruledList());
        this.cards.set("tools", this.advanced);
        this.left.append(this.advanced.root);
        this.fold(true);
        const log = new GroupBox("Log");
        log.root.classList.add("install-log");
        this.logText.classList.add("log-text");
        log.body.append(this.logText);
        this.right.append(log.root);
        const columns = element("div", "config-columns");
        columns.append(this.left, this.right);
        this.planOnly = switchButton(on => send({ action: "installation_option", name: "plan_only", on }), "Plan only");
        this.planOnly.title = "Print what the install would do, and change nothing";
        this.check = command("Check again", () => send({ action: "installation_check" }, "Checking this PC…"));
        this.missing = command("Fill in missing", () => send({ action: "installation_install", missing: true }, "Starting the install…"));
        this.install = command("Install selected", () => send({ action: "installation_install", missing: false }, "Starting the install…"), { role: "primary" });
        const bar = element("div", "config-actions");
        bar.append(this.planOnly, this.check.root, element("span", "spacer"), this.missing.root, this.install.root);
        this.strip = new StatusStrip({ action: ["Stop", () => send({ action: "installation_stop" }, "Stopping the step…")] });
        this.page.append(header, columns, bar, this.strip.root);
        this.tabs.addPermanent("components", "Components", this.page);
        this.root.append(this.tabs.root);
    }
    fold(folded) {
        this.advanced.body.hidden = folded;
        this.disclosure.classList.toggle("open", !folded);
        this.disclosure.title = folded ? "Show the development tools" : "Hide the development tools";
    }
    installation() { return this.ctx.latest.installation; }
    async runFpga(step, label) {
        const cable = CABLE[step];
        if (cable && !await confirm(cable[0], cable[1], { confirmText: label, kind: "info" }))
            return;
        this.ctx.transport.send({ action: "installation_step", step }, `Starting ${label.toLowerCase()}…`);
    }
    update(state) {
        const log = state.installation_log;
        if (log)
            this.append(log);
        const installation = this.installation();
        if (!installation || !state.installation)
            return;
        const busy = Boolean(installation.running);
        // A step starts from the status: not while it is being checked.
        const waiting = busy || installation.checking;
        for (const component of installation.components) {
            let row = this.rows.get(component.id);
            if (!row) {
                row = new Row(component.id, component.selectable, this.ctx);
                this.rows.set(component.id, row);
                this.cards.get(component.card)?.body.querySelector(".ruled")?.append(row.root);
            }
            row.update(component, busy);
        }
        const by = (id) => installation.components.find(component => component.id === id);
        const vivado = by("vivado")?.state === "ok";
        for (const [step, button, needsVivado] of this.fpga) {
            button.disabled = waiting || (needsVivado && !vivado) || (!FPGA_FREE.has(step) && by("board")?.state === "error");
            button.title = needsVivado && !vivado ? "Needs Vivado" : "";
        }
        this.rewarm.disabled = waiting || by("slm_gpu")?.state !== "ok";
        this.installCurrent.disabled = waiting;
        setChecked(this.force, installation.force_build);
        setChecked(this.planOnly, installation.plan_only);
        for (const button of [this.check, this.missing, this.install])
            button.disabled = waiting;
        this.planOnly.disabled = busy;
        this.force.disabled = busy;
        const states = installation.components.map(component => component.state);
        setState(this.dot, busy || installation.checking || installation.asking ? "task" : states.includes("error") ? "error" : states.includes("missing") ? "warning" : "running");
        const file = installation.log_file ?? "";
        this.logName.textContent = file.split(/[\\/]/).pop() ?? "";
        this.logName.title = file ? `This window's log: ${file}` : "";
        this.strip.action.hidden = !busy;
        this.strip.action.disabled = Boolean(installation.running?.stopping);
        // Closing while a step runs: the operator says whether it stops.
        if (installation.close_asked > this.asked) {
            this.asked = installation.close_asked;
            void confirm("A step is under way", `${sentence(installation.running?.title ?? "a step")} is still running. Stop it and close the window?`, { confirmText: "Stop and close" })
                .then(yes => { if (yes)
                this.ctx.transport.send({ action: "installation_close" }); });
        }
        // The program this window runs from was downloaded anew: asked once, it starts again to run it.
        if (installation.restart && !this.restartAsked && !busy) {
            this.restartAsked = true;
            void confirm("Restart installation.exe", "The new program is downloaded. installation.exe closes and starts again to put it in place.", { confirmText: "Restart", cancelText: "Later", kind: "info" })
                .then(yes => { if (yes)
                this.ctx.transport.send({ action: "installation_restart" }); });
        }
        this.status();
    }
    append(log) {
        // A page (re)loaded is sent the whole log from its first line.
        if (log.from === 0) {
            this.logText.textContent = "";
            this.lines = 0;
        }
        if (log.from !== this.lines)
            return;
        const follow = this.logText.scrollTop >= this.logText.scrollHeight - this.logText.clientHeight - 4;
        this.logText.append((this.lines ? "\n" : "") + log.lines.join("\n"));
        this.lines += log.lines.length;
        if (follow)
            this.logText.scrollTop = this.logText.scrollHeight;
    }
    status() {
        const installation = this.installation();
        const idle = installation?.running ? [`${sentence(installation.running.title)}${installation.queue.length ? ` (then ${installation.queue.join(", ")})` : ""}…`, "task"]
            : installation?.checking ? ["Checking this PC…", "task"]
                : installation?.asking ? ["Checking GitHub for newer commits…", "task"]
                    : installation?.last ? [installation.last.text, installation.last.ok ? "idle" : "error"]
                        : installation ? ["What is missing or broken starts checked: Fill in missing installs that, Install selected what is checked.", "idle"] : ["Checking this PC…", "task"];
        const [text, word] = queueStatus(this.ctx.transport, idle);
        this.strip.show(text, word);
    }
}
export const installation = (ctx) => new InstallationView(ctx);
