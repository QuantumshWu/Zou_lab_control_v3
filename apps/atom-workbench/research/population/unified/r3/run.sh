#!/bin/bash
# usage: run.sh TEST CASE [VAR=VALUE ...]  (one run at a time, BelowNormal)
source /c/Users/eadri/AppData/Local/Temp/claude/C--Users-eadri--claude-spares-spare-aYE3fY/5f28e5a8-891d-4b8f-bd56-7c0275f16fce/scratchpad/r3/env.sh
lowprio
TEST=$1; CASE=$2; shift 2
for kv in "$@"; do export "$kv"; done
RES=C:/Users/eadri/WorkCode/zlc_v3_atom_research/apps/atom-workbench/research/population/inputs
SC=C:/Users/eadri/AppData/Local/Temp/claude/C--Users-eadri--claude-spares-spare-aYE3fY/5f28e5a8-891d-4b8f-bd56-7c0275f16fce/scratchpad/inputs
if [ -f "$RES/$CASE.request.json" ]; then export ATOM_R3_REQUEST="$RES/$CASE.request.json"; else export ATOM_R3_REQUEST="$SC/$CASE.request.json"; fi
export ATOM_R3_OUT=${ATOM_R3_OUT:-C:/Users/eadri/AppData/Local/Temp/claude/C--Users-eadri--claude-spares-spare-aYE3fY/5f28e5a8-891d-4b8f-bd56-7c0275f16fce/scratchpad/r3/out.jsonl}
EXE=$(ls -t $WT/apps/atom-workbench/rust-solver/target/x86_64-pc-windows-msvc/iterate/deps/atom_lindblad-*.exe | head -1)
"$EXE" "$TEST" --ignored --nocapture --test-threads=1 2>&1 | grep -v "^running\|^test result\|^$"
