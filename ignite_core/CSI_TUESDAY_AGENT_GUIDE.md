# CSI Tuesday collection and Excel guide

## What is already automatic

The frozen `ttl_kit.tar` is the collection authority. For every confirmed setup,
`stage.sh` copies the new camera image into the visit as `setup_NN.*` (or a
retake name), then appends this metadata to `records/staging_log.csv`:

- setup number
- status (`staged`, `retake`, or `skipped`)
- camera file name and retained copy name
- UTC completion time
- SHA-256 fingerprint

That is what the teammate meant by “built into the code.” The frozen kit does
not create an Excel workbook. `csi_excel_report.py` is a separate, read-only
reporter that turns those authoritative CSV/JSON logs into `.xlsx` after the
scripts record them.

Do not rebuild or modify the frozen kit. Its required outer SHA-256 is:

```text
c5374baebdb47a75e1ca2a4ca31a625f735675babb4e974359eefd479e0eb71a
```

## Division of responsibility

- Tony reads the Tuesday list, checks the physical fixture, ticks the setup,
  and states any deviation.
- The operator tells Codex `setup NN ready`, `retake NN`, `skip NN because …`,
  or `done`.
- Codex only drives the frozen scripts and reports their output. It does not
  inspect images, infer ground truth, adjust thresholds, or repair the kit.
- The frozen script records the capture automatically. The Tuesday plan supplies
  predeclared ground truth. Software predictions never become ground truth.
- The Excel reporter reads metadata only. It never opens a staged image and it
  writes the workbook outside `~/ttl`.

This removes repeated handwritten transcription while preserving one small,
important human task: confirming that the physical arrangement matches the
staging list and stating any exception.

## Prompt for the collection agent

```text
Use only the frozen ttl_kit scripts. Do not edit or rebuild the kit, inspect the
pictures, change thresholds, score staged data, or touch the camera/network/
robot/PLC controls. Tony is the physical ground-truth checker.

Run start.sh and require SETUP OK. Run stage.sh. For each message I give you:
- "setup NN ready": confirm the script is asking for NN, submit Enter once,
  and report its retained filename, dimensions, UTC log time, and SHA prefix.
- "retake NN": use the script's retake command only; preserve the earlier copy.
- "skip NN because REASON": use the script's skip command and repeat the reason
  back; the frozen CSV records the skip but not its reason, so Tony records that
  exception on the Tuesday sheet.
- "done": quit staging, run seal.sh, and require literal SEALED USB OK.

Stop on a setup-number mismatch, refused/duplicate image, missing file, invalid
hash, unexpected prompt, or anything other than the documented script output.
Do not improvise a fix. Finish the frozen seal/live/export workflow. Generate
Excel afterward from the verified visit export on a machine containing the
reporter. Do not include predictions until physical labels are locked.
```

## Excel command

`csi_excel_report.py` and `tuesday_staging_plan.csv` are not inside
`ttl_kit.tar`. Do not insert them into it, repack it, or download them over the
CSI network. The default workflow is to complete `export.sh`, verify the exported
archive, extract it on a machine with this repository, and point `--ttl-root` at
the extraction folder that contains the `visit_*` directory.

If CSI explicitly approves running the report on its compute module, carry only
those two files separately on approved media and verify their hashes first; the
frozen kit remains byte-for-byte unchanged.

```text
csi_excel_report.py       adb7aa4a490ac4cff3a34ca18502ba94d01f615ebc1ae2af044c592d0936a46b
tuesday_staging_plan.csv  34a4fd0e330058a929b5fb9cc8cff6ab8cf13cb9197675b481b8c6ba9a00d495
```

Run the command from a checkout containing this guide. Put the report in a
separate report directory, never beneath the evidence root:

```bash
mkdir -p reports
python3 ignite_core/csi_excel_report.py \
  --ttl-root "/path/to/extracted/export" \
  --plan ignite_core/tuesday_staging_plan.csv \
  --output "reports/Tuesday_Data_Collection.xlsx"
```

The workbook contains the setup checklist, all predeclared ground-truth rows,
staging attempts and retakes, derived staging intervals, exact live-session
durations, events, frame timing, interventions, alerts, timing definitions, and
source hashes. Predictions are excluded by default to keep labeling blind.
The reporter also refuses to overwrite an existing workbook unless the operator
explicitly supplies `--overwrite`; normally, use a new filename for each visit.

Staging intervals are the time between completed log rows. They include physical
placement, camera saving, and operator response, so they are not pure hands-on
setup times. Live duration is exact when `session.json` contains both timestamps.

## Source documents used

- `Tuesday_Staging_List.pdf` SHA-256:
  `01b8b71c419a375c517e80bfe999be2e347c360233f37dfe2fd1fbdf5df78378`
- `Tuesday_Walkthrough.pdf` SHA-256:
  `4e0eeded61ba7d137e79560d618ad77d8e6e23c8ed1b25118a097510e6b3ca32`

The PDFs refer to a separate start prompt and agent guide, but those files were
not inside the frozen kit or the supplied Downloads folder. This guide is a
tested companion workflow, not a claim that those missing documents were tested.
