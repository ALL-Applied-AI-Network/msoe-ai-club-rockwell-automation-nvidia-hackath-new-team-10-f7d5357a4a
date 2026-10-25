## 7. Monday Oct 26 on CSI's GB10: USB in to results on USB

**12:30–2:00 only; no installation before the slot.** Bring the Sunday-attested package `.tar`,
its `.sha256`, all five `.sh` scripts, `operator.py` and `monday.sha256` together at the USB root.
Bring printed staging/numbering sheets, label and intervention logs, laptop, charger and pen.
Confirm CSI's supervisor, screen and approved capture route. Never change cameras, controls or network settings.
Live uses frozen rule B′; organizer permission for its DEV-fit thresholds remains a condition.

1. **12:30–12:45 setup.** Find the USB mount with `lsblk -o NAME,SIZE,MOUNTPOINT` if needed.
   Type the actual mount path in place of `/media/...`:

   ```bash
   bash /media/.../start.sh
   ```

   Enter the USB directory and Sunday's package basename (without `.tar`) when asked.
   Expected: script/package checksums `OK`, `PREFLIGHT OK`, `BOOTSTRAP OK`, then **SETUP OK** with the
   saved context path. It creates a fresh `~/ttl/visit_*` and installs the short commands under `~/ttl`.
   A closed terminal loses no context: reopen it and continue with the commands below.
   **STOP** means preserve the visit; repeat setup for a fresh installation. Never unpack over a prior visit.
   Check the printed archive commit against Sunday's attestation. No successful preflight means no live run.

2. **Confirm capture delivery.** CSI saves staging captures in a camera folder separate from the live feed.
   Start step 3 before the first save; existing camera images are ignored. Supported live sizes are
   1920×1080 and 640×480. Ask CSI for supported stills if necessary; never resize, crop or re-aim.
   Real capped 1080 images, CSI's own GB10 and its stream remain unverified.

3. **12:45–1:30 staging; no inspector runs.**

   ```bash
   bash ~/ttl/stage.sh
   ```

   Enter the camera folder once. At `setup NN: Enter after CSI saves it, s = skip, q = done`, press
   Enter after each completed save. The newest new image is copied as `staged/setup_NN.<ext>`;
   the source is untouched. Expected: filename, image size, SHA256 and `setup NN: staged` (or `skipped`).
   No new image leaves the same prompt pending. `q` ends with **STAGING OK** and the actual staged count.
   Restarting resumes the next setup using saved context. The CSV records setup, status, source name,
   copy name, time and SHA256. Noor/Tony still record actual positions/classes, tilt evidence, collector,
   capture time, deviations and timing on the label sheet; predictions never establish ground truth.
   Side photos/filming need CSI approval. Put label sheets and a legible paper-log photo/transcription
   in the visit's `records` folder before sealing; retain the originals.

4. **1:30–1:35 seal.** Finish staging and ask CSI to stop saving staged captures.

   ```bash
   bash ~/ttl/seal.sh
   ```

   Expected: **RECONCILED: N setups actually staged; S skipped**, archive SHA256 checks, then
   **SEALED USB OK**. N must be at least 1; every staged filename and hash must match the log.
   If USB copying/verification fails, **SEALED DISK OK: USB FAILED** identifies the verified GB10
   archive; live may then start, with USB backup retried at export. **STOP** means no live run.
   Sealing closes staging for that visit. Keep every source, copy and archive.

5. **1:35–1:50 live, after a verified seal and CSI's fixture reset.**

   ```bash
   bash ~/ttl/live.sh
   ```

   Enter a fresh live folder or approved RTSP/HTTP(S) URL; answer y/n for `--gate each` (yes for independent
   stills), then confirm CSI reset the fixture and cleared staged/buffered frames. Staged and original
   staging-camera folders, aliases and overlapping folders are refused. Each start creates a new timestamped
   `OUT` and saves its context. Expected: `LIVE OUT`, inspector/hash, `watching ...` for folders, then console
   grids as new images arrive. The first live save must follow `watching`; pre-existing files are ignored.
   `??` means check by eye. Optional browser view: the printed OUT's `view.html`.
   Streams remain untested on a real camera; use CSI-saved files if a stream fails.
   Never feed staged captures into live. Log every reset, restart, gap and intervention on paper.

6. **1:50 stop.** Press Ctrl-C once; expect the frame/event summary, **LIVE OK**, and
   `session.json` with `stop_reason: ctrl-c`. **STOP** means retain evidence and record the abnormal stop.
   Save final label sheets and paper notes in `records`. If CSI refuses live, record staging-only and the
   reason in `records/live_status.txt`; do not claim a completed hardware demo. No screen/terminal:
   ask CSI to retain/copy captures and logs for later review without staged predictions.

7. **1:50–2:00 export every visit.** Stop all writers, restore the USB mount if needed, and type:

   ```bash
   bash ~/ttl/export.sh
   ```

   Expected: archive SHA256 checks, then **RESULTS USB OK: N visits verified** and the destination.
   Includes failed installs, retries, staging-only visits, all attempts, labels and logs. Every export is new;
   allow several GB plus recordings. **STOP** means keep GB10 files and arrange CSI-assisted recovery.
   Retain GB10 files until the Mac verifies the copied checksum. Delete nothing.
   Monday evening: adjudicate without predictions, lock labels by SHA256, then score each declared arm once (§5).

Cut optional live defects first, filming second, then live duration. Never cut labels or backups.
The long manual blocks are retained in Appendix A for reference; use these scripts on Monday.
