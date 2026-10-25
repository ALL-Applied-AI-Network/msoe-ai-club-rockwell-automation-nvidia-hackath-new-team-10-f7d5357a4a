## 7. Monday Oct 26 — CSI's GB10, 12:30–2:00

Bring an **exFAT USB labelled TTL**: one Sunday-attested `ttl_gb10_*.tar`, its `.sha256`, five scripts,
`monday_ops.py`, `monday.sha256`; allow several GB free. Bring printed staging/numbering sheets and paper logs.
Confirm CSI's supervisor and capture route. Never change cameras, controls or network settings.
Live rule B′ is DEV-fit; organizer permission remains required. Labels and paper logs stay on paper and go to the Mac.

1. **12:30–12:45 — Setup:** `bash /media/$USER/TTL/start.sh`
   No prompts: USB and the single package are found automatically. If the mount differs, use
   `lsblk -o NAME,SIZE,MOUNTPOINT`, then type `bash <mount>/start.sh`.
   Expect checksums **OK → PREFLIGHT OK → BOOTSTRAP OK → SETUP OK**. Check the printed commit against
   Sunday's attestation. **STOP:** fix the listed package/mount/checksum problem and repeat; every start
   keeps a new visit. No successful preflight means no live run. Context survives a closed terminal.

2. **Confirm delivery.** Start staging before CSI's first save. CSI may use the **same camera folder** for
   staging and live, including one repeatedly rewritten filename. Supported sizes: **1920×1080 or 640×480**.
   Ask CSI for supported captures; never resize, crop or re-aim.

3. **12:45–1:30 — Stage:** `bash ~/ttl/stage.sh`
   Enter the camera folder once. At each setup: **Enter** after CSI saves; **s** skip; **r** retake the last
   setup after a new save; **q** done. Expect filename, size, SHA256 and `staged`/`skipped`/`retake`, ending
   **STAGING OK**. Retakes become `setup_NN_rK`; all attempts stay. Duplicate bytes or wrong sizes are
   **REFUSED** without advancing: ask for a new supported capture. **STOP:** preserve files and reconcile
   the reported problem before resuming. Record setup numbers, labels, tilt evidence and timing on paper.

4. **1:30–1:35 — Seal:** `bash ~/ttl/seal.sh`
   Expect counts, final/earlier-attempt lists, checksum **OK**, then **SEALED USB OK**. Last attempt per
   setup is final. Zero captures prints **WARNING** and still seals. USB failure gives **SEALED DISK OK**;
   live is allowed, but export must back up the disk copy. **STOP:** reconcile log/files/hashes; never erase
   evidence. An already sealed visit refuses resealing or more staging. Reset the fixture before live.

5. **1:35–1:50 — Live:** `bash ~/ttl/live.sh`
   Enter the camera folder (the staging camera folder is allowed) or approved RTSP/HTTP(S) URL.
   **Enter = n (continuous camera feed); y only for stills CSI saves on request.** Confirm fixture reset and
   cleared buffered staged frames. Expect new **LIVE OUT**, inspector/hash, `watching ...` for folders,
   then console grids. Existing files are ignored; CSI saves new live frames after `watching`.
   Staged/sealed copies and overlapping folders are refused. Never recopy staged captures into live.
   `??` means check by eye. **STOP:** correct the reported source/seal problem; restart into a new OUT.

6. **1:50 — Stop:** Ctrl-C once. Expect frame/event summary and **LIVE OK** (`stop_reason: ctrl-c`).
   **STOP:** retain evidence and record the abnormal stop on paper. Log every restart/gap/intervention.
   CSI refuses live or there is no usable terminal: preserve captures; record staging-only, not a live demo.

7. **1:50–2:00 — Export:** stop writers, restore USB, then `bash ~/ttl/export.sh`.
   Expect archive checksums **OK → RESULTS USB OK: N visits verified**; includes every retry and attempt.
   **STOP:** keep GB10 files and arrange CSI-assisted recovery. Re-verify on the Mac before removing anything.
   On the Mac: adjudicate paper labels without predictions, lock labels by SHA256, score each arm once (§5).

Cut optional live defects, then filming, then live duration; never labels or backups. Real camera streaming and
capped 1080 performance remain unverified. Appendix A retains the long blocks for reference; these steps govern.
