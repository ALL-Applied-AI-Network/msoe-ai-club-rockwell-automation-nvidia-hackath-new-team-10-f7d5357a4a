# Handover for Vang

Updated Tue Oct 27, 2026. Work from `~/code/crusaders`.

## Frozen system

- Freeze tag: `freeze-2026-10-25`, commit `4c24ec2`.
- Live inspector: rule B′ (DEV-fit).
- Score arms A, B, B′ and B″ each once on the sealed set with
  `score_sealed.py`. Lock labels first, before any scoring.

## Today's CSI session — Tue Oct 27

Outcome pending confirmation from Noor or Shamar. Confirm which setups were
staged, whether sealing and export passed, the live-run outcome, and who holds
the `~/ttl_kit/` copy before treating any step as complete.

## Where things are

- Kit: `ttl_kit.tar` in Noor's Drive.
  SHA256: `c5374baebdb47a75e1ca2a4ca31a625f735675babb4e974359eefd479e0eb71a`.
- Agent guides: `twin_to_line/csi_day2/`.
- Session results: CSI's GB10 in `~/ttl_kit/` and `~/ttl/` until copied.
  Keep the GB10 files until the received copy is verified.

## Next steps, in order

1. Get the whole `~/ttl_kit/` copy from Shamar and verify the copy.
2. Label the sealed pictures from the paper log BEFORE any scoring.
3. Lock the labels by SHA256.
4. Run `score_sealed.py` once per frozen arm from `~/code/crusaders`.
   Keep its once-only lock ledger in that clone.
5. Do error analysis after scoring.
6. Ask Shamar for a 30-minute live-run slot.
7. Finish the write-up Oct 30.
8. Finish the video and one-page summary Oct 31.
9. Submit Sun Nov 1 by 11:59 pm CT.

## Rules

- Never train or tune on the sealed pictures.
- Never commit raw data, CSI images or results.
- Change `main` only through reviewed PRs.
- Keep the frozen code, scripts, thresholds and models unchanged.
