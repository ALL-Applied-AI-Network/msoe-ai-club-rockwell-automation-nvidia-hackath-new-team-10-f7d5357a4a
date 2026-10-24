# v2 render on cc-isaac-gate-2 (D040): exact commands
Budget **$10** of instance time at $4.38/hr (D040). **Noor's OK before step 1.** If the instance won't start: stop and
tell Noor; **never create a new instance**. Stop it as soon as the pack is copied back and its sha256 matches.
Expected: about 1.6–1.9 h ≈ $7–8.3 (check render 1–3 tries × ~4 min; 568 scenes × 7.1 s ≈ 67 min; crops, pack).
Mac commands run in `~/code/ignite/twin_to_line`; `I=cc-isaac-gate-2`.

## 0. The night before (done)
```bash
U=~/Downloads/isaac_train_v2/upload; mkdir -p $U
S=$(git -C ~/code/ignite rev-parse --short HEAD)
git -C ~/code/ignite archive --format=tar.gz -o "$U/code_$S.tar.gz" HEAD twin_to_line      # absolute -o: -C changes the cwd
tar -czf $U/twin_fix2.tar.gz -C ~/Downloads/csi_digital_twin twin_fix2
(cd $U && shasum -a 256 code_*.tar.gz twin_fix2.tar.gz > upload.sha256 && cat upload.sha256)
```
`twin_fix2` was built with `twin_fix2.py` next to `twin_fix1` and passed `--check` (only `tray_good_support`'s
binding and the new material differ from work1).

## 1. Start (9 am, after Noor's OK)
```bash
I=cc-isaac-gate-2; brev start $I          # or the console's Start button
brev ls                                    # -> RUNNING within ~10 min; otherwise STOP and tell Noor
brev refresh; date -u +%H:%M               # write the start time down (budget)
```

## 2. Where does ssh land?
```bash
ssh $I 'test -x /isaac-sim/python.sh && echo IN_ISAAC_CONTAINER || docker ps --format "{{.Names}}  {{.Image}}"'
```
- `IN_ISAAC_CONTAINER` → use 3a.
- A container list → note the Isaac/vscode container's name as `K` and use 3b.
- `ls /root/train_v1` (inside) must show `csi_digital_twin_v1` and `twin_fix1`; if they are gone, stop and tell Noor
  (the twin would have to be copied again).

## 3a. Upload and run (ssh lands in the Isaac container)
```bash
ssh $I 'mkdir -p /root/train_v2/upload'
scp ~/Downloads/isaac_train_v2/upload/* $I:/root/train_v2/upload/
ssh $I 'cd /root/train_v2 && (cd upload && sha256sum -c upload.sha256) && tar -xzOf upload/code_*.tar.gz twin_to_line/v2_render.sh > v2_render.sh && nohup bash v2_render.sh > run.out 2>&1 &'
```
## 3b. Upload and run (ssh lands on the host; K = container name)
```bash
ssh $I 'mkdir -p /tmp/v2upload'; scp ~/Downloads/isaac_train_v2/upload/* $I:/tmp/v2upload/
ssh $I "docker exec $K mkdir -p /root/train_v2 && docker cp /tmp/v2upload $K:/root/train_v2/upload"
ssh $I "docker exec $K bash -c 'cd /root/train_v2 && (cd upload && sha256sum -c upload.sha256) && tar -xzOf upload/code_*.tar.gz twin_to_line/v2_render.sh > v2_render.sh && nohup bash v2_render.sh > run.out 2>&1 &'"
```
(For 3b, prefix every later `ssh $I '…'` command with `docker exec $K`, and copy files out with
`docker cp $K:/root/train_v2/<file> /tmp/` before `scp`.)

## 4. Watch (every few minutes)
```bash
ssh $I 'tail -4 /root/train_v2/v2_render.log; tail -2 /root/train_v2/run.out'
```
Expected order: `== 1 upload checksums` (both `OK`) · `CHECK OK: only tray_good_support's binding …` · `selftest passed`
· `== 4 check render` · per try a table, then `CHECK PASS: render the batch` (or `FAIL` + `next try: value x…, blue x…`)
· `== 5 budget guard` · `train: rendered 520 of 520; not ok: 0` · `tilt: rendered 48 of 48; not ok: 0` · `PACK READY`.
- **Check failed 3 times** (`STOP: check render failed after 3 tries: no batch`, exit 2): copy back
  `train_v2_check_only.tar` and its `.sha256` as in step 5, then step 6. v2 is not rendered (D040).
- **Budget guard** (`STOP: … would exceed $10`, exit 3): copy back the check results, step 6, tell Noor.
- Anything else stops with `STOP: …` or a traceback in `run.out`: copy back `v2_render.log`, step 6, tell Noor.

## 5. Copy back and compare
```bash
P=~/Downloads/isaac_train_v2/pack; mkdir -p $P
scp $I:/root/train_v2/train_v2_pack.tar $I:/root/train_v2/train_v2_pack.tar.sha256 $P/
ssh $I 'cat /root/train_v2/train_v2_pack.tar.sha256'; (cd $P && shasum -a 256 -c train_v2_pack.tar.sha256)   # -> OK
```

## 6. Stop at once (sha256 matched)
```bash
brev stop $I; brev ls; date -u +%H:%M     # -> STOPPED; cost ≈ (stop − start) × $4.38 (the console has the billed figure)
```

## 7. On the Mac (no instance)
```bash
mkdir -p results/train_v2 && tar -xf ~/Downloads/isaac_train_v2/pack/train_v2_pack.tar -C results/train_v2
(cd results/train_v2 && shasum -a 256 -c SHA256SUMS | grep -vc ': OK$')          # -> 0
../.venv/bin/python train_resnet.py --data results/train_v2/data_train results/train_v2/data_tilt \
    --out results/train/resnet18_v2 --aug configs/aug_v2.json                     # ~30 min on MPS
../.venv/bin/python g3_check.py --model-dir results/train/resnet18_v2 --out results/g3/resnet18_v2   # once (D040)
```
GO → v2 replaces v1 as arm A. NO-GO → v1 stays arm A, v2 is an extra arm. The live inspector stays the rule baseline.
