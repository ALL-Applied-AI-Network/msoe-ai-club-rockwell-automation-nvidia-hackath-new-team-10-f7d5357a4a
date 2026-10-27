"""D041 input-boundary checks: synthetic labels only, never call the fitter."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_rule_dev_good_only import select_rows


def fixture():
    jobs = [('1', ev, f'frame{ev}') for ev in range(20)]
    positions = [f'G{i:02}' for i in range(1,31)]
    labels = []
    for i, (run, event, frame, pos) in enumerate((r,e,f,p) for r,e,f in jobs for p in positions):
        gt, seq = ('pass','') if i<112 else ('no_vial','waiting') if i<551 else ('crooked_cap','')
        labels.append(dict(run=run,event=event,frame=frame,position=pos,ground_truth=gt,gt_sequence=seq,usable='1'))
    return jobs, labels, positions


def test_good_only_excludes_defects_and_missed_placement():
    jobs, labels, positions = fixture()
    labels[-1].update(ground_truth='no_vial',gt_sequence='missed')
    rows=select_rows(jobs,labels,positions)
    assert len(rows)==551
    assert all(r['ground_truth']=='pass' or (r['ground_truth']=='no_vial' and r['gt_sequence']=='waiting') for r in rows)


def test_good_only_stops_when_usable_pool_changes():
    jobs, labels, positions = fixture()
    labels[0]['usable']='0'
    with pytest.raises(ValueError, match='expected 112 pass'):
        select_rows(jobs,labels,positions)


def test_good_only_refuses_nondev_event_before_join():
    jobs, labels, positions = fixture()
    jobs[0]=('4',0,'synthetic_protected_identifier')
    with pytest.raises(ValueError, match='only DEV'):
        select_rows(jobs,labels,positions)
