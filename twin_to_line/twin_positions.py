"""Extract per-position 3D anchor points (world, meters) from the tray station."""
import json, re, sys
import numpy as np
from pxr import Usd, UsdGeom

def load(path):
    st = Usd.Stage.Open(path)
    xc = UsdGeom.XformCache()
    out = {}
    for p in st.GetPrimAtPath('/TrayInspection').GetChildren():
        m = re.match(r'([GS])(\d\d)_(tube|cap)$', p.GetName())
        if not m: continue
        pid = m.group(1) + m.group(2)
        M = np.array(xc.GetLocalToWorldTransform(p))          # row-vector convention: p_w = [p,1] @ M
        tw = lambda q: (np.append(q, 1.0) @ M)[:3]
        d = out.setdefault(pid, {"fixture": m.group(1), "index": int(m.group(2))})
        if m.group(3) == 'tube':
            d["tube_base"] = tw([0, 0, 0]).tolist(); d["tube_top"] = tw([0, 0, 4.1]).tolist()
            d["tube_axis"] = (tw([0, 0, 1]) - tw([0, 0, 0])).tolist()
        else:
            d["cap_base"] = tw([0, 0, 0]).tolist(); d["cap_top"] = tw([0, 0, 0.75]).tolist()
            d["cap_rim"] = [tw([0.668*np.cos(a), 0.668*np.sin(a), 0.75]).tolist() for a in np.linspace(0, 2*np.pi, 24, endpoint=False)]
    for name in ('tray_good_top', 'tray_scrap_top'):
        p = st.GetPrimAtPath('/TrayInspection/' + name); M = np.array(xc.GetLocalToWorldTransform(p))
        out[name] = [(np.array([x, y, 0.125, 1]) @ M)[:3].tolist() for x, y in [(-6.125,-6.125),(6.125,-6.125),(6.125,6.125),(-6.125,6.125)]]
    return out

if __name__ == "__main__":
    pos = load(sys.argv[1])
    json.dump(pos, open(sys.argv[2] if len(sys.argv) > 2 else "positions_3d.json", "w"), indent=1)
    G = [pos[f"G{i:02d}"] for i in range(1, 31)]
    ct = np.array([g["cap_top"] for g in G])
    print("G cap tops z (m): %.4f..%.4f" % (ct[:,2].min(), ct[:,2].max()))
    print("G grid spread x %.3f..%.3f  y %.3f..%.3f" % (ct[:,0].min(), ct[:,0].max(), ct[:,1].min(), ct[:,1].max()))
    for i in range(1, 31):
        g = pos[f"G{i:02d}"]; print(f"G{i:02d}", np.round(g["cap_top"], 4))
