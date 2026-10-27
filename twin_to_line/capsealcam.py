"""CapSealCam forward model: twin world point (m, Z up) -> pixel in the real 640x480 cap_seal image.

Matches tools/apply_lens.py exactly: pinhole with focal_px_final and principal point at the image
center, then OpenCV radial distortion with k1 only (k2=p1=p2=k3=0). The 768x576 render is the same
pinhole with a wider canvas (same focal in px), so a point maps to render pixel
(x*f + 384, y*f + 288) before distortion.
"""
import json
import numpy as np


class CapSealCam:
    def __init__(self, path="cameras/CapSealCam.json"):
        c = json.load(open(path))
        self.W, self.H = c["final_image_px"]
        self.rw, self.rh = c["render_px"]
        self.f = float(c["focal_px_final"])
        self.cx, self.cy = c["principal_point_final_px"]
        self.k1 = float(c["lens_k1"])
        self.C = np.array(c["position_m"], float)          # camera center, world
        self.R = np.array(c["R_world_to_cv"], float)        # world -> OpenCV camera axes

    def to_cam(self, Pw):
        return (np.atleast_2d(Pw) - self.C) @ self.R.T      # rows: (X right, Y down, Z forward)

    def project(self, Pw, distort=True):
        """Return Nx2 pixels in the final 640x480 image, plus depth (m). Points behind the camera -> nan."""
        Pc = self.to_cam(Pw)
        z = Pc[:, 2]
        x, y = Pc[:, 0] / z, Pc[:, 1] / z
        if distort:
            r2 = x * x + y * y
            s = 1 + self.k1 * r2
            x, y = x * s, y * s
        uv = np.stack([self.f * x + self.cx, self.f * y + self.cy], 1)
        uv[z <= 0] = np.nan
        return uv, z

    def project_render(self, Pw):
        """Pixel in the undistorted 768x576 pinhole render (what Isaac Sim writes before apply_lens)."""
        Pc = self.to_cam(Pw)
        z = Pc[:, 2]
        return np.stack([self.f * Pc[:, 0] / z + self.rw / 2, self.f * Pc[:, 1] / z + self.rh / 2], 1), z

    def ray_ground(self, u, v, z_plane):
        """Back-project a final-image pixel onto the horizontal plane Z = z_plane (undoes k1 iteratively)."""
        xd, yd = (u - self.cx) / self.f, (v - self.cy) / self.f
        x, y = xd, yd
        for _ in range(20):
            s = 1 + self.k1 * (x * x + y * y)
            x, y = xd / s, yd / s
        d_cam = np.array([x, y, 1.0])
        d_w = self.R.T @ d_cam
        t = (z_plane - self.C[2]) / d_w[2]
        return self.C + t * d_w


if __name__ == "__main__":
    cam = CapSealCam()
    fwd = cam.R.T @ np.array([0, 0, 1.0])
    print("camera center (m):", cam.C.round(3))
    print("optical axis (world):", fwd.round(3), "-> tilt from straight down: %.1f deg" % np.degrees(np.arccos(-fwd[2])))
    print("R orthonormal:", np.allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-6), " det:", round(np.linalg.det(cam.R), 4))
    # Center pixel should land on the optical axis
    p = cam.C + 1.5 * fwd
    print("point 1.5 m along axis ->", cam.project(p)[0].round(2))
    # Field of view and ground footprint at a few plausible table heights
    for zt in (0.8, 0.9, 1.0):
        corners = [cam.ray_ground(u, v, zt) for u, v in [(0, 0), (639, 0), (639, 479), (0, 479)]]
        w = np.linalg.norm(corners[1] - corners[0]); h = np.linalg.norm(corners[3] - corners[0])
        print(f"footprint at Z={zt} m: {w*100:.0f} x {h*100:.0f} cm, distance {cam.C[2]-zt:.2f} m, ~{640/(w*100):.1f} px/cm")
    hfov = 2 * np.degrees(np.arctan(320 / cam.f)); vfov = 2 * np.degrees(np.arctan(240 / cam.f))
    print(f"FOV (pinhole, final image): {hfov:.1f} x {vfov:.1f} deg")
