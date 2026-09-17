"""Extract optical flow features from a single video. Called as subprocess."""
import sys, json, cv2, numpy as np, pandas as pd

def extract(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None
    h, w = prev_frame.shape[:2]
    scale = min(512 / max(h, w), 1.0)
    if scale < 1.0:
        prev_frame = cv2.resize(prev_frame, None, fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_mags, frame_stds, frame_coherences, frame_maxes, all_mags = [], [], [], [], []
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        frame_mags.append(float(np.mean(mag)))
        frame_stds.append(float(np.std(mag)))
        frame_maxes.append(float(np.percentile(mag, 99)))
        dx, dy = flow[..., 0].flatten(), flow[..., 1].flatten()
        norms = np.sqrt(dx**2 + dy**2) + 1e-8
        ux, uy = dx / norms, dy / norms
        sig = mag.flatten() > np.percentile(mag, 50)
        coh = float(np.sqrt(np.mean(ux[sig])**2 + np.mean(uy[sig])**2)) if sig.sum() > 10 else 0.0
        frame_coherences.append(coh)
        all_mags.extend(mag.flatten()[::10].tolist())
        prev_gray = gray
    cap.release()
    if len(frame_mags) < 3: return None

    fm = np.array(frame_mags); fs = np.array(frame_stds)
    fc = np.array(frame_coherences); fx = np.array(frame_maxes)
    am = np.array(all_mags)
    tcv = float(np.std(fm) / (np.mean(fm) + 1e-8))
    tt = float(np.polyfit(range(len(fm)), fm, 1)[0])
    ac = float(np.corrcoef(fm[:-1], fm[1:])[0, 1]) if len(fm) > 2 else 0.0

    return {
        "mag_mean": float(np.mean(fm)), "mag_std": float(np.std(fm)),
        "mag_median": float(np.median(fm)), "mag_p25": float(np.percentile(fm, 25)),
        "mag_p75": float(np.percentile(fm, 75)), "mag_p95": float(np.percentile(fm, 95)),
        "mag_max_mean": float(np.mean(fx)), "mag_max_std": float(np.std(fx)),
        "intra_std_mean": float(np.mean(fs)), "intra_std_std": float(np.std(fs)),
        "coherence_mean": float(np.mean(fc)), "coherence_std": float(np.std(fc)),
        "coherence_min": float(np.min(fc)),
        "temporal_cv": tcv, "temporal_trend": tt,
        "temporal_autocorr": ac if not np.isnan(ac) else 0.0,
        "pixel_mag_p10": float(np.percentile(am, 10)), "pixel_mag_p50": float(np.percentile(am, 50)),
        "pixel_mag_p90": float(np.percentile(am, 90)), "pixel_mag_p99": float(np.percentile(am, 99)),
        "pixel_mag_skew": float(pd.Series(am).skew()),
        "pixel_mag_kurtosis": float(pd.Series(am).kurtosis()),
        "pct_moving_p50": float(np.mean(am > np.median(am))),
        "pct_moving_1px": float(np.mean(am > 1.0)),
        "pct_moving_2px": float(np.mean(am > 2.0)),
        "n_frames": len(fm), "fps": fps,
    }

if __name__ == "__main__":
    feats = extract(sys.argv[1])
    if feats:
        print(json.dumps(feats))
    else:
        print("null")
