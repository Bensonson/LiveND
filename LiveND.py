import argparse
import glob
import numpy as np
import rawpy
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

try:
    import imageio.v3 as iio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

try:
    import tifffile as tiff
    HAVE_TIFFFILE = True
except Exception:
    HAVE_TIFFFILE = False

def srgb_encode(linear: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        (1 + a) * np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.4) - a
    )

def srgb_decode(srgb: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + a) / (1 + a), 2.4)
    )

def to_gray_linear(rgb_linear: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb_linear[..., 0] +
            0.7152 * rgb_linear[..., 1] +
            0.0722 * rgb_linear[..., 2]).astype(np.float32)

def to_gray_8bit(rgb_linear: np.ndarray) -> np.ndarray:
    gray_lin = to_gray_linear(rgb_linear)
    gray_srgb = srgb_encode(gray_lin)
    return np.clip(gray_srgb * 255.0, 0, 255).astype(np.uint8)

def normalize_exposure(frame_lin: np.ndarray, ref_mean: float, eps: float = 1e-6) -> np.ndarray:
    fmean = float(to_gray_linear(frame_lin).mean())
    if fmean < eps: return frame_lin
    return np.clip(frame_lin * (ref_mean / fmean), 0.0, 1.0)

def load_image_linear(path: str, wb: str = "camera", no_auto_bright: bool = True) -> np.ndarray:
    path_lower = path.lower()
    
    # Process compressed formats (JPG, PNG)
    if path_lower.endswith(('.jpg', '.jpeg', '.png')):
        # Read as unchanged (preserves 16-bit PNG if applicable)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        
        # Convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # Drop alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
        # Normalize to [0, 1]
        if img.dtype == np.uint8:
            img_float = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img_float = img.astype(np.float32) / 65535.0
        else:
            img_float = img.astype(np.float32).clip(0.0, 1.0)
            
        # Decode sRGB back to linear space
        img_lin = srgb_decode(img_float)
        return img_lin
        
    else:
        # Process RAW formats using rawpy (relies on rawpy deciding if it's supported)
        try:
            with rawpy.imread(path) as raw:
                kwargs = dict(output_bps=16, no_auto_bright=no_auto_bright, output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1))
                if wb == "camera": kwargs["use_camera_wb"] = True
                elif wb == "auto": kwargs["use_auto_wb"] = True
                elif wb == "daylight":
                    kwargs["use_auto_wb"] = False
                    kwargs["user_wb"] = [2.0, 1.0, 1.5, 1.0]
                else: kwargs["use_camera_wb"] = True
                rgb16 = raw.postprocess(**kwargs)
            return (rgb16.astype(np.float32) / 65535.0).clip(0.0, 1.0)
        except Exception as e:
            raise ValueError(f"Unsupported file format or rawpy error for: {path} - {e}")

def save_png16(path: str, img_lin_or_srgb: np.ndarray, is_linear: bool) -> None:
    arr16 = np.clip(img_lin_or_srgb * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    # Always use cv2 for 16-bit png, since imageio/PIL crashes on 16-bit RGB arrays
    cv2.imwrite(path, arr16[..., ::-1])

def save_tiff16(path: str, img_lin: np.ndarray, compress: str = "deflate", srgb: bool = False) -> None:
    if not HAVE_TIFFFILE: raise RuntimeError("tifffile not installed.")
    out = img_lin if not srgb else srgb_encode(img_lin)
    arr16 = np.clip(out * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    tiff.imwrite(path, arr16, photometric="rgb", planarconfig="contig", compression=compress)

def save_exr(path: str, img_lin: np.ndarray) -> None:
    if not HAVE_IMAGEIO: raise RuntimeError("imageio not available.")
    iio.imwrite(path, np.clip(img_lin, 0.0, 1.0).astype(np.float32), extension=".exr")


class FeatureAligner:
    def __init__(self, ref_gray: np.ndarray, downscale_factor: float = 0.25, roi_rect: tuple = None):
        """
        ref_gray: 8-bit grayscale image of the reference frame
        roi_rect: Optional (x, y, w, h) in original resolution for static anchoring
        """
        self.downscale = downscale_factor
        self.sift = cv2.SIFT_create(nfeatures=5000)
        
        # Downscale for performance and memory
        self.ref_gray_small = cv2.resize(ref_gray, (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_AREA)

        mask = None
        if roi_rect is not None and roi_rect != (0, 0, 0, 0):
            rx, ry, rw, rh = roi_rect
            # scale onto downscaled image
            sx, sy, sw, sh = int(rx * self.downscale), int(ry * self.downscale), int(rw * self.downscale), int(rh * self.downscale)
            mask = np.zeros(self.ref_gray_small.shape, dtype=np.uint8)
            mask[sy:sy+sh, sx:sx+sw] = 255

        self.keypoints_ref, self.descriptors_ref = self.sift.detectAndCompute(self.ref_gray_small, mask)
        
        # Flann matcher is fast and accurate for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def align(self, mov_rgb_lin: np.ndarray) -> np.ndarray:
        mov_gray = to_gray_8bit(mov_rgb_lin)
        mov_gray_small = cv2.resize(mov_gray, (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_AREA)
        
        keypoints_mov, descriptors_mov = self.sift.detectAndCompute(mov_gray_small, None)
        
        if descriptors_mov is None or len(descriptors_mov) < 10:
            print("Warning: Not enough features to match. Returning unaligned frame.")
            return mov_rgb_lin

        matches = self.matcher.knnMatch(descriptors_mov, self.descriptors_ref, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            print("Warning: Not enough good matches. Returning unaligned frame.")
            return mov_rgb_lin

        # Multiply by 1/downscale to map coordinates back to full-res
        scale = 1.0 / self.downscale
        src_pts = np.float32([keypoints_mov[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale
        dst_pts = np.float32([self.keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale

        # Estimate a rigid transformation (translation, rotation, uniform scale) to avoid perspective distortion
        matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        if matrix is None:
            print("Warning: Affine computation failed. Returning unaligned frame.")
            return mov_rgb_lin

        h, w = mov_rgb_lin.shape[:2]
        return cv2.warpAffine(mov_rgb_lin, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust LiveND using SIFT Stacking (supports many RAW formats, JPG, PNG).")
    parser.add_argument("--glob", type=str, required=True, help="Glob for image files (e.g. photos/*.RAW or *.DNG)")
    parser.add_argument("--mode", type=str, default="median", choices=["median", "mean", "average", "ema"])
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--align", action="store_true", help="Enable SIFT-based alignment", default=True)
    parser.add_argument("--select-roi", action="store_true", help="Interactively select a static ROI for alignment")
    parser.add_argument("--wb", type=str, default="camera", choices=["camera", "auto", "daylight"], help="White balance (RAW files only)")
    parser.add_argument("--match-exposure", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--tiff", type=str, default="")
    parser.add_argument("--tiff-compress", type=str, default="deflate", choices=["deflate", "lzw", "none"])
    parser.add_argument("--tiff-srgb", action="store_true")
    parser.add_argument("--exr", type=str, default="")
    
    args = parser.parse_args()
    files = sorted(glob.glob(args.glob))
    if len(files) < 2: raise SystemExit(f"Need at least 2 files, got {len(files)}")
    
    print(f"Found {len(files)} frames.")
    
    print("Loading reference frame...")
    ref = load_image_linear(files[0], wb=args.wb)
    ref_mean = float(to_gray_linear(ref).mean())
    
    aligner = None
    if args.align:
        roi_rect = None
        if args.select_roi:
            print("Please select the static ROI on the pop-up window and press SPACE or ENTER.")
            disp_img = cv2.cvtColor(np.clip(srgb_encode(ref) * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            h, w = disp_img.shape[:2]
            max_dim = 1080
            scale = 1.0
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                disp_img = cv2.resize(disp_img, (int(w*scale), int(h*scale)))
            
            roi = cv2.selectROI("Select Static Region", disp_img, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
            
            if roi == (0, 0, 0, 0):
                print("ROI selection cancelled. Using full image.")
            else:
                rx, ry, rw, rh = roi
                roi_rect = (rx/scale, ry/scale, rw/scale, rh/scale)

        print("Extracting features from reference frame...")
        aligner = FeatureAligner(to_gray_8bit(ref), roi_rect=roi_rect)

    frames_list = [ref] if args.mode == "median" else []
    accumulator = ref.copy()
    current_out = ref.copy()
    
    def process_frame(p):
        mov = load_image_linear(p, wb=args.wb)
        if args.match_exposure:
            mov = normalize_exposure(mov, ref_mean)
        if aligner:
            mov = aligner.align(mov)
        return mov

    print("Processing frames...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process_frame, p) for p in files[1:]]
        for future in tqdm(futures, total=len(futures), desc="Merging frames"):
            mov = future.result()
            if args.mode == "median":
                frames_list.append(mov)
            elif args.mode in ["average", "mean"]:
                accumulator += mov
            else:
                current_out = (1.0 - args.ema_alpha) * current_out + args.ema_alpha * mov
            
    if args.mode == "median":
        print("Computing median (this may take a moment)...")
        out = np.median(frames_list, axis=0).astype(np.float32).clip(0.0, 1.0)
    elif args.mode in ["average", "mean"]:
        out = (accumulator / len(files)).clip(0.0, 1.0)
    else:
        out = current_out.clip(0.0, 1.0)
    
    wrote_any = False
    if args.tiff:
        save_tiff16(args.tiff, out, compress=args.tiff_compress, srgb=args.tiff_srgb)
        wrote_any = True
    if args.exr:
        save_exr(args.exr, out)
        wrote_any = True
    if args.out:
        to_write = out if args.linear else srgb_encode(out)
        save_png16(args.out, to_write, is_linear=args.linear)
        wrote_any = True
        
    if not wrote_any: raise SystemExit("No output specified. Use --tiff, --exr or --out.")
    print("Done!")
