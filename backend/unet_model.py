"""
unet_model.py
U-Net + ResNet34 backbone for olive grove segmentation from Sentinel-2 imagery.

Pipeline:
  1. prepare_dataset()  → fetch S2 tiles from GEE + build binary masks from EZZAYRA polygons
  2. train_unet()        → train the model, save to unet_olive.pth
  3. segment_zone()      → run inference on a new Sentinel-2 tile, return polygon contours

Input bands: B3 (Green), B4 (Red), B5 (RedEdge), B8 (NIR)  → 4-channel images
Spatial resolution: 10 m/pixel, 256×256 patches
"""

import json, logging, math, os
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ─── Optional heavy deps (graceful import) ─────────────────────────────────

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import segmentation_models_pytorch as smp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch / segmentation_models_pytorch not installed. "
                "Run: pip install torch segmentation-models-pytorch Pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ─── Constants ──────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "unet_olive.pth"
TILE_DIR   = DATA_DIR / "tiles"          # cached numpy tiles
MASK_DIR   = DATA_DIR / "masks"          # cached binary masks

PATCH_SIZE = 256          # pixels
SCALE_M    = 10           # GEE export scale (10 m per pixel — S2 native res)
BANDS      = ["B3", "B4", "B5", "B8"]   # 4 channels fed to U-Net
N_CHANNELS = len(BANDS)
DEVICE     = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

# ─── 1. GEE tile fetching ───────────────────────────────────────────────────

def _init_gee(project: str = None):
    if not GEE_AVAILABLE:
        raise RuntimeError("earthengine-api not installed")
    project = project or os.environ.get("GEE_PROJECT")
    sa_key   = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    sa_email = os.environ.get("GEE_SERVICE_ACCOUNT")
    if sa_key and sa_email:
        credentials = ee.ServiceAccountCredentials(sa_email, sa_key)
        ee.Initialize(credentials)
    else:
        ee.Initialize(project=project)


def _parcel_to_bbox(coords: List[dict], pad_m: float = 300.0):
    """Return (minLng, minLat, maxLng, maxLat) with a padding in metres."""
    lats = [c["lat"] for c in coords]
    lngs = [c["lng"] for c in coords]
    lat_c = (max(lats) + min(lats)) / 2
    pad_deg_lat = pad_m / 111_000
    pad_deg_lng = pad_m / (111_000 * math.cos(math.radians(lat_c)))
    return (
        min(lngs) - pad_deg_lng,
        min(lats) - pad_deg_lat,
        max(lngs) + pad_deg_lng,
        max(lats) + pad_deg_lat,
    )


def fetch_gee_tile(parcel: dict, project: str = None) -> np.ndarray:
    """
    Fetch a (PATCH_SIZE, PATCH_SIZE, N_CHANNELS) float32 array from GEE
    representing the Sentinel-2 May-June 2025 composite around this parcel.
    Values are normalised to [0, 1].
    """
    _init_gee(project)

    min_lng, min_lat, max_lng, max_lat = _parcel_to_bbox(parcel["coordinates"])
    region = ee.Geometry.Rectangle([min_lng, min_lat, max_lng, max_lat])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2025-05-01", "2025-06-30")
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(BANDS)
        .median()
        .divide(10000)          # normalise SR values to [0, 1]
    )

    # Use sampleRectangle to get a small array back (no Cloud Storage needed)
    data = s2.sampleRectangle(region=region, defaultValue=0).getInfo()

    arrays = [np.array(data["properties"][b], dtype=np.float32) for b in BANDS]
    # arrays[i] shape: (H, W)
    raw = np.stack(arrays, axis=-1)   # (H, W, C)

    # Resize to PATCH_SIZE × PATCH_SIZE
    if CV2_AVAILABLE:
        tile = cv2.resize(raw, (PATCH_SIZE, PATCH_SIZE),
                          interpolation=cv2.INTER_LINEAR)
    else:
        from PIL import Image
        tile = np.stack([
            np.array(Image.fromarray(raw[..., c]).resize(
                (PATCH_SIZE, PATCH_SIZE), Image.BILINEAR))
            for c in range(N_CHANNELS)
        ], axis=-1)

    return tile.astype(np.float32)


def _poly_to_mask(coords: List[dict],
                  bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Rasterise a polygon into a (PATCH_SIZE, PATCH_SIZE) binary mask
    given the tile's bounding box (minLng, minLat, maxLng, maxLat).
    """
    min_lng, min_lat, max_lng, max_lat = bbox
    lng_range = max_lng - min_lng
    lat_range = max_lat - min_lat

    def to_px(c):
        px = int((c["lng"] - min_lng) / lng_range * (PATCH_SIZE - 1))
        py = int((max_lat - c["lat"]) / lat_range * (PATCH_SIZE - 1))  # flip Y
        return (px, py)

    pts = np.array([to_px(c) for c in coords], dtype=np.int32)
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

    if CV2_AVAILABLE:
        cv2.fillPoly(mask, [pts], 1)
    else:
        # Simple scanline fallback (no cv2)
        from PIL import Image, ImageDraw
        img = Image.new("L", (PATCH_SIZE, PATCH_SIZE), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in pts], fill=1)
        mask = np.array(img, dtype=np.uint8)

    return mask


# ─── 2. Dataset preparation ─────────────────────────────────────────────────

def prepare_dataset(project: str = None):
    """
    For every EZZAYRA parcel:
      - Fetch the GEE tile (saved as <id>.npy in TILE_DIR)
      - Generate the binary mask (saved as <id>_mask.npy in MASK_DIR)

    Safe to run multiple times — skips already-downloaded tiles.
    """
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    parcels = []
    for fname in ["parcelles_OlivierExtensif.json",
                  "parcellesOliviersIntensifs.json"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                parcels.extend(json.load(f)["parcels"])

    log.info("Preparing U-Net dataset for %d EZZAYRA parcels …", len(parcels))

    for i, parcel in enumerate(parcels):
        pid       = parcel["id"]
        tile_path = TILE_DIR / f"{pid}.npy"
        mask_path = MASK_DIR / f"{pid}_mask.npy"

        if tile_path.exists() and mask_path.exists():
            log.debug("Skip %s (already cached)", pid)
            continue

        try:
            log.info("[%d/%d] Fetching tile for %s …", i + 1, len(parcels), pid)
            tile = fetch_gee_tile(parcel, project=project)
            np.save(tile_path, tile)

            bbox = _parcel_to_bbox(parcel["coordinates"])
            mask = _poly_to_mask(parcel["coordinates"], bbox)
            np.save(mask_path, mask)

        except Exception as exc:
            log.warning("Failed to fetch tile for %s: %s", pid, exc)

    log.info("Dataset ready: %d tiles in %s", len(list(TILE_DIR.glob("*.npy"))), TILE_DIR)


# ─── 3. PyTorch Dataset ─────────────────────────────────────────────────────

class OliveDataset(Dataset):
    def __init__(self, augment: bool = False):
        self.tile_paths = sorted(TILE_DIR.glob("*.npy"))
        self.augment    = augment

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        pid       = self.tile_paths[idx].stem
        tile      = np.load(self.tile_paths[idx])           # (H, W, C)
        mask_path = MASK_DIR / f"{pid}_mask.npy"
        mask      = np.load(mask_path).astype(np.float32)   # (H, W)

        if self.augment and np.random.rand() > 0.5:
            tile = np.fliplr(tile).copy()
            mask = np.fliplr(mask).copy()
        if self.augment and np.random.rand() > 0.5:
            tile = np.flipud(tile).copy()
            mask = np.flipud(mask).copy()

        # PyTorch expects (C, H, W)
        tile_t = torch.from_numpy(tile.transpose(2, 0, 1))
        mask_t = torch.from_numpy(mask).unsqueeze(0)        # (1, H, W)
        return tile_t, mask_t


# ─── 4. Model definition ────────────────────────────────────────────────────

def build_model() -> "smp.Unet":
    """
    U-Net with ResNet34 encoder, pretrained on ImageNet.
    We adapt the first conv layer to accept 4 channels instead of 3.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=N_CHANNELS,     # 4 S2 bands
        classes=1,                  # binary mask
        activation=None,            # raw logits (use BCEWithLogitsLoss)
    )
    return model


# ─── 5. Training ────────────────────────────────────────────────────────────

def train_unet(epochs: int = 30, batch_size: int = 4, lr: float = 1e-4,
               project: str = None):
    """
    Train the U-Net on EZZAYRA tiles. Saves best weights to unet_olive.pth.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. "
                           "pip install torch segmentation-models-pytorch")

    if not list(TILE_DIR.glob("*.npy")):
        log.info("No cached tiles found — running prepare_dataset() first …")
        prepare_dataset(project=project)

    dataset    = OliveDataset(augment=True)
    n_val      = max(1, len(dataset) // 5)
    n_train    = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    model     = build_model().to(DEVICE)
    criterion = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for tiles, masks in train_loader:
            tiles, masks = tiles.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(tiles)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tiles, masks in val_loader:
                tiles, masks = tiles.to(DEVICE), masks.to(DEVICE)
                preds     = model(tiles)
                val_loss += criterion(preds, masks).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step()

        log.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                 epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            log.info("  → Best model saved (val_loss=%.4f)", best_val_loss)

    log.info("Training complete. Best val_loss=%.4f. Model at %s",
             best_val_loss, MODEL_PATH)
    return model


# ─── 6. Inference on unknown zone ───────────────────────────────────────────

def segment_zone(zone_bbox: dict, project: str = None,
                 threshold: float = 0.5) -> List[List[dict]]:
    """
    Run U-Net inference on a new, unknown area.

    Args:
        zone_bbox:  {"minLng": …, "minLat": …, "maxLng": …, "maxLat": …}
        project:    GEE project ID
        threshold:  sigmoid threshold for binary mask

    Returns:
        List of detected parcel polygons, each as a list of {lat, lng} dicts.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run train_unet() first.")

    _init_gee(project)

    # Build a fake parcel dict so we can reuse fetch_gee_tile
    min_lng = zone_bbox["minLng"]
    min_lat = zone_bbox["minLat"]
    max_lng = zone_bbox["maxLng"]
    max_lat = zone_bbox["maxLat"]
    fake_parcel = {
        "id": "zone",
        "coordinates": [
            {"lat": min_lat, "lng": min_lng},
            {"lat": max_lat, "lng": min_lng},
            {"lat": max_lat, "lng": max_lng},
            {"lat": min_lat, "lng": max_lng},
        ],
    }

    # Fetch the S2 tile for this zone
    log.info("Fetching Sentinel-2 tile for zone inference …")
    tile = fetch_gee_tile(fake_parcel, project=project)  # (H, W, C)

    # Run U-Net
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    tile_t = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(tile_t)[0, 0].cpu().numpy()   # (H, W)

    prob_mask = 1 / (1 + np.exp(-logit))            # sigmoid
    binary    = (prob_mask > threshold).astype(np.uint8)

    # Extract contours → convert pixel coords back to lat/lng
    if not CV2_AVAILABLE:
        log.warning("cv2 not available — returning bounding box as single parcel")
        return [[
            {"lat": min_lat, "lng": min_lng},
            {"lat": max_lat, "lng": min_lng},
            {"lat": max_lat, "lng": max_lng},
            {"lat": min_lat, "lng": max_lng},
        ]]

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    parcels = []
    lng_range = max_lng - min_lng
    lat_range = max_lat - min_lat

    for cnt in contours:
        # Skip tiny noise contours
        if cv2.contourArea(cnt) < 50:
            continue

        # Simplify the polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)

        coords = []
        for pt in approx:
            px, py = int(pt[0][0]), int(pt[0][1])
            lng    = min_lng + (px / (PATCH_SIZE - 1)) * lng_range
            lat    = max_lat - (py / (PATCH_SIZE - 1)) * lat_range  # flip Y
            coords.append({"lat": round(lat, 6), "lng": round(lng, 6)})

        if len(coords) >= 3:
            parcels.append(coords)

    log.info("U-Net detected %d parcel(s) in the zone.", len(parcels))
    return parcels


# ─── 7. CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="U-Net olive grove segmentation")
    parser.add_argument("command", choices=["prepare", "train", "test"],
                        help="prepare: download GEE tiles | train: train model | "
                             "test: run inference on a sample zone")
    parser.add_argument("--project", default=None,
                        help="GEE Cloud Project ID (overrides .env)")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    gee_project = args.project or os.environ.get("GEE_PROJECT")

    if args.command == "prepare":
        prepare_dataset(project=gee_project)

    elif args.command == "train":
        train_unet(epochs=args.epochs, project=gee_project)

    elif args.command == "test":
        # Run inference on a sample Tunisian olive belt zone
        test_bbox = {
            "minLng": 10.15, "minLat": 36.70,
            "maxLng": 10.25, "maxLat": 36.80,
        }
        polygons = segment_zone(test_bbox, project=gee_project)
        print(f"\nDetected {len(polygons)} parcel(s):")
        for i, poly in enumerate(polygons):
            print(f"  Parcel {i+1}: {len(poly)} vertices, "
                  f"centroid=({sum(c['lat'] for c in poly)/len(poly):.4f}, "
                  f"{sum(c['lng'] for c in poly)/len(poly):.4f})")
