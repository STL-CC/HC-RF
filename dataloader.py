"""Data center for HC-RF.

This module provides:
1) Dataset implementation for HC-RF batch format
2) Subject-level train/val/test splitting
3) DataLoader construction
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROI1_LABELS = (17, 18, 53, 54)
ROI2_LABELS = (4, 5, 43, 44)
SCORE_NAMES = [
    "MMSE_Total_Score",
    "GDSCALE_Total_Score",
    "Global_CDR",
    "FAQ_Total_Score",
    "NPI-Q_Total_Score",
]
SCORE_STATS = {
    "MMSE_Total_Score": {"mean": 24.08, "std": 4.56},
    "GDSCALE_Total_Score": {"mean": 1.84, "std": 1.94},
    "Global_CDR": {"mean": 0.72, "std": 0.48},
    "FAQ_Total_Score": {"mean": 11.38, "std": 8.64},
    "NPI-Q_Total_Score": {"mean": 3.45, "std": 4.02},
}


def _score_normalize(name: str, value: float) -> float:
    stat = SCORE_STATS.get(name)
    if stat is None:
        return float(value)
    return (float(value) - stat["mean"]) / max(stat["std"], 1e-6)


def _extract_scores(attrs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    vals, missing = [], []
    for key in SCORE_NAMES:
        val = attrs.get(key, np.nan)
        if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)) or (isinstance(val, (int, float)) and val < 0):
            vals.append(0.0)
            missing.append(1.0)
        else:
            vals.append(_score_normalize(key, float(val)))
            missing.append(0.0)
    return np.asarray(vals, dtype=np.float32), np.asarray(missing, dtype=np.float32)


def _roi_masks(label2d: np.ndarray) -> np.ndarray:
    roi1 = np.isin(label2d, ROI1_LABELS).astype(np.float32)
    roi2 = np.isin(label2d, ROI2_LABELS).astype(np.float32)
    return np.stack([roi1, roi2], axis=0)


def _roi_area_mm2(label2d: np.ndarray) -> np.ndarray:
    roi1 = float(np.sum(np.isin(label2d, ROI1_LABELS)))
    roi2 = float(np.sum(np.isin(label2d, ROI2_LABELS)))
    return np.asarray([roi1, roi2], dtype=np.float32)


def _resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    if img.shape == target_size:
        return img
    ten = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(ten, size=target_size, mode="bilinear", align_corners=False)
    return out.squeeze().numpy()


def _resize_label(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    if img.shape == target_size:
        return img
    ten = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(ten, size=target_size, mode="nearest")
    return out.squeeze().numpy()


def _is_slice_dataset(h5f: h5py.File) -> bool:
    version = str(h5f.attrs.get("version", "")).lower()
    if "2d" in version:
        return True
    for key in h5f.keys():
        grp = h5f[key]
        if "image" in grp:
            return len(grp["image"].shape) == 2
    return False


def _build_slice_index_map(h5f: h5py.File):
    visit_to_slice_key: Dict[str, Dict[int, str]] = defaultdict(dict)
    visit_to_indices: Dict[str, List[int]] = defaultdict(list)
    for key in h5f.keys():
        grp = h5f[key]
        attrs = dict(grp.attrs)
        subject_id = str(attrs.get("Subject_ID", attrs.get("Subject ID", "unknown")))
        visit_no = int(attrs.get("Visit_No", attrs.get("Visit No", 0)))
        visit_key = f"{subject_id}-{visit_no}"
        slice_idx = attrs.get("Slice_Index", None)
        if slice_idx is None:
            try:
                slice_idx = int(str(key).rsplit("-", 1)[-1])
            except Exception:
                continue
        slice_idx = int(slice_idx)
        visit_to_slice_key[visit_key][slice_idx] = key
        visit_to_indices[visit_key].append(slice_idx)
    for k in list(visit_to_indices.keys()):
        visit_to_indices[k] = sorted(set(visit_to_indices[k]))
    return visit_to_slice_key, visit_to_indices


def get_apoe_risk(a1: int, a2: int) -> int:
    alleles = sorted([a1, a2])
    if alleles == [4, 4]:
        return 3
    if 4 in alleles:
        return 2
    if alleles == [3, 3]:
        return 1
    return 0


def _build_hcrf_samples(h5_path: str, args):
    """Build HC-RF sample metadata and normalization statistics."""
    samples = []
    img_size = tuple(getattr(args, "img_size", [224, 224]))

    with h5py.File(h5_path, "r") as h5:
        is_slice = _is_slice_dataset(h5)
        if not is_slice:
            raise RuntimeError("Release code expects 2D/slice-form HDF5 dataset.")

        visit_to_slice_key, visit_to_indices = _build_slice_index_map(h5)
        subjects: Dict[str, set] = {}
        genders = set()
        apoe_values = set()

        for key in h5.keys():
            attrs = dict(h5[key].attrs)
            subject_id = str(attrs.get("Subject_ID", attrs.get("Subject ID", key.split("-")[0])))
            visit_no = int(attrs.get("Visit_No", attrs.get("Visit No", 1)))
            subjects.setdefault(subject_id, set()).add(visit_no)
            genders.add(str(attrs.get("Gender", "Unknown")))
            apoe_values.add(get_apoe_risk(int(attrs.get("APOE A1", 3)), int(attrs.get("APOE A2", 3))))

        roi_values = []
        max_samples = int(getattr(args, "max_samples", -1))
        for subject_id, visit_set in subjects.items():
            visits = sorted(list(visit_set))
            if len(visits) < 2:
                continue
            for j in range(1, len(visits)):
                hist_visits = visits[: j + 1]
                current_visit = visits[j - 1]
                target_visit = visits[j]

                slice_sets = []
                for v in hist_visits + [target_visit]:
                    vk = f"{subject_id}-{v}"
                    slice_sets.append(set(visit_to_indices.get(vk, [])))
                common = set.intersection(*slice_sets) if slice_sets else set()
                if not common:
                    continue

                for slice_idx in sorted(common):
                    target_key = f"{subject_id}-{target_visit}"
                    if target_key not in visit_to_slice_key or slice_idx not in visit_to_slice_key[target_key]:
                        continue
                    label_key = visit_to_slice_key[target_key][slice_idx]
                    label = np.asarray(h5[label_key]["label"], dtype=np.int32)
                    label = _resize_label(label, img_size).astype(np.int32)
                    roi_values.append(_roi_area_mm2(label))

                    samples.append(
                        {
                            "subject_id": subject_id,
                            "history_visits": hist_visits,
                            "current_visit": current_visit,
                            "target_visit": target_visit,
                            "slice_idx": int(slice_idx),
                        }
                    )
                    if max_samples > 0 and len(samples) >= max_samples:
                        break
                if max_samples > 0 and len(samples) >= max_samples:
                    break
            if max_samples > 0 and len(samples) >= max_samples:
                break

    if not samples:
        raise RuntimeError("No valid HC-RF samples were built. Please check dataset content.")

    roi_stack = np.stack(roi_values, axis=0).astype(np.float32)
    roi_mean = roi_stack.mean(axis=0)
    roi_std = np.clip(roi_stack.std(axis=0), a_min=1e-6, a_max=None)

    stats = {
        "genders": sorted(list(genders | {"Unknown"})),
        "apoe_risks": sorted(list(apoe_values)) if apoe_values else [0],
    }
    shared = {
        "visit_to_slice_key": visit_to_slice_key,
        "visit_to_indices": visit_to_indices,
        "roi_mean": roi_mean,
        "roi_std": roi_std,
        "stats": stats,
    }
    return samples, shared


class HCRFDataset(Dataset):
    """HC-RF training dataset.

    Each sample returns all tensors required by `HCRF2D`.
    """

    def __init__(
        self,
        h5_path: str,
        samples: List[Dict],
        shared: Dict,
        args,
        gender_map: Optional[Dict[str, int]] = None,
    ):
        self.h5_path = h5_path
        self.samples = samples
        self.visit_to_slice_key = shared["visit_to_slice_key"]
        self.visit_to_indices = shared["visit_to_indices"]
        self.roi_mean = shared["roi_mean"]
        self.roi_std = shared["roi_std"]
        self.args = args
        self.gender_map = gender_map or {"Unknown": 0}

        self.img_size = tuple(getattr(args, "img_size", [224, 224]))
        self.max_visits = int(getattr(args, "he_max_visits", 9))
        self.input_channels = int(getattr(args, "cf_input_channels", 3))

        self.h5 = None

    def _ensure_open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")

    def _close(self):
        if self.h5 is not None:
            try:
                self.h5.close()
            except Exception:
                pass
            self.h5 = None

    def __del__(self):
        self._close()

    def __len__(self):
        return len(self.samples)

    def _get_slice_key(self, visit_key: str, slice_idx: int) -> Optional[str]:
        if visit_key not in self.visit_to_slice_key:
            return None
        if slice_idx in self.visit_to_slice_key[visit_key]:
            return self.visit_to_slice_key[visit_key][slice_idx]
        choices = self.visit_to_indices.get(visit_key, [])
        if not choices:
            return None
        nearest = min(choices, key=lambda x: abs(x - slice_idx))
        return self.visit_to_slice_key[visit_key][nearest]

    def _load_image(self, visit_key: str, slice_idx: int) -> np.ndarray:
        key = self._get_slice_key(visit_key, slice_idx)
        if key is None:
            return np.zeros((1, *self.img_size), dtype=np.float32)
        img = np.asarray(self.h5[key]["image"], dtype=np.float32)
        return _resize_image(img, self.img_size)[None, ...]

    def _load_label(self, visit_key: str, slice_idx: int) -> np.ndarray:
        key = self._get_slice_key(visit_key, slice_idx)
        if key is None:
            return np.zeros(self.img_size, dtype=np.int32)
        label = np.asarray(self.h5[key]["label"], dtype=np.int32)
        return _resize_label(label, self.img_size).astype(np.int32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()
        sample = self.samples[index]

        subject_id = sample["subject_id"]
        history_visits = sample["history_visits"]
        current_visit = sample["current_visit"]
        target_visit = sample["target_visit"]
        slice_idx = sample["slice_idx"]

        history_images, score_vals, score_missing = [], [], []
        visit_ages, visit_rois, visit_times = [], [], []
        baseline_time = None

        for visit_no in history_visits:
            visit_key = f"{subject_id}-{visit_no}"
            slice_key = self._get_slice_key(visit_key, slice_idx)
            if slice_key is None:
                continue

            attrs = dict(self.h5[slice_key].attrs)
            image = self._load_image(visit_key, slice_idx)
            label = self._load_label(visit_key, slice_idx)
            scores, missing = _extract_scores(attrs)

            age = float(attrs.get("Age", 70.0))
            age_norm = (age - 70.0) / 15.0
            roi_area = _roi_area_mm2(label)

            t_abs = float(attrs.get("Years_from_Baseline", 0.0))
            if baseline_time is None:
                baseline_time = t_abs
            t_rel = t_abs - baseline_time

            history_images.append(image)
            score_vals.append(scores)
            score_missing.append(missing)
            visit_ages.append([age_norm])
            visit_rois.append(roi_area)
            visit_times.append(t_rel)

        if not history_images:
            raise RuntimeError("Empty history sequence in sample.")

        current_key = f"{subject_id}-{current_visit}"
        target_key = f"{subject_id}-{target_visit}"
        current_slice_key = self._get_slice_key(current_key, slice_idx)
        target_slice_key = self._get_slice_key(target_key, slice_idx)

        current_attrs = dict(self.h5[current_slice_key].attrs) if current_slice_key is not None else {}
        target_attrs = dict(self.h5[target_slice_key].attrs) if target_slice_key is not None else {}

        input_mri = self._load_image(current_key, slice_idx)
        target_mri = self._load_image(target_key, slice_idx)
        input_label = self._load_label(current_key, slice_idx)
        target_label = self._load_label(target_key, slice_idx)

        roi_input = _roi_masks(input_label)
        roi_target = _roi_masks(target_label)
        roi_focus = (roi_target.sum(axis=0, keepdims=True) > 0).astype(np.float32)

        time_gap = float(target_attrs.get("Years_from_Baseline", 0.0)) - float(current_attrs.get("Years_from_Baseline", 0.0))

        gender_raw = str(current_attrs.get("Gender", "Unknown"))
        gender_id = self.gender_map.get(gender_raw, self.gender_map.get("Unknown", 0))
        apoe_risk = get_apoe_risk(int(current_attrs.get("APOE A1", 3)), int(current_attrs.get("APOE A2", 3)))

        static_age = float(current_attrs.get("Age", 70.0))
        target_age = float(target_attrs.get("Age", 70.0))
        static_age_norm = (static_age - 70.0) / 15.0

        target_roi_mm2 = _roi_area_mm2(target_label)
        target_roi_norm = (target_roi_mm2 - self.roi_mean) / self.roi_std

        history_images = history_images[: self.max_visits]
        score_vals = score_vals[: self.max_visits]
        score_missing = score_missing[: self.max_visits]
        visit_ages = visit_ages[: self.max_visits]
        visit_rois = visit_rois[: self.max_visits]
        visit_times = visit_times[: self.max_visits]

        history_mask = np.zeros(self.max_visits, dtype=np.float32)
        history_mask[: len(history_images)] = 1.0

        while len(history_images) < self.max_visits:
            history_images.append(np.zeros_like(history_images[0]))
            score_vals.append(np.zeros_like(score_vals[0]))
            score_missing.append(np.zeros_like(score_missing[0]))
            visit_ages.append(np.zeros_like(visit_ages[0]))
            visit_rois.append(np.zeros_like(visit_rois[0]))
            visit_times.append(0.0)

        visit_rois_norm = (np.stack(visit_rois, axis=0) - self.roi_mean) / self.roi_std

        input_tensor = torch.from_numpy(input_mri).float()
        target_tensor = torch.from_numpy(target_mri).float()

        if self.input_channels == 3:
            input_tensor = torch.cat([input_tensor, torch.from_numpy(roi_input).float()], dim=0)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "input_mri": torch.from_numpy(input_mri).float(),
            "target_mri": torch.from_numpy(target_mri).float(),
            "roi_mask_input": torch.from_numpy(roi_input).float(),
            "roi_mask_target": torch.from_numpy(roi_target).float(),
            "roi_mask_focus": torch.from_numpy(roi_focus).float(),
            "history_images": torch.tensor(np.stack(history_images, axis=0), dtype=torch.float32),
            "score_vals": torch.tensor(np.stack(score_vals, axis=0), dtype=torch.float32),
            "score_missing": torch.tensor(np.stack(score_missing, axis=0), dtype=torch.float32),
            "visit_ages": torch.tensor(np.stack(visit_ages, axis=0), dtype=torch.float32),
            "visit_rois": torch.tensor(visit_rois_norm, dtype=torch.float32),
            "visit_times": torch.tensor(np.asarray(visit_times, dtype=np.float32)),
            "history_mask": torch.tensor(history_mask, dtype=torch.float32),
            "static_cats": torch.tensor([gender_id, apoe_risk], dtype=torch.long),
            "static_age": torch.tensor([static_age_norm], dtype=torch.float32),
            "time_gap": torch.tensor([time_gap], dtype=torch.float32),
            "target_roi": torch.tensor(target_roi_norm, dtype=torch.float32),
            "target_roi_mm2": torch.tensor(target_roi_mm2, dtype=torch.float32),
            "subject_id": subject_id,
            "slice_idx": int(slice_idx),
            "input_visit": int(current_visit),
            "target_visit": int(target_visit),
            "input_age": static_age,
            "target_age": target_age,
        }


def _split_subjects(subjects: List[str], args):
    rng = np.random.default_rng(int(args.random_seed))
    subjects = sorted(subjects)
    rng.shuffle(subjects)

    val_ratio = float(getattr(args, "val_ratio", 0.1))
    test_ratio = float(getattr(args, "test_ratio", 0.1))

    n_total = len(subjects)
    n_test = int(round(n_total * test_ratio)) if test_ratio > 0 else 0
    n_test = min(max(n_test, 1 if n_total > 1 else 0), max(n_total - 1, 0)) if n_total > 1 else 0

    test_subjects = set(subjects[:n_test])
    pool = subjects[n_test:]

    n_val = int(round(len(pool) * val_ratio)) if val_ratio > 0 else 0
    n_val = min(max(n_val, 1 if len(pool) > 1 else 0), max(len(pool) - 1, 0)) if pool else 0
    val_subjects = set(pool[:n_val])
    train_subjects = set(pool[n_val:])

    if not train_subjects:
        train_subjects = set(pool)
        val_subjects = set()

    return train_subjects, val_subjects, test_subjects


def _samples_by_subject(samples: List[Dict], subjects: set[str]) -> List[Dict]:
    return [sample for sample in samples if sample["subject_id"] in subjects]


def _make_loader(dataset, batch_size: int, shuffle: bool, args) -> DataLoader:
    num_workers = max(0, int(getattr(args, "num_workers", 0)))
    pin_memory = bool(getattr(args, "pin_memory", False))
    persistent = bool(getattr(args, "persistent_workers", False)) and num_workers > 0

    def _worker_init_fn(worker_id: int):
        base = torch.initial_seed() % (2**32)
        np.random.seed(base + worker_id)

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _worker_init_fn,
        "persistent_workers": persistent,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def get_loaders(args, logger):
    """Return train/val/test dataloaders and split statistics."""
    samples, shared = _build_hcrf_samples(args.data_path, args)

    all_subjects = sorted({sample["subject_id"] for sample in samples})
    train_subjects, val_subjects, test_subjects = _split_subjects(all_subjects, args)

    gender_map = {g: i for i, g in enumerate(shared["stats"]["genders"])}
    args.he_num_genders = len(gender_map)
    args.he_num_apoe = max(shared["stats"]["apoe_risks"]) + 1 if shared["stats"]["apoe_risks"] else 1

    train_samples = _samples_by_subject(samples, train_subjects)
    val_samples = _samples_by_subject(samples, val_subjects)
    test_samples = _samples_by_subject(samples, test_subjects)

    train_ds = HCRFDataset(args.data_path, train_samples, shared, args, gender_map=gender_map)
    val_ds = HCRFDataset(args.data_path, val_samples, shared, args, gender_map=gender_map) if val_samples else None
    test_ds = HCRFDataset(args.data_path, test_samples, shared, args, gender_map=gender_map) if test_samples else None

    train_loader = _make_loader(train_ds, int(args.batch_size), shuffle=True, args=args)
    val_loader = _make_loader(val_ds, int(args.eval_batch_size), shuffle=False, args=args) if val_ds is not None else None
    test_loader = _make_loader(test_ds, int(args.eval_batch_size), shuffle=False, args=args) if test_ds is not None else None

    stats = {
        "split": "train_val_test",
        "num_total_samples": len(samples),
        "num_subjects": len(all_subjects),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "train_subjects": len(train_subjects),
        "val_subjects": len(val_subjects),
        "test_subjects": len(test_subjects),
    }
    logger.info("Data split train_val_test stats: %s", stats)

    return {
        "split_name": "train_val_test",
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "stats": stats,
    }


def get_train_test_loaders(h5_path: str, args, logger):
    """Compatibility wrapper for legacy code path."""
    args.data_path = h5_path
    bundle = get_loaders(args, logger)
    return bundle["train_loader"], bundle["val_loader"], bundle["test_loader"], bundle["stats"]
