"""TacDream indexes the selected local dataset, never a copied absolute-path cache."""

from pathlib import Path

from tacmind0.data.dataset import JsonlDataset


class TacDreamDataset(JsonlDataset):
    def _get_index_cache(self, jsonl_dir: str) -> dict:
        root = Path(jsonl_dir).resolve()
        files = sorted(root.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No episode JSONL files in {root}")
        index = {}
        for path in files:
            with path.open() as handle:
                count = sum(bool(line.strip()) for line in handle)
            if count < 2:
                raise ValueError(f"Episode must contain at least two frames: {path}")
            index[str(path)] = count
        return {"data": index}
