"""Collector for WELFake (welfake_dataset.csv) — Module 1.

WELFake merges four established English fake-news corpora (Kaggle,
McIntire, Reuters, BuzzFeed Political), 72,134 rows total. Hosted on
Zenodo with a direct, no-login download URL — confirmed real and
working (see configs/module1_sources.yaml).

Label convention (documented on Zenodo/Kaggle): 0 = fake, 1 = real —
opposite of this project's target convention (fake=1, real=0). The
flip happens in Module 1's label-standardization step
(`src/data_collection/label_standardization.py`), not here — this
collector's only job is to get the untouched file onto disk.

Output: datasets/raw/english/welfake/WELFake_Dataset.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_collection.base_collector import BaseCollector


class WelfakeCollector(BaseCollector):
    source_name = "welfake"

    URL = "https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1"
    FILENAME = "WELFake_Dataset.csv"

    def collect(self, force: bool = False, **kwargs: Any) -> dict[str, Any]:
        if not force and self.already_collected(self.FILENAME):
            path = self.output_dir / self.FILENAME
            self.logger.info("%s already present (%.2f MB); skipping download. Pass force=True to re-download.", path, path.stat().st_size / 1e6)
            return {
                "source": self.source_name,
                "status": "skipped_existing",
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1e6, 2),
            }

        self.logger.info("Downloading WELFake from %s", self.URL)
        response = self.fetch(self.URL)
        if response is None:
            return {"source": self.source_name, "status": "failed", "reason": "request_failed_or_disallowed"}

        path = self.save_binary(response.content, self.FILENAME)
        size_mb = round(path.stat().st_size / 1e6, 2)
        self.logger.info("Saved %s (%.2f MB)", path, size_mb)

        return {
            "source": self.source_name,
            "status": "ok",
            "path": str(path),
            "size_mb": size_mb,
        }


if __name__ == "__main__":
    collector = WelfakeCollector(
        output_root=Path(__file__).resolve().parents[3] / "datasets" / "raw" / "english",
        user_agent="Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; +mailto:research@example.edu)",
    )
    print(collector.collect())
