"""Collector for the ISOT Fake News Dataset — Module 1.

University of Victoria ISOT research lab, ~45,000 articles across two
files (`Fake.csv`, `True.csv` — the filename itself is the label, no
ambiguity). Official direct zip download, no login required — confirmed
real and working (see configs/module1_sources.yaml).

Known finding from this project's earlier collection run: ISOT's
article content turned out to be ~100% duplicated inside WELFake once
both are merged and deduplicated (WELFake's own construction already
folds in ISOT-like source material). This collector still downloads
ISOT independently — the duplication is something Module 1's
deduplication step surfaces and reports on, not something to silently
skip collecting.

Output:
  datasets/raw/english/isot/News-_dataset.zip
  datasets/raw/english/isot/extracted/Fake.csv
  datasets/raw/english/isot/extracted/True.csv
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from src.data_collection.base_collector import BaseCollector


class IsotCollector(BaseCollector):
    source_name = "isot"

    URL = "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip"
    ZIP_FILENAME = "News-_dataset.zip"
    EXTRACT_SUBDIR = "extracted"

    def collect(self, force: bool = False, **kwargs: Any) -> dict[str, Any]:
        extract_dir = self.output_dir / self.EXTRACT_SUBDIR
        fake_csv = extract_dir / "Fake.csv"
        true_csv = extract_dir / "True.csv"

        if not force and fake_csv.exists() and true_csv.exists():
            self.logger.info("ISOT already extracted at %s; skipping. Pass force=True to re-download.", extract_dir)
            return {
                "source": self.source_name,
                "status": "skipped_existing",
                "fake_csv": str(fake_csv),
                "true_csv": str(true_csv),
            }

        self.logger.info("Downloading ISOT from %s", self.URL)
        response = self.fetch(self.URL)
        if response is None:
            return {"source": self.source_name, "status": "failed", "reason": "request_failed_or_disallowed"}

        zip_path = self.save_binary(response.content, self.ZIP_FILENAME)
        zip_size_mb = round(zip_path.stat().st_size / 1e6, 2)
        self.logger.info("Saved %s (%.2f MB)", zip_path, zip_size_mb)

        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_files: list[str] = []
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            for member in archive.namelist():
                member_name = Path(member).name
                if not member_name.lower().endswith(".csv"):
                    continue
                target = extract_dir / member_name
                target.write_bytes(archive.read(member))
                extracted_files.append(str(target))
                self.logger.info("Extracted %s", target)

        return {
            "source": self.source_name,
            "status": "ok",
            "zip_path": str(zip_path),
            "zip_size_mb": zip_size_mb,
            "extracted_files": extracted_files,
        }


if __name__ == "__main__":
    collector = IsotCollector(
        output_root=Path(__file__).resolve().parents[3] / "datasets" / "raw" / "english",
        user_agent="Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; +mailto:research@example.edu)",
    )
    print(collector.collect())
