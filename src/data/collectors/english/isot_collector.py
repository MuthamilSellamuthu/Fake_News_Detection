"""Collector for the ISOT Fake News Dataset (English).

~45,000 articles (Fake.csv + True.csv) curated by the ISOT research lab,
University of Victoria, from legitimate news sites and sources flagged
unreliable by Politifact.

Primary source: official direct .zip download (no login required):
  https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

Fallback: Kaggle mirror `clmentbisaillon/fake-and-real-news-dataset`,
via the `kaggle` CLI, for users who already have credentials configured.

Output:
  data/raw/english/isot/News-_dataset.zip   (original archive, kept for provenance)
  data/raw/english/isot/extracted/Fake.csv
  data/raw/english/isot/extracted/True.csv

Extraction just unzips — no column edits, relabeling, or merging happens
here.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from src.data.collectors.base_collector import BaseCollector


class IsotCollector(BaseCollector):
    source_name = "english/isot"

    DEFAULT_URL = (
        "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/"
        "sites/7295/2023/03/News-_dataset.zip"
    )
    DEFAULT_FILENAME = "News-_dataset.zip"
    KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"

    def collect(
        self,
        url: str = DEFAULT_URL,
        filename: str = DEFAULT_FILENAME,
        force: bool = False,
        extract: bool = True,
        use_kaggle_fallback: bool = True,
    ) -> dict[str, Any]:
        zip_path = self.output_dir / filename

        if zip_path.exists() and not force:
            self.logger.info("%s already exists, skipping download (pass force=True to re-download).", filename)
        else:
            self.logger.info("Downloading ISOT dataset from University of Victoria...")
            response = self.fetch(url, stream=True)

            if response is not None:
                with zip_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                self.logger.info("Saved %s (%.2f MB)", zip_path, zip_path.stat().st_size / (1024 * 1024))
            elif use_kaggle_fallback:
                self.logger.warning("Direct download failed, attempting Kaggle fallback...")
                return self._collect_via_kaggle()
            else:
                return {"source": self.source_name, "status": "failed"}

        result: dict[str, Any] = {"source": self.source_name, "status": "ok", "zip_path": str(zip_path)}

        if extract:
            extracted_dir = self.output_dir / "extracted"
            extracted_dir.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extracted_dir)
                extracted_files = [str(p) for p in extracted_dir.rglob("*") if p.is_file()]
                self.logger.info("Extracted %d file(s) to %s", len(extracted_files), extracted_dir)
                result["extracted_files"] = extracted_files
            except zipfile.BadZipFile as exc:
                self.logger.error("Downloaded file is not a valid zip: %s", exc)
                result["status"] = "failed"
                result["reason"] = "bad_zip"

        return result

    def _collect_via_kaggle(self) -> dict[str, Any]:
        """Fallback path using the Kaggle CLI (`pip install kaggle`).

        Requires Kaggle API credentials already configured. See:
        https://github.com/Kaggle/kaggle-api#api-credentials
        """
        if shutil.which("kaggle") is None:
            self.logger.error(
                "Kaggle CLI not found. Install with `pip install kaggle` and configure "
                "credentials (~/.kaggle/kaggle.json), then re-run."
            )
            return {"source": self.source_name, "status": "failed", "reason": "kaggle_cli_missing"}

        extracted_dir = self.output_dir / "extracted"
        extracted_dir.mkdir(exist_ok=True)
        try:
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    self.KAGGLE_DATASET,
                    "-p",
                    str(extracted_dir),
                    "--unzip",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            self.logger.error("Kaggle download failed: %s", exc)
            return {"source": self.source_name, "status": "failed", "reason": "kaggle_download_error"}

        return {
            "source": self.source_name,
            "status": "ok_via_kaggle",
            "path": str(extracted_dir),
            "note": f"Downloaded {self.KAGGLE_DATASET} via Kaggle API (expects Fake.csv / True.csv).",
        }


if __name__ == "__main__":
    collector = IsotCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect())
