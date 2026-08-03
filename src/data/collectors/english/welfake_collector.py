"""Collector for the WELFake dataset (English).

WELFake merges four popular English fake-news corpora (Kaggle,
McIntire, Reuters, BuzzFeed Political) into 72,134 labeled articles.

Primary source: Zenodo (open, no account/API key required).
  https://zenodo.org/records/4561253

Fallback: Kaggle mirror, via the `kaggle` CLI/API, for users who
already have Kaggle credentials configured (~/.kaggle/kaggle.json or
KAGGLE_USERNAME / KAGGLE_KEY env vars). Not used unless the Zenodo
download fails or is explicitly requested.

Output: data/raw/english/welfake/WELFake_Dataset.csv (unmodified CSV,
as published). No cleaning, relabeling, or column changes happen here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.data.collectors.base_collector import BaseCollector


class WelfakeCollector(BaseCollector):
    source_name = "english/welfake"

    DEFAULT_URL = "https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1"
    DEFAULT_FILENAME = "WELFake_Dataset.csv"
    KAGGLE_DATASET = "saurabhshahane/fake-news-classification"

    def collect(
        self,
        url: str = DEFAULT_URL,
        filename: str = DEFAULT_FILENAME,
        force: bool = False,
        use_kaggle_fallback: bool = True,
    ) -> dict[str, Any]:
        if self.already_collected(filename) and not force:
            self.logger.info("%s already exists, skipping (pass force=True to re-download).", filename)
            return {"source": self.source_name, "status": "skipped", "path": str(self.output_dir / filename)}

        self.logger.info("Downloading WELFake dataset from Zenodo...")
        response = self.fetch(url, stream=True)

        if response is not None:
            path = self.output_dir / filename
            with path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            size_mb = path.stat().st_size / (1024 * 1024)
            self.logger.info("Saved %s (%.2f MB)", path, size_mb)
            return {"source": self.source_name, "status": "ok", "path": str(path), "size_mb": round(size_mb, 2)}

        if use_kaggle_fallback:
            self.logger.warning("Zenodo download failed, attempting Kaggle fallback...")
            return self._collect_via_kaggle(filename)

        return {"source": self.source_name, "status": "failed"}

    def _collect_via_kaggle(self, filename: str) -> dict[str, Any]:
        """Fallback path using the Kaggle CLI (`pip install kaggle`).

        Requires Kaggle API credentials to already be configured on the
        machine running this script. See:
        https://github.com/Kaggle/kaggle-api#api-credentials
        """
        if shutil.which("kaggle") is None:
            self.logger.error(
                "Kaggle CLI not found. Install with `pip install kaggle` and configure "
                "credentials (~/.kaggle/kaggle.json), then re-run."
            )
            return {"source": self.source_name, "status": "failed", "reason": "kaggle_cli_missing"}

        try:
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    self.KAGGLE_DATASET,
                    "-p",
                    str(self.output_dir),
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
            "path": str(self.output_dir),
            "note": f"Downloaded {self.KAGGLE_DATASET} via Kaggle API; verify filename matches {filename}.",
        }


if __name__ == "__main__":
    collector = WelfakeCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect())
