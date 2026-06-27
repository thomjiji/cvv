"""Hash file writers for TeraCopy (.xxh) and ASC MHL formats."""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


class HashFileWriter:
    """Generates hash files in TeraCopy (.xxh) and ASC MHL formats."""

    ALGO_DISPLAY = {
        "xxh3_64": "xxHash3-64",
        "xxh64": "xxHash-64",
        "md5": "MD5",
        "sha1": "SHA-1",
        "sha256": "SHA-256",
    }

    @staticmethod
    def write_xxh(
        entries: list[tuple[Path, str, int]],
        output_path: Path,
        hash_algorithm: str,
    ) -> Path:
        """Write a TeraCopy-compatible hash file (.xxh/.md5/.sha1/.sha256)."""
        algo_name = HashFileWriter.ALGO_DISPLAY.get(hash_algorithm, hash_algorithm)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"; {algo_name} checksums created by cvv\n")
            f.write(";\n\n")
            for rel_path, hash_hex, _size in entries:
                f.write(f"{hash_hex.upper()} *{rel_path.as_posix()}\n")
        return output_path

    @staticmethod
    def write_mhl(
        entries: list[tuple[Path, str, int]],
        output_dir: Path,
        hash_algorithm: str,
        source_name: str,
    ) -> Path:
        """Write an ASC MHL XML file in ascmhl/ subdirectory."""
        ascmhl_dir = output_dir / "ascmhl"
        ascmhl_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d_%H%M%S")
        output_path = ascmhl_dir / f"0001_{source_name}_{timestamp}.mhl"

        root = ET.Element("hashlist", version="2.0")

        creator = ET.SubElement(root, "creatorinfo")
        ET.SubElement(creator, "creationtool").text = "cvv 0.0.1"
        ET.SubElement(creator, "creationdate").text = now.isoformat()

        hashes_elem = ET.SubElement(root, "hashes")
        for rel_path, hash_hex, size in entries:
            hash_elem = ET.SubElement(hashes_elem, "hash")
            path_elem = ET.SubElement(hash_elem, "path", size=str(size))
            path_elem.text = rel_path.as_posix()
            ET.SubElement(hash_elem, hash_algorithm).text = hash_hex.lower()

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="UTF-8", xml_declaration=True)

        return output_path
