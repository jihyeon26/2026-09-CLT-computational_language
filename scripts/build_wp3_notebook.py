from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from wp3_notebook_cells import notebook_cells


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "stage1_wp3_open_models_contract.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nb["cells"] = notebook_cells()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
