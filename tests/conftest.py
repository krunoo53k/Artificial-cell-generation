import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))
