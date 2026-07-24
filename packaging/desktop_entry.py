from __future__ import annotations

import multiprocessing

from local_lm.desktop import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
