"""CLI entry for running a single-video pipeline job in a separate process.

The dashboard spawns this with:
    python web/single_video_job.py <job_id> <input_path> <label> <device>

It runs detect -> track -> re-ID -> identify -> render (see video_worker.py)
and writes the result JSON to web/outputs/<job_id>_result.json. Progress is
printed to stdout so the server can stream it into the job console.
"""

import sys

# script dir (web/) is on sys.path when launched as `python web/single_video_job.py`
from video_worker import main

if __name__ == "__main__":
    main()
