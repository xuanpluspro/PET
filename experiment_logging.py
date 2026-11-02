"""Mirror terminal output into the experiment directory (standard library only)."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import os
import sys
import traceback


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def experiment_console(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Separate files avoid concurrent writes in distributed runs.
    rank = int(os.environ.get("RANK", "0"))
    filename = "console.log" if rank == 0 else f"console.rank{rank}.log"
    with (output_dir / filename).open("a", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                yield
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)
