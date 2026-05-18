"""
Standalone entry point — runs ONLY the Vibrations feedback session.

Equivalent to `python main.py` but with just the vibrations mode. Performs the
full setup ceremony (participant ID, consent, device connection, EEG signal
check) and then runs a single session, saving data under
data/{pid}/session_{timestamp}_vibrations_*.csv (same scheme as main.py).
"""

from main import run_experiment


if __name__ == "__main__":
    run_experiment(["vibrations"])
