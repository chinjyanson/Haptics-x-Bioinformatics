"""
Visual Oddball Task (P300 Paradigm)
====================================
Displays a classic two-stimulus oddball sequence in a dedicated pygame window.

On macOS (and many Linux WMs) pygame's display subsystem must run on the
*main thread* of a process — it cannot be driven from a background thread
alongside a Tkinter / FreeSimpleGUI application in the same process.

Solution: the oddball task runs in a **child process** (via
``multiprocessing.Process``).  The child process owns its own main thread
so pygame is happy.  A ``multiprocessing.Queue`` carries stimulus-onset
events back to the parent so they can be timestamped and stored in the
shared SynchronizedDataStore.

Public API (used by main.py)
----------------------------
    oddball = OddballTask(data_store, lock)
    oddball.start()    # spawn child process + drain thread
    oddball.stop()     # terminate child, flush remaining events
"""

import multiprocessing
import queue
import random
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import SynchronizedDataStore

# ── Configurable parameters ────────────────────────────────────────────────
SOA_MS           = 600   # Stimulus onset asynchrony in ms (onset-to-onset)
STIM_DURATION_MS = 100   # How long the circle is visible (ms)
ODDBALL_PROB     = 0.20  # Probability of oddball on each trial (20 %)

WINDOW_W      = 400
WINDOW_H      = 400
WINDOW_TITLE  = "Fixation / Oddball Task"

BG_COLOR        = (30,  30,  30)
FIXATION_COLOR  = (220, 220, 220)
STANDARD_COLOR  = (100, 149, 237)   # blue  — standard
ODDBALL_COLOR   = (220,  50,  47)   # red   — target / oddball

CIRCLE_RADIUS = 80
FIXATION_SIZE = 24
# ───────────────────────────────────────────────────────────────────────────


# ── Child-process entry point ──────────────────────────────────────────────

def _oddball_process(event_queue: multiprocessing.Queue, stop_event: multiprocessing.Event):
    """
    Runs inside the child process.  Owns pygame entirely on its main thread.
    Puts dicts onto *event_queue* for each stimulus onset:
        {"stim_type": "standard"|"oddball", "trial_number": int, "timestamp": float}
    Exits when *stop_event* is set or the window is closed.
    """
    import pygame  # import here so the parent never touches pygame display

    pygame.display.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(WINDOW_TITLE)

    font = pygame.font.SysFont("monospace", FIXATION_SIZE, bold=True)
    cx, cy = WINDOW_W // 2, WINDOW_H // 2
    clock  = pygame.time.Clock()

    trial_number = 0

    def show_fixation():
        screen.fill(BG_COLOR)
        cross = font.render("+", True, FIXATION_COLOR)
        screen.blit(cross, cross.get_rect(center=(cx, cy)))
        pygame.display.flip()

    def wait_ms(duration_ms):
        """Sleep for duration_ms while pumping events; return True if stop requested."""
        deadline = time.monotonic() + duration_ms / 1000.0
        while time.monotonic() < deadline:
            if stop_event.is_set():
                return True
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    stop_event.set()
                    return True
            clock.tick(200)   # cap at 200 Hz poll
        return False

    try:
        while not stop_event.is_set():
            # ISI — show fixation cross
            show_fixation()
            if wait_ms(SOA_MS - STIM_DURATION_MS):
                break

            # Decide stimulus type
            is_oddball = random.random() < ODDBALL_PROB
            colour     = ODDBALL_COLOR if is_oddball else STANDARD_COLOR
            stim_label = "oddball" if is_oddball else "standard"

            # Draw stimulus
            screen.fill(BG_COLOR)
            pygame.draw.circle(screen, colour, (cx, cy), CIRCLE_RADIUS)

            # Capture timestamp right before flip for best accuracy
            onset_time = time.time()
            pygame.display.flip()

            # Send event to parent
            event_queue.put({
                "stim_type":    stim_label,
                "trial_number": trial_number,
                "timestamp":    onset_time,
            })
            trial_number += 1

            # Stimulus on-time
            if wait_ms(STIM_DURATION_MS):
                break

    finally:
        pygame.font.quit()
        pygame.display.quit()


# ── Parent-side class ──────────────────────────────────────────────────────

class OddballTask:
    """
    Manages the child pygame process and drains its event queue into the
    shared SynchronizedDataStore.
    """

    def __init__(self, data_store: "SynchronizedDataStore", lock: threading.Lock):
        self._data_store = data_store
        self._lock       = lock
        self._proc: multiprocessing.Process | None = None
        self._queue: multiprocessing.Queue | None  = None
        self._stop_event: multiprocessing.Event | None = None
        self._drain_thread: threading.Thread | None    = None
        self._drain_stop   = threading.Event()

    # ------------------------------------------------------------------
    def start(self):
        """Spawn the child process and start draining its event queue."""
        ctx = multiprocessing.get_context("spawn")   # safest on macOS
        self._queue      = ctx.Queue()
        self._stop_event = ctx.Event()

        self._proc = ctx.Process(
            target=_oddball_process,
            args=(self._queue, self._stop_event),
            daemon=True,
            name="OddballProcess",
        )
        self._proc.start()

        # Drain thread: reads events from the queue and writes TaskMarkers
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_queue,
            daemon=True,
            name="OddballDrain",
        )
        self._drain_thread.start()

        print("[Oddball] Task started.")

    def stop(self):
        """Stop the child process and flush any remaining queued events."""
        # Signal the child to exit
        if self._stop_event is not None:
            self._stop_event.set()

        # Give child a moment to exit cleanly, then terminate
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)

        # Stop the drain thread
        self._drain_stop.set()
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=2.0)

        print("[Oddball] Task stopped.")

    # ------------------------------------------------------------------
    def _drain_queue(self):
        """
        Background thread that continuously reads stimulus events from the
        child process queue and writes them to the data store as TaskMarkers.
        """
        while not self._drain_stop.is_set():
            try:
                ev = self._queue.get(timeout=0.05)
            except Exception:
                continue

            stim_label   = ev["stim_type"]
            trial_number = ev["trial_number"]
            onset_time   = ev["timestamp"]
            event_name   = "oddball_onset" if stim_label == "oddball" else "standard_onset"

            with self._lock:
                self._data_store.add_task_marker(
                    timestamp=onset_time,
                    task_number=0,   # 0 = oddball event, distinct from haptic tasks 1-5
                    event=event_name,
                    extra={
                        "stim_type":    stim_label,
                        "trial_number": trial_number,
                    },
                )
            print(f"[Oddball] trial {trial_number + 1:04d}: {stim_label}")

        # Flush any events that arrived after drain_stop was set
        if self._queue is not None:
            while True:
                try:
                    ev = self._queue.get_nowait()
                except Exception:
                    break
                stim_label   = ev["stim_type"]
                trial_number = ev["trial_number"]
                onset_time   = ev["timestamp"]
                event_name   = "oddball_onset" if stim_label == "oddball" else "standard_onset"
                with self._lock:
                    self._data_store.add_task_marker(
                        timestamp=onset_time,
                        task_number=0,
                        event=event_name,
                        extra={
                            "stim_type":    stim_label,
                            "trial_number": trial_number,
                        },
                    )
