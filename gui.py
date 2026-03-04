"""FreeSimpleGUI screens for the BCI experiment."""

from __future__ import annotations

import random
import time
import threading
from typing import TYPE_CHECKING, Optional, List, Dict

if TYPE_CHECKING:
    from main import SynchronizedCollector, SynchronizedDataStore
    from haptics import HapticsController


def show_participant_screen() -> Optional[str]:
    """Show participant ID input screen. Returns participant ID or None if cancelled."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    layout = [
        [sg.Text('BCI Experiment', font=('Helvetica', 24, 'bold'), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Please enter your Participant ID:', font=('Helvetica', 14))],
        [sg.Input(key='-PARTICIPANT_ID-', font=('Helvetica', 14), size=(30, 1), justification='center')],
        [sg.Text('')],
        [sg.Text('', key='-ERROR-', text_color='red', font=('Helvetica', 10))],
        [sg.Button('Continue', size=(15, 1), font=('Helvetica', 12), bind_return_key=True),
         sg.Button('Exit', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window('BCI Experiment - Participant Registration', layout,
                       element_justification='center', finalize=True, size=(500, 300))

    participant_id = None

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        if event == 'Continue':
            pid = values['-PARTICIPANT_ID-'].strip()
            if not pid:
                window['-ERROR-'].update('Please enter a valid Participant ID')
            elif not pid.replace('_', '').replace('-', '').isalnum():
                window['-ERROR-'].update('Participant ID should only contain letters, numbers, - and _')
            else:
                participant_id = pid
                break

    window.close()
    return participant_id


def show_connection_screen(collector: 'SynchronizedCollector') -> bool:
    """Show device connection screen with status updates. Returns True if all connected."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    layout = [
        [sg.Text('Connecting to Devices', font=('Helvetica', 20, 'bold'), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Please ensure all devices are powered on and in range.', font=('Helvetica', 11))],
        [sg.Text('')],

        # Muse status
        [sg.Frame('Muse 2 EEG Headband', [
            [sg.Text('Status:', font=('Helvetica', 11)),
             sg.Text('Disabled' if not collector.use_muse else 'Waiting...',
                     key='-MUSE_STATUS-', font=('Helvetica', 11, 'bold'),
                     text_color='gray' if collector.use_muse else 'purple')],
            [sg.ProgressBar(100, orientation='h', size=(30, 20), key='-MUSE_PROGRESS-', bar_color=('blue', 'lightgray'))]
        ], font=('Helvetica', 12))],

        [sg.Text('')],

        # Polar status
        [sg.Frame('Polar H10 Heart Rate Monitor', [
            [sg.Text('Status:', font=('Helvetica', 11)),
             sg.Text('Disabled' if not collector.use_polar else 'Waiting...',
                     key='-POLAR_STATUS-', font=('Helvetica', 11, 'bold'),
                     text_color='gray' if collector.use_polar else 'purple')],
            [sg.ProgressBar(100, orientation='h', size=(30, 20), key='-POLAR_PROGRESS-', bar_color=('blue', 'lightgray'))]
        ], font=('Helvetica', 12))],

        [sg.Text('')],

        # GSR status
        [sg.Frame('eSense GSR (Skin Response)', [
            [sg.Text('Status:', font=('Helvetica', 11)),
             sg.Text('Disabled' if not collector.use_gsr else 'Waiting...',
                     key='-GSR_STATUS-', font=('Helvetica', 11, 'bold'),
                     text_color='gray' if collector.use_gsr else 'purple')],
            [sg.ProgressBar(100, orientation='h', size=(30, 20), key='-GSR_PROGRESS-', bar_color=('blue', 'lightgray'))]
        ], font=('Helvetica', 12))],

        [sg.Text('')],

        # Arduino status
        [sg.Frame('Arduino Uno R3 (Encoder + Vibration)', [
            [sg.Text('Status:', font=('Helvetica', 11)),
             sg.Text('Disabled' if not collector.use_arduino else 'Waiting...',
                     key='-ARDUINO_STATUS-', font=('Helvetica', 11, 'bold'),
                     text_color='gray' if collector.use_arduino else 'purple')],
            [sg.ProgressBar(100, orientation='h', size=(30, 20), key='-ARDUINO_PROGRESS-', bar_color=('blue', 'lightgray'))]
        ], font=('Helvetica', 12))],

        [sg.Text('')],
        [sg.Button('Connect Devices', size=(15, 1), font=('Helvetica', 12), key='-CONNECT-'),
         sg.Button('Skip Connection', size=(15, 1), font=('Helvetica', 12), key='-SKIP-', visible=False),
         sg.Button('Continue', size=(15, 1), font=('Helvetica', 12), key='-CONTINUE-', disabled=True),
         sg.Button('Cancel', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window('BCI Experiment - Device Connection', layout,
                       element_justification='center', finalize=True, size=(550, 650))

    muse_connected    = not collector.use_muse   # disabled counts as "ok"
    polar_connected   = not collector.use_polar
    gsr_connected     = not collector.use_gsr
    arduino_connected = not collector.use_arduino

    def connect_muse_thread():
        nonlocal muse_connected
        muse_connected = collector.connect_muse()

    def connect_polar_thread():
        nonlocal polar_connected
        polar_connected = collector.connect_polar()

    def connect_gsr_thread():
        nonlocal gsr_connected
        gsr_connected = collector.connect_gsr()

    def connect_arduino_thread():
        nonlocal arduino_connected
        if collector.arduino:
            arduino_connected = collector.arduino.connect()

    while True:
        event, values = window.read(timeout=100)

        if event in (sg.WIN_CLOSED, 'Cancel'):
            collector.disconnect_devices()
            window.close()
            return False

        if event == '-CONNECT-':
            window['-CONNECT-'].update(disabled=True)
            if collector.use_muse:
                window['-MUSE_STATUS-'].update('Connecting...', text_color='orange')
                window['-MUSE_PROGRESS-'].update(50)
            if collector.use_polar:
                window['-POLAR_STATUS-'].update('Connecting...', text_color='orange')
                window['-POLAR_PROGRESS-'].update(50)
            if collector.use_gsr:
                window['-GSR_STATUS-'].update('Connecting...', text_color='orange')
                window['-GSR_PROGRESS-'].update(50)
            if collector.use_arduino:
                window['-ARDUINO_STATUS-'].update('Connecting...', text_color='orange')
                window['-ARDUINO_PROGRESS-'].update(50)
            window.refresh()

            # Connect only enabled devices in threads
            threads = []
            if collector.use_muse:
                t = threading.Thread(target=connect_muse_thread)
                t.start()
                threads.append(t)
            if collector.use_polar:
                t = threading.Thread(target=connect_polar_thread)
                t.start()
                threads.append(t)
            if collector.use_gsr:
                t = threading.Thread(target=connect_gsr_thread)
                t.start()
                threads.append(t)
            if collector.use_arduino:
                t = threading.Thread(target=connect_arduino_thread)
                t.start()
                threads.append(t)

            # Wait for connections with GUI updates
            while any(t.is_alive() for t in threads):
                event2, _ = window.read(timeout=100)
                if event2 in (sg.WIN_CLOSED, 'Cancel'):
                    collector.disconnect_devices()
                    window.close()
                    return False

            # Update Muse status
            if not collector.use_muse:
                window['-MUSE_STATUS-'].update('Disabled', text_color='purple')
                window['-MUSE_PROGRESS-'].update(0)
            elif muse_connected:
                window['-MUSE_STATUS-'].update('Connected', text_color='green')
                window['-MUSE_PROGRESS-'].update(100)
            else:
                error_msg = collector.muse_error or 'Connection failed'
                window['-MUSE_STATUS-'].update(f'Failed: {error_msg[:30]}', text_color='red')
                window['-MUSE_PROGRESS-'].update(0)

            # Update Polar status
            if not collector.use_polar:
                window['-POLAR_STATUS-'].update('Disabled', text_color='purple')
                window['-POLAR_PROGRESS-'].update(0)
            elif polar_connected:
                window['-POLAR_STATUS-'].update('Connected', text_color='green')
                window['-POLAR_PROGRESS-'].update(100)
            else:
                error_msg = collector.polar_error or 'Connection failed'
                window['-POLAR_STATUS-'].update(f'Failed: {error_msg[:30]}', text_color='red')
                window['-POLAR_PROGRESS-'].update(0)

            # Update GSR status
            if not collector.use_gsr:
                window['-GSR_STATUS-'].update('Disabled', text_color='purple')
                window['-GSR_PROGRESS-'].update(0)
            elif gsr_connected:
                window['-GSR_STATUS-'].update('Connected', text_color='green')
                window['-GSR_PROGRESS-'].update(100)
            else:
                error_msg = collector.gsr_error or 'Connection failed'
                window['-GSR_STATUS-'].update(f'Failed: {error_msg[:30]}', text_color='red')
                window['-GSR_PROGRESS-'].update(0)

            # Update Arduino status
            if not collector.use_arduino:
                window['-ARDUINO_STATUS-'].update('Disabled', text_color='purple')
                window['-ARDUINO_PROGRESS-'].update(0)
            elif arduino_connected:
                window['-ARDUINO_STATUS-'].update('Connected', text_color='green')
                window['-ARDUINO_PROGRESS-'].update(100)
            else:
                error_msg = collector.arduino_error or 'Connection failed'
                window['-ARDUINO_STATUS-'].update(f'Failed: {error_msg[:30]}', text_color='red')
                window['-ARDUINO_PROGRESS-'].update(0)

            # Enable continue if all enabled devices connected
            if muse_connected and polar_connected and gsr_connected and arduino_connected:
                window['-CONTINUE-'].update(disabled=False)
            else:
                window['-CONNECT-'].update(disabled=False, text='Retry Connection')
                window['-SKIP-'].update(visible=True)

        if event == '-SKIP-':
            # Allow continuing with partial connection for testing
            window.close()
            return True

        if event == '-CONTINUE-':
            window.close()
            return True

    window.close()
    return False


def show_consent_screen() -> bool:
    """Show consent form screen. Returns True if consent given."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    consent_text = """
INFORMED CONSENT FOR BCI EXPERIMENT

Purpose of the Study:
This experiment collects brain activity (EEG) and heart rate data to evaluate
cognitive load in multimodal user interfaces.

Data Collection:
- EEG signals from the Muse 2 headband (4 channels)
- Heart rate and heart rate variability from the Polar H10

Your Rights:
- Participation is voluntary
- You may withdraw at any time without penalty
- Your data will be anonymized and stored securely
- Data will only be used for research purposes

Procedure:
1. Devices will record your physiological data
2. You will perform tasks as instructed
3. You may stop at any time by pressing the Stop button

Duration:
The experiment will continue until you choose to stop it or you have completed the task.

By clicking "I Agree", you confirm that:
- You have read and understood the above information
- You voluntarily agree to participate
- You are at least 18 years of age
"""

    layout = [
        [sg.Text('Consent Form', font=('Helvetica', 20, 'bold'), justification='center', expand_x=True)],
        [sg.Multiline(consent_text, size=(60, 20), font=('Helvetica', 10), disabled=True,
                      background_color='white', key='-CONSENT_TEXT-')],
        [sg.Text('')],
        [sg.Checkbox('I have read and understood the consent form', key='-READ-', font=('Helvetica', 11))],
        [sg.Checkbox('I voluntarily agree to participate in this study', key='-AGREE-', font=('Helvetica', 11))],
        [sg.Text('')],
        [sg.Button('I Agree & Continue', size=(18, 1), font=('Helvetica', 12), key='-CONSENT-', disabled=True),
         sg.Button('I Do Not Consent', size=(18, 1), font=('Helvetica', 12), key='-NO_CONSENT-')]
    ]

    window = sg.Window('BCI Experiment - Consent Form', layout,
                       element_justification='center', finalize=True, size=(600, 550))

    while True:
        event, values = window.read(timeout=100)

        if event in (sg.WIN_CLOSED, '-NO_CONSENT-'):
            window.close()
            return False

        # Enable consent button only when both checkboxes are checked
        both_checked = values['-READ-'] and values['-AGREE-']
        window['-CONSENT-'].update(disabled=not both_checked)

        if event == '-CONSENT-' and both_checked:
            window.close()
            return True

    window.close()
    return False


def show_reconnect_screen(collector: 'SynchronizedCollector') -> bool:
    """
    Block until all device threads signal they are ready (or user aborts).
    Show live status for each device so the researcher can see progress.
    Returns True when all devices are ready, False if aborted.
    """
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    def status_row(label, key_status, disabled=False):
        init_text  = 'Disabled'  if disabled else 'Connecting...'
        init_color = 'purple'    if disabled else 'orange'
        return [
            sg.Text(f'{label}:', font=('Helvetica', 11), size=(22, 1)),
            sg.Text(init_text, key=key_status, font=('Helvetica', 11, 'bold'),
                    text_color=init_color, size=(18, 1)),
        ]

    layout = [
        [sg.Text('Reconnecting Devices', font=('Helvetica', 20, 'bold'),
                 justification='center', expand_x=True)],
        [sg.Text('Please wait while all sensors reconnect...',
                 font=('Helvetica', 11), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Frame('Device Status', [
            status_row('Muse 2 EEG',      '-MUSE_S-',    disabled=not collector.use_muse),
            status_row('Polar H10 HR',    '-POLAR_S-',   disabled=not collector.use_polar),
            status_row('eSense GSR',      '-GSR_S-',     disabled=not collector.use_gsr),
            status_row('Arduino Uno R3',  '-ARDUINO_S-', disabled=not collector.use_arduino),
        ], font=('Helvetica', 12), pad=(10, 10))],
        [sg.Text('')],
        [sg.Button('Abort', size=(12, 1), font=('Helvetica', 12),
                   button_color=('white', 'red'), key='-ABORT-')],
    ]

    window = sg.Window('BCI Experiment - Reconnecting', layout,
                       element_justification='center', finalize=True, size=(440, 280),
                       disable_close=True)

    result = False
    while True:
        event, _ = window.read(timeout=100)

        if event == '-ABORT-':
            if sg.popup_yes_no('Abort the experiment?',
                               title='Confirm Abort', font=('Helvetica', 11)) == 'Yes':
                break

        muse_ready    = (not collector.use_muse)    or collector._muse_ready.is_set()
        polar_ready   = (not collector.use_polar)   or collector._polar_ready.is_set()
        gsr_ready     = (not collector.use_gsr)     or collector._gsr_ready.is_set()
        arduino_ready = (not collector.use_arduino) or collector._arduino_ready.is_set()

        for key, ready, enabled in [
            ('-MUSE_S-',    muse_ready,    collector.use_muse),
            ('-POLAR_S-',   polar_ready,   collector.use_polar),
            ('-GSR_S-',     gsr_ready,     collector.use_gsr),
            ('-ARDUINO_S-', arduino_ready, collector.use_arduino),
        ]:
            if not enabled:
                continue  # leave label as 'Disabled'
            elem = window[key]
            if elem is not None:
                if ready:
                    elem.update(value='Ready', text_color='green')
                else:
                    elem.update(value='Connecting...', text_color='orange')

        if muse_ready and polar_ready and gsr_ready and arduino_ready:
            result = True
            break

    window.close()
    return result


def show_countdown_screen() -> bool:
    """Show countdown before experiment starts. Returns True when complete."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    layout = [
        [sg.Text('Get Ready!', font=('Helvetica', 24, 'bold'), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Experiment starting in:', font=('Helvetica', 14), justification='center', expand_x=True)],
        [sg.Text('3', font=('Helvetica', 72, 'bold'), key='-COUNTDOWN-', justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Please remain still and relaxed.', font=('Helvetica', 12), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Button('Cancel', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window('BCI Experiment - Starting', layout,
                       element_justification='center', finalize=True, size=(400, 350))

    for i in range(3, 0, -1):
        window['-COUNTDOWN-'].update(str(i))
        window.refresh()

        # Wait 1 second, checking for cancel
        start = time.time()
        while time.time() - start < 1.0:
            event, _ = window.read(timeout=50)
            if event in (sg.WIN_CLOSED, 'Cancel'):
                window.close()
                return False

    window['-COUNTDOWN-'].update('GO!')
    window.refresh()
    time.sleep(0.5)

    window.close()
    return True


def show_experiment_screen(collector: 'SynchronizedCollector', output_path: str,
                           threads=None, haptics: 'Optional[HapticsController]' = None,
                           haptic_targets: Optional[List[int]] = None) -> None:
    """
    Show experiment running screen with stop button. Runs until user stops.

    Args:
        collector:      The synchronized data collector.
        output_path:    Base path to save data to.
        threads:        Optional (muse_thread, polar_thread, gsr_thread, arduino_thread)
                        if already started before this call. If None, threads are
                        started here (legacy behaviour).
        haptics:        Optional HapticsController for audio/vibration feedback.
                        If None, no feedback is given.
        haptic_targets: List of target tick positions, one per task. If None or
                        shorter than TASKS_PER_DEVICE, missing entries default to 0.
    """
    import FreeSimpleGUI as sg
    from main import TASKS_PER_DEVICE, start_collection_threads

    sg.theme('LightBlue2')

    # ── Oddball stimulus constants ─────────────────────────────────────────
    _OB_SOA_MS    = 600   # stimulus onset asynchrony (ms)
    _OB_STIM_MS   = 100   # stimulus visible duration (ms)
    _OB_PROB      = 0.20  # oddball probability
    _OB_CX        = 150   # canvas centre x
    _OB_CY        = 150   # canvas centre y
    _OB_R         = 60    # circle radius (pixels)
    _OB_BG        = '#1e1e1e'
    _OB_FIX_COL   = '#dcdcdc'
    _OB_STD_COL   = '#6495ed'   # blue — standard
    _OB_ODD_COL   = '#dc322f'   # red  — oddball

    layout = [
        [sg.Text('Experiment in Progress', font=('Helvetica', 20, 'bold'), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Initializing devices...', key='-STATUS-', font=('Helvetica', 12))],
        [sg.Text('')],

        [sg.Column([
            # Left: recording status
            [sg.Frame('Recording Status', [
                [sg.Text('Duration:', font=('Helvetica', 11)),
                 sg.Text('00:00:00', key='-DURATION-', font=('Helvetica', 14, 'bold'))],
                [sg.Text('EEG Samples:', font=('Helvetica', 11)),
                 sg.Text('0', key='-EEG_COUNT-', font=('Helvetica', 11, 'bold'))],
                [sg.Text('HR Samples:', font=('Helvetica', 11)),
                 sg.Text('0', key='-HR_COUNT-', font=('Helvetica', 11, 'bold'))],
                [sg.Text('GSR Samples:', font=('Helvetica', 11)),
                 sg.Text('0', key='-GSR_COUNT-', font=('Helvetica', 11, 'bold'))],
                [sg.Text('Arduino Events:', font=('Helvetica', 11)),
                 sg.Text('0', key='-ARDUINO_COUNT-', font=('Helvetica', 11, 'bold'))],
                [sg.Text('Latest HR:', font=('Helvetica', 11)),
                 sg.Text('-- bpm', key='-LATEST_HR-', font=('Helvetica', 11, 'bold'))],
                [sg.Text('Latest GSR:', font=('Helvetica', 11)),
                 sg.Text('-- µS', key='-LATEST_GSR-', font=('Helvetica', 11, 'bold'))],
            ], font=('Helvetica', 12))],
            [sg.Text('Current Task:', font=('Helvetica', 11)),
             sg.Text(f'1 / {TASKS_PER_DEVICE}', key='-TASK_NUM-', font=('Helvetica', 11, 'bold'))],
            [sg.Text('Press Right Arrow to end current task and move to next.', font=('Helvetica', 10))],
            [sg.Text('')],
            [sg.Text(f'Session ends automatically after task {TASKS_PER_DEVICE}.', font=('Helvetica', 10))],
            [sg.Text('')],
            [sg.Button('STOP EXPERIMENT', size=(20, 2), font=('Helvetica', 14, 'bold'),
                       button_color=('white', 'red'), key='-STOP-', disabled=True)],
        ], vertical_alignment='top'),

         sg.Column([
            # Right: oddball stimulus canvas
            [sg.Text('Oddball Task', font=('Helvetica', 12, 'bold'), justification='center')],
            [sg.Text('Count the RED circles', font=('Helvetica', 10), justification='center')],
            [sg.Graph(canvas_size=(300, 300), graph_bottom_left=(0, 300),
                      graph_top_right=(300, 0),
                      background_color=_OB_BG, key='-ODDBALL-')],
        ], vertical_alignment='top', element_justification='center')],
    ]

    window = sg.Window('BCI Experiment - Recording', layout,
                       element_justification='center', finalize=True, size=(820, 520),
                       disable_close=True)  # Prevent accidental close

    # Bind right arrow for task transitions (window must be focused)
    window.bind("<Right>", "-NEXT_TASK-")

    current_task = 1

    # Normalise haptic_targets to a full-length list (pad with 0 if needed)
    _targets: List[int] = list(haptic_targets) if haptic_targets else []
    while len(_targets) < TASKS_PER_DEVICE:
        _targets.append(0)

    # Track last arduino_data index so we only feed new encoder events to haptics
    _last_arduino_idx: int = 0

    if threads is not None:
        # Threads were pre-started before the countdown — reuse them.
        muse_thread, polar_thread, gsr_thread, arduino_thread = threads
    else:
        # Legacy path: start threads now (used when called without pre-starting).
        muse_thread, polar_thread, gsr_thread, arduino_thread = start_collection_threads(collector)

    # Wait for all enabled devices to be ready
    if collector.use_muse:
        window['-STATUS-'].update('Waiting for Muse to be ready...')
        window.refresh()

        while not collector._muse_ready.is_set():
            event, _ = window.read(timeout=100)
            if event == '-STOP-' or event == sg.WIN_CLOSED:
                collector._stop_event.set()
                window.close()
                return

    if collector.use_polar:
        window['-STATUS-'].update('Waiting for Polar to be ready...')
        window.refresh()

        while not collector._polar_ready.is_set():
            event, _ = window.read(timeout=100)
            if event == '-STOP-' or event == sg.WIN_CLOSED:
                collector._stop_event.set()
                window.close()
                return

    if collector.use_gsr:
        window['-STATUS-'].update('Waiting for GSR to be ready...')
        window.refresh()

        while not collector._gsr_ready.is_set():
            event, _ = window.read(timeout=100)
            if event == '-STOP-' or event == sg.WIN_CLOSED:
                collector._stop_event.set()
                window.close()
                return

    if collector.use_arduino:
        window['-STATUS-'].update('Waiting for Arduino to be ready...')
        window.refresh()

        while not collector._arduino_ready.is_set():
            event, _ = window.read(timeout=100)
            if event == '-STOP-' or event == sg.WIN_CLOSED:
                collector._stop_event.set()
                window.close()
                return

    # All devices ready - set session_start and signal to start recording
    window['-STATUS-'].update('Starting synchronized recording...')
    window.refresh()

    # Set the session start time RIGHT NOW - this is time=0
    collector.data_store.session_start = time.time()

    # Signal all threads to start recording simultaneously
    collector._start_recording.set()

    window['-STATUS-'].update('Data is being recorded...')
    window['-STOP-'].update(disabled=False)
    window.refresh()

    print(f"[Sync] All devices ready. Recording started at t=0")

    # Start haptics and arm the first task target
    if haptics is not None:
        haptics.start()
        haptics.set_target(_targets[0])
        print(f"[Haptics] Session started. Task 1 target = {_targets[0]} ticks.")

    # ── Oddball state machine initialisation ──────────────────────────────
    # States: 'fixation' → show "+" and wait ISI, then → 'stimulus'
    #         'stimulus' → show circle, then → 'fixation'
    _ob_canvas      = window['-ODDBALL-']
    _ob_state       = 'fixation'           # current state
    _ob_deadline    = time.monotonic()     # when to transition next
    _ob_trial       = 0
    _ob_is_oddball  = False

    def _ob_draw_fixation():
        _ob_canvas.erase()
        _ob_canvas.draw_text('+', (_OB_CX, _OB_CY), color=_OB_FIX_COL,
                             font=('Helvetica', 36, 'bold'))

    def _ob_draw_stimulus(is_oddball: bool):
        colour = _OB_ODD_COL if is_oddball else _OB_STD_COL
        _ob_canvas.erase()
        _ob_canvas.draw_circle((_OB_CX, _OB_CY), _OB_R, fill_color=colour, line_color=colour)

    _ob_draw_fixation()

    # Update loop — 50 ms timeout gives ~20 Hz tick for oddball state machine
    while True:
        event, _ = window.read(timeout=50)

        # Feed new encoder deltas to haptics
        if haptics is not None:
            with collector._lock:
                new_events = collector.data_store.arduino_data[_last_arduino_idx:]
                _last_arduino_idx += len(new_events)
            for ev in new_events:
                if ev.event_type == "encoder":
                    haptics.update_encoder(ev.data.get("delta", 0))

        if event == "-NEXT_TASK-":
            # Record error before advancing
            enc_error = (haptics.current_position - haptics.target) if haptics else 0
            enc_pos   = haptics.current_position if haptics else 0
            enc_target = haptics.target if haptics else 0

            timestamp = time.time()
            with collector._lock:
                collector.data_store.add_task_marker(
                    timestamp, current_task, "task_end",
                    extra={
                        "encoder_position": enc_pos,
                        "target":           enc_target,
                        "encoder_error":    enc_error,
                    }
                )
            print(f"[Task] Ended task {current_task}  encoder_error={enc_error:+d}")

            if current_task >= TASKS_PER_DEVICE:
                # All tasks done — end the session automatically
                print(f"[Task] All {TASKS_PER_DEVICE} tasks complete. Ending session.")
                break

            current_task += 1
            # Arm next task target
            if haptics is not None:
                haptics.set_target(_targets[current_task - 1])
                print(f"[Haptics] Task {current_task} target = {_targets[current_task - 1]} ticks.")
            elem = window['-TASK_NUM-']
            if elem is not None:
                elem.update(value=f'{current_task} / {TASKS_PER_DEVICE}')

        if event == '-STOP-':
            # Manual early stop with confirmation
            if sg.popup_yes_no('Are you sure you want to stop the experiment early?',
                              title='Confirm Stop', font=('Helvetica', 11)) == 'Yes':
                break

        # Update display
        elapsed = time.time() - collector.data_store.session_start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        window['-DURATION-'].update(f'{hours:02d}:{minutes:02d}:{seconds:02d}')

        with collector._lock:
            eeg_count = len(collector.data_store.eeg_data)
            hr_count = len(collector.data_store.hr_data)
            gsr_count = len(collector.data_store.gsr_data)
            arduino_count = len(collector.data_store.arduino_data)
            if collector.data_store.hr_data:
                latest_hr = collector.data_store.hr_data[-1].heart_rate
            else:
                latest_hr = None
            if collector.data_store.gsr_data:
                latest_gsr = collector.data_store.gsr_data[-1].gsr_uS
            else:
                latest_gsr = None

        window['-EEG_COUNT-'].update(str(eeg_count))
        window['-HR_COUNT-'].update(str(hr_count))
        window['-GSR_COUNT-'].update(str(gsr_count))
        window['-ARDUINO_COUNT-'].update(str(arduino_count))
        if latest_hr:
            window['-LATEST_HR-'].update(f'{latest_hr} bpm')
        if latest_gsr is not None:
            window['-LATEST_GSR-'].update(f'{latest_gsr:.4f} µS')

        # ── Oddball state machine tick ─────────────────────────────────────
        now = time.monotonic()
        if _ob_state == 'fixation' and now >= _ob_deadline:
            # ISI elapsed — pick and show next stimulus
            _ob_is_oddball = random.random() < _OB_PROB
            _ob_draw_stimulus(_ob_is_oddball)
            onset_time = time.time()
            stim_label = "oddball" if _ob_is_oddball else "standard"
            with collector._lock:
                collector.data_store.add_task_marker(
                    timestamp=onset_time,
                    task_number=0,
                    event="oddball_onset" if _ob_is_oddball else "standard_onset",
                    extra={"stim_type": stim_label, "trial_number": _ob_trial},
                )
            print(f"[Oddball] trial {_ob_trial + 1:04d}: {stim_label}")
            _ob_trial    += 1
            _ob_state     = 'stimulus'
            _ob_deadline  = now + _OB_STIM_MS / 1000.0

        elif _ob_state == 'stimulus' and now >= _ob_deadline:
            # Stimulus duration elapsed — return to fixation
            _ob_draw_fixation()
            _ob_state    = 'fixation'
            _ob_deadline = now + (_OB_SOA_MS - _OB_STIM_MS) / 1000.0

    # Stop haptics before threads so motors are zeroed and audio is silenced
    if haptics is not None:
        haptics.stop()

    # Stop collection
    collector._stop_event.set()
    window['-STOP-'].update('Stopping devices...', disabled=True)
    window.refresh()

    # Wait for threads to finish — Muse BrainFlow release can be slow, give it up to 15s
    if muse_thread is not None:
        muse_thread.join(timeout=15)
        if muse_thread.is_alive():
            print("[Muse] WARNING: thread did not exit cleanly within timeout.")

    if polar_thread is not None:
        polar_thread.join(timeout=10)
    if gsr_thread is not None:
        gsr_thread.join(timeout=10)
    if arduino_thread is not None:
        arduino_thread.join(timeout=5)
        if arduino_thread.is_alive():
            print("[Arduino] WARNING: thread did not exit cleanly within timeout.")

    # Give the Bluetooth stack a moment to fully release the BrainFlow session
    # before any subsequent prepare_session() call on the next device round.
    time.sleep(2.0)

    window['-STOP-'].update('Saving data...', disabled=True)
    window.refresh()

    # Save data
    collector.save_data(output_path)

    window.close()


def show_nasa_tlx_screen(device_name: str) -> Optional[Dict[str, int]]:
    """
    Show NASA TLX questionnaire for the given device condition.
    Returns a dict of dimension -> score (0-100), or None if cancelled.
    """
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    dimensions = [
        ('Mental Demand',    'How mentally demanding was the task?'),
        ('Physical Demand',  'How physically demanding was the task?'),
        ('Temporal Demand',  'How hurried or rushed was the pace of the task?'),
        ('Performance',      'How successful were you in accomplishing the task?'),
        ('Effort',           'How hard did you have to work to accomplish your level of performance?'),
        ('Frustration',      'How insecure, discouraged, irritated, stressed, and annoyed were you?'),
    ]

    slider_rows = []
    for dim, desc in dimensions:
        key = f'-{dim.upper().replace(" ", "_")}-'
        slider_rows += [
            [sg.Text(dim, font=('Helvetica', 12, 'bold'))],
            [sg.Text(desc, font=('Helvetica', 10), text_color='gray')],
            [sg.Text('Low', font=('Helvetica', 9)),
             sg.Slider(range=(0, 100), default_value=50, orientation='h',
                       size=(40, 20), key=key, font=('Helvetica', 10)),
             sg.Text('High', font=('Helvetica', 9))],
            [sg.Text('')],
        ]

    layout = [
        [sg.Text(f'NASA Task Load Index — {device_name}',
                 font=('Helvetica', 18, 'bold'), justification='center', expand_x=True)],
        [sg.Text('Please rate your experience with the device you just used.',
                 font=('Helvetica', 11), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Column(slider_rows, scrollable=True, vertical_scroll_only=True, size=(620, 420))],
        [sg.Text('')],
        [sg.Button('Submit', size=(15, 1), font=('Helvetica', 12), key='-SUBMIT-'),
         sg.Button('Cancel', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window(f'NASA TLX — {device_name}', layout,
                       element_justification='center', finalize=True, size=(680, 620))

    scores = None
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        if event == '-SUBMIT-':
            scores = {}
            for dim, _ in dimensions:
                key = f'-{dim.upper().replace(" ", "_")}-'
                scores[dim] = int(values[key])
            break

    window.close()
    return scores


def show_device_swap_screen(next_device: str) -> bool:
    """
    Show a screen prompting the researcher to swap to the next device.
    Returns True when ready to continue, False if cancelled.
    """
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    layout = [
        [sg.Text('Device Swap', font=('Helvetica', 24, 'bold'),
                 justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('The current session has ended.', font=('Helvetica', 13),
                 justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Frame('Next Step', [
            [sg.Text(f'Please swap to the next device:', font=('Helvetica', 12))],
            [sg.Text(f'   {next_device}', font=('Helvetica', 16, 'bold'), text_color='navy')],
        ], font=('Helvetica', 12))],
        [sg.Text('')],
        [sg.Text('When the new device is in place and the participant is ready,',
                 font=('Helvetica', 11))],
        [sg.Text('click Continue to proceed to the next session.',
                 font=('Helvetica', 11))],
        [sg.Text('')],
        [sg.Button('Continue to Next Session', size=(25, 2), font=('Helvetica', 12),
                   button_color=('white', 'green'), key='-CONTINUE-'),
         sg.Button('Abort Experiment', size=(18, 2), font=('Helvetica', 12),
                   button_color=('white', 'red'), key='-ABORT-')]
    ]

    window = sg.Window('BCI Experiment — Device Swap', layout,
                       element_justification='center', finalize=True, size=(540, 380),
                       disable_close=True)

    result = False
    while True:
        event, _ = window.read()
        if event == '-CONTINUE-':
            result = True
            break
        if event == '-ABORT-':
            if sg.popup_yes_no('Are you sure you want to abort the experiment?',
                               title='Confirm Abort', font=('Helvetica', 11)) == 'Yes':
                break

    window.close()
    return result


def show_completion_screen(sessions: List[Dict], output_base: str) -> None:
    """
    Show experiment completion summary for all three device sessions.

    Args:
        sessions: list of dicts with keys 'device', 'collector', 'output_path', 'tlx_scores'
        output_base: base output folder path shown to user
    """
    import FreeSimpleGUI as sg
    from main import SynchronizedDataStore

    sg.theme('LightBlue2')

    tlx_dim_order = ['Mental Demand', 'Physical Demand', 'Temporal Demand',
                     'Performance', 'Effort', 'Frustration']

    session_rows = []
    for s in sessions:
        device = s['device']
        data_store: SynchronizedDataStore = s['data_store']
        tlx: Optional[Dict[str, int]] = s['tlx_scores']

        last_ts = (data_store.eeg_data[-1].timestamp if data_store.eeg_data
                   else data_store.session_start)
        total_duration = data_store.get_relative_time(last_ts)
        hours = int(total_duration // 3600)
        mins = int((total_duration % 3600) // 60)
        secs = int(total_duration % 60)

        eeg_count = len(data_store.eeg_data)
        hr_count = len(data_store.hr_data)
        gsr_count = len(data_store.gsr_data)

        tlx_lines = []
        if tlx:
            for dim in tlx_dim_order:
                tlx_lines.append([sg.Text(f'  {dim}: {tlx[dim]}', font=('Helvetica', 10))])
            avg_tlx = sum(tlx.values()) / len(tlx)
            tlx_lines.append([sg.Text(f'  Average TLX: {avg_tlx:.1f}', font=('Helvetica', 10, 'bold'))])
        else:
            tlx_lines.append([sg.Text('  Not completed', font=('Helvetica', 10), text_color='red')])

        session_rows.append(
            sg.Frame(f'{device}', [
                [sg.Text(f'Duration: {hours:02d}:{mins:02d}:{secs:02d}  |  '
                         f'EEG: {eeg_count:,}  |  HR: {hr_count:,}  |  GSR: {gsr_count:,}',
                         font=('Helvetica', 10))],
                [sg.Text('NASA TLX Scores:', font=('Helvetica', 10, 'bold'))],
                *tlx_lines,
            ], font=('Helvetica', 11))
        )

    layout = [
        [sg.Text('Experiment Complete!', font=('Helvetica', 24, 'bold'),
                 text_color='green', justification='center', expand_x=True)],
        [sg.Text('Thank you for participating.', font=('Helvetica', 13),
                 justification='center', expand_x=True)],
        [sg.Text('')],
        *[[row] for row in session_rows],
        [sg.Text('')],
        [sg.Text(f'Data saved to: {output_base}', font=('Helvetica', 10))],
        [sg.Text('')],
        [sg.Button('Close', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window('BCI Experiment — Complete', layout,
                       element_justification='center', finalize=True, size=(650, 620))

    while True:
        event, _ = window.read()
        if event in (sg.WIN_CLOSED, 'Close'):
            break

    window.close()


def show_error_popup(message: str) -> None:
    """Show an error popup dialog."""
    import FreeSimpleGUI as sg
    sg.popup_error(message, title='Error')
