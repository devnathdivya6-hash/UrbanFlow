"""
emergency_audio_handler.py
--------------------------
Integrates YAMNet-based siren detection (AudioDetector) into the
UrbanFlow traffic signal system.

Pipeline:
  audio chunk  →  AudioDetector.process_audio_chunk()
               →  confidence gate + cooldown
               →  density check
               →  gradually_clear_lane()  (if congested)
               →  give_green_signal()
"""

import time
import numpy as np
from audio_detector import AudioDetector


# ─────────────────────────────────────────────────
# Stub traffic-control functions
# (replace with real implementations in production)
# ─────────────────────────────────────────────────

def get_traffic_density(lane_id: int) -> int:
    """Return the current vehicle count for the given lane.
    Stub: returns a random value between 0 and 30 for simulation.
    """
    return np.random.randint(0, 31)


def gradually_clear_lane(lane_id: int) -> None:
    """Gradually clear a congested lane before giving green signal.
    Stub: simulates a brief clearing delay.
    """
    print(f"  [Signal] Gradually clearing Lane {lane_id}...")
    time.sleep(1.0)   # Replace with real phase-transition logic
    print(f"  [Signal] Lane {lane_id} cleared.")


def give_green_signal(lane_id: int) -> None:
    """Grant a green signal to the specified emergency lane.
    Stub: logs the action.
    """
    print(f"  [Signal] [GREEN] Signal granted to Lane {lane_id}.")


# ─────────────────────────────────────────────────
# Density threshold above which a lane is 'congested'
# ─────────────────────────────────────────────────
CONGESTION_THRESHOLD = 10


class EmergencyAudioHandler:
    """
    Wraps AudioDetector and adds:
      • a configurable confidence threshold
      • a cooldown window to prevent burst re-triggering
      • traffic-density-aware signal escalation
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        cooldown_seconds: float = 12.0,
    ):
        """
        Parameters
        ----------
        confidence_threshold : float
            Minimum YAMNet score required to treat a detection as valid.
        cooldown_seconds : float
            Minimum gap (in seconds) between two consecutive emergency overrides
            for the same lane.  Detections arriving inside the cooldown window
            are silently ignored.
        """
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds

        # Per-lane timestamp of the last successful trigger.
        # Using a dict so we can extend to arbitrary lane counts later.
        self._last_trigger_time: dict[int, float] = {}

        # Initialise the underlying YAMNet detector once (expensive).
        print("[AudioHandler] Initialising AudioDetector (YAMNet)...")
        self.detector = AudioDetector()
        print("[AudioHandler] AudioDetector ready.")

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def handle_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        lane_id: int,
    ) -> bool:
        """
        Process one audio chunk for a specific lane.

        Parameters
        ----------
        waveform    : 1-D numpy float32 array of audio samples.
        sample_rate : Sampling rate of the waveform (Hz).
        lane_id     : The traffic lane this audio feed is associated with.

        Returns
        -------
        bool  True if an emergency override was triggered, False otherwise.
        """
        # ── 1. Guard: reject obviously malformed input ────────────────
        if not self._is_valid_waveform(waveform):
            print(f"[AudioHandler] Lane {lane_id}: invalid/empty waveform - skipping.")
            return False

        # ── 2. Run YAMNet inference ───────────────────────────────────
        try:
            is_emergency, detected_class, confidence = self.detector.process_audio_chunk(
                waveform, sample_rate
            )
        except Exception as exc:
            print(f"[AudioHandler] Lane {lane_id}: model error - {exc}")
            return False

        # ── 3. Apply confidence gate ──────────────────────────────────
        if not is_emergency or confidence < self.confidence_threshold:
            # Not an emergency, or not confident enough – nothing to do.
            return False

        print(
            f"[AudioHandler] Emergency detected: {detected_class} | "
            f"Confidence: {confidence:.3f}"
        )

        # ── 4. Cooldown check ─────────────────────────────────────────
        if self._is_in_cooldown(lane_id):
            print(f"[AudioHandler] Cooldown active, skipping detection for Lane {lane_id}.")
            return False

        # ── 5. Trigger signal transition ──────────────────────────────
        self._trigger_emergency_sequence(lane_id)
        return True

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _is_valid_waveform(waveform) -> bool:
        """Return True only if the waveform is a non-empty float32 array."""
        if waveform is None:
            return False
        if not isinstance(waveform, np.ndarray):
            return False
        if waveform.ndim != 1 or waveform.size == 0:
            return False
        return True

    def _is_in_cooldown(self, lane_id: int) -> bool:
        """Return True if lane_id received an override within the cooldown window."""
        last = self._last_trigger_time.get(lane_id)
        if last is None:
            return False  # Never triggered for this lane
        return (time.monotonic() - last) < self.cooldown_seconds

    def _trigger_emergency_sequence(self, lane_id: int) -> None:
        """
        Execute the density-aware signal override sequence:
          1. Record trigger timestamp (start cooldown immediately).
          2. Log the event.
          3. If the lane is congested, call gradually_clear_lane() first.
          4. Grant green signal.
        """
        # Record timestamp before doing the network/IO calls so that
        # any async callers entering during the sequence are still blocked.
        self._last_trigger_time[lane_id] = time.monotonic()

        print(f"[AudioHandler] Triggering priority for lane: {lane_id}")

        # Check current density and escalate appropriately.
        density = get_traffic_density(lane_id)
        print(f"  [Signal]  Lane {lane_id} density = {density} vehicles.")

        if density >= CONGESTION_THRESHOLD:
            # Lane is congested – phase it out gradually before going green.
            gradually_clear_lane(lane_id)

        give_green_signal(lane_id)


# ─────────────────────────────────────────────────
# Simulation Loop
# ─────────────────────────────────────────────────

def run_simulation(
    handler: EmergencyAudioHandler,
    num_lanes: int = 4,
    iterations: int = 20,
    interval_seconds: float = 3.0,
) -> None:
    """
    Simulates continuous audio polling across all lanes.

    Each iteration picks a random lane and feeds it a synthetic waveform
    (white-noise placeholder).  In a real deployment each lane would map
    to its own microphone stream.

    Parameters
    ----------
    handler          : Initialised EmergencyAudioHandler instance.
    num_lanes        : Number of lanes to cycle over.
    iterations       : How many polling cycles to run before stopping.
    interval_seconds : Seconds to wait between polling cycles.
    """
    SAMPLE_RATE = 16_000        # YAMNet requires 16 kHz
    CHUNK_DURATION = 1.0        # seconds of audio per chunk
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    print("\n" + "=" * 60)
    print("  UrbanFlow  -  Emergency Audio Simulation")
    print("=" * 60 + "\n")

    for iteration in range(1, iterations + 1):
        lane_id = np.random.randint(1, num_lanes + 1)

        # ── Generate a synthetic waveform (white noise placeholder) ──
        # In production: capture from real microphone / audio buffer.
        waveform = np.random.uniform(-0.1, 0.1, chunk_samples).astype(np.float32)

        print(f"[Sim] Iteration {iteration:02d}/{iterations} | Polling Lane {lane_id}...")

        triggered = handler.handle_audio(waveform, SAMPLE_RATE, lane_id)

        if triggered:
            print(f"[Sim] [OK] Emergency override completed for Lane {lane_id}.")
        else:
            print(f"[Sim] [--] No action taken for Lane {lane_id}.")

        print()
        time.sleep(interval_seconds)

    print("=" * 60)
    print("  Simulation complete.")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    handler = EmergencyAudioHandler(
        confidence_threshold=0.25,
        cooldown_seconds=12.0,
    )
    run_simulation(handler, num_lanes=4, iterations=20, interval_seconds=3.0)
