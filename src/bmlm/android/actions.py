"""Action execution for Android devices via AndroidWorld and Jeeves."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import structlog
from PIL import Image

from bmlm.android.jeeves_client import JeevesClient
from bmlm.tracing import trace_action

log = structlog.get_logger()


class ActionType(Enum):
    TAP = "tap"
    LONG_PRESS = "long_press"
    TYPE = "type"
    SWIPE = "swipe"
    SCROLL = "scroll"
    NAVIGATE_HOME = "navigate_home"
    NAVIGATE_BACK = "navigate_back"
    WAIT = "wait"
    OPEN_APP = "open_app"


@dataclass
class Action:
    """An action to execute on the Android device."""

    action_type: ActionType
    target_index: int | None = None  # Jeeves element index
    text: str | None = None
    direction: str | None = None  # up, down, left, right
    duration_ms: int | None = None  # for long_press


class ActionExecutor:
    """Executes actions on an Android device via AndroidWorld + Jeeves.

    Uses Jeeves for UI element indexing and AndroidWorld for action execution.
    """

    def __init__(self, env: Any, device_serial: str = "emulator-5554"):
        """Initialize with an AndroidWorld environment.

        Args:
            env: AndroidWorld environment instance
            device_serial: ADB device serial
        """
        self.env = env
        self.device_serial = device_serial
        self.jeeves = JeevesClient(device_serial=device_serial)
        self._ui_elements_cache: list[dict] = []

    def get_ui_elements(self) -> list[dict]:
        """Get current UI elements from Jeeves.

        Returns:
            List of UI element dictionaries with index, text, type, etc.
            Indices match the numbered overlays shown on screen.
        """
        try:
            elements = self.jeeves.refresh()
            self._ui_elements_cache = elements
            return elements
        except Exception as e:
            log.warning("failed_to_get_ui_elements", error=str(e))
            return self._ui_elements_cache

    def get_screenshot(self) -> Optional[Image.Image]:
        """Capture a screenshot from the Android device.

        The screenshot will include Jeeves overlays showing numbered elements.

        Returns:
            PIL Image of the current screen, or None if capture failed
        """
        try:
            # AndroidWorld's env has get_state which returns screen pixels
            state = self.env.get_state()
            if hasattr(state, "pixels") and state.pixels is not None:
                # Convert numpy array to PIL Image
                import numpy as np
                pixels = np.array(state.pixels)
                return Image.fromarray(pixels)
            else:
                log.warning("screenshot_no_pixels")
                return None
        except Exception as e:
            log.error("screenshot_failed", error=str(e))
            return None

    def execute(self, action_dict: dict) -> bool:
        """Execute an action on the device.

        Args:
            action_dict: Action dictionary with keys:
                - action: str (tap, type, swipe, etc.)
                - target_index: int | None (Jeeves element index)
                - input_text: str | None (for type action)
                - direction: str | None (for swipe/scroll)

        Returns:
            True if action executed successfully
        """
        action_type = action_dict.get("action", "wait")
        target_index = action_dict.get("target_index")
        text = action_dict.get("input_text")
        direction = action_dict.get("direction")

        with trace_action(
            action_type=action_type,
            target_id=str(target_index) if target_index is not None else None,
            coords=self.jeeves.get_tap_coordinates(target_index) if target_index is not None else None,
        ) as trace_result:
            start_time = time.perf_counter()
            try:
                # Map action to execution method
                if action_type == "tap":
                    success = self._tap_by_index(target_index)
                elif action_type == "long_press":
                    success = self._long_press_by_index(target_index)
                elif action_type == "type":
                    success = self._type_text(text or "")
                elif action_type == "swipe":
                    success = self._swipe(direction or "up", target_index)
                elif action_type == "scroll":
                    success = self._scroll(direction or "down", target_index)
                elif action_type == "navigate_home":
                    success = self._navigate_home()
                elif action_type == "navigate_back":
                    success = self._navigate_back()
                elif action_type == "wait":
                    success = self._wait()
                elif action_type == "open_app":
                    success = self._open_app(text or "")
                else:
                    log.warning("unknown_action_type", action_type=action_type)
                    success = False

                trace_result["success"] = success
                trace_result["duration_ms"] = (time.perf_counter() - start_time) * 1000
                return success

            except Exception as e:
                log.error("action_execution_failed", action=action_type, error=str(e))
                trace_result["success"] = False
                trace_result["error"] = str(e)
                trace_result["duration_ms"] = (time.perf_counter() - start_time) * 1000
                return False

    def _get_coords_by_index(self, index: int | None) -> Optional[tuple[int, int]]:
        """Get tap coordinates for a Jeeves element index.

        Args:
            index: Jeeves element index (matches overlay numbers)

        Returns:
            (x, y) center coordinates, or None if not found
        """
        if index is None:
            return None
        return self.jeeves.get_tap_coordinates(index)

    def _tap_by_index(self, index: int | None) -> bool:
        """Execute a tap action by Jeeves element index.

        Args:
            index: Jeeves element index to tap

        Returns:
            True if tap executed successfully
        """
        coords = self._get_coords_by_index(index)

        if not coords:
            log.warning("tap_no_coords", target_index=index)
            return False

        from android_world.env.json_action import CLICK, JSONAction

        log.info("tap_by_index", index=index, coords=coords)
        action = JSONAction(action_type=CLICK, x=coords[0], y=coords[1])
        self.env.execute_action(action)
        return True

    def _long_press_by_index(self, index: int | None) -> bool:
        """Execute a long press action by Jeeves element index.

        Args:
            index: Jeeves element index to long press

        Returns:
            True if long press executed successfully
        """
        coords = self._get_coords_by_index(index)

        if not coords:
            log.warning("long_press_no_coords", target_index=index)
            return False

        from android_world.env.json_action import LONG_PRESS, JSONAction

        log.info("long_press_by_index", index=index, coords=coords)
        action = JSONAction(action_type=LONG_PRESS, x=coords[0], y=coords[1])
        self.env.execute_action(action)
        return True

    def _type_text(self, text: str) -> bool:
        """Type text into the focused field."""
        from android_world.env.json_action import INPUT_TEXT, JSONAction

        action = JSONAction(action_type=INPUT_TEXT, text=text)
        self.env.execute_action(action)
        return True

    def _swipe(self, direction: str, target_index: int | None = None) -> bool:
        """Execute a swipe action.

        Args:
            direction: Swipe direction (up, down, left, right)
            target_index: Optional Jeeves element index to swipe on.
                         If provided, swipe is bounded within the element.
        """
        # If target specified, do a bounded swipe within element using ADB swipe
        if target_index is not None:
            swipe_coords = self.jeeves.get_swipe_coordinates(target_index, direction)
            if swipe_coords:
                start, end = swipe_coords
                log.info(
                    "swipe_on_element_bounded",
                    index=target_index,
                    direction=direction,
                    start=start,
                    end=end,
                )
                # Use ADB swipe command directly for precise start/end control
                self._adb_swipe(start[0], start[1], end[0], end[1])
                return True
            else:
                log.info("swipe_bounded_fallback", target_index=target_index, direction=direction,
                         reason="element not found or too small for bounded swipe")

        # Fallback: generic screen swipe
        from android_world.env.json_action import SWIPE, JSONAction
        action = JSONAction(action_type=SWIPE, direction=direction)
        self.env.execute_action(action)
        return True

    def _adb_swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300) -> None:
        """Execute a swipe via ADB command directly.

        Args:
            start_x, start_y: Start coordinates
            end_x, end_y: End coordinates
            duration_ms: Swipe duration in milliseconds
        """
        import subprocess
        cmd = [
            "adb", "-s", self.device_serial,
            "shell", "input", "swipe",
            str(start_x), str(start_y), str(end_x), str(end_y), str(duration_ms)
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
        except Exception as e:
            log.error("adb_swipe_failed", error=str(e))

    def _scroll(self, direction: str, target_index: int | None = None) -> bool:
        """Execute a scroll action.

        Args:
            direction: Scroll direction (up, down, left, right)
            target_index: Optional Jeeves element index to scroll on.
                         If provided, scroll is bounded within the element.
        """
        # If target specified, do a bounded scroll within element using ADB swipe
        if target_index is not None:
            swipe_coords = self.jeeves.get_swipe_coordinates(target_index, direction)
            if swipe_coords:
                start, end = swipe_coords
                log.info(
                    "scroll_on_element_bounded",
                    index=target_index,
                    direction=direction,
                    start=start,
                    end=end,
                )
                # Use ADB swipe command directly for precise start/end control
                self._adb_swipe(start[0], start[1], end[0], end[1])
                return True
            else:
                log.info("scroll_bounded_fallback", target_index=target_index, direction=direction,
                         reason="element not found or too small for bounded scroll")

        # Fallback: generic screen scroll
        from android_world.env.json_action import SCROLL, JSONAction
        action = JSONAction(action_type=SCROLL, direction=direction)
        self.env.execute_action(action)
        return True

    def _navigate_home(self) -> bool:
        """Navigate to home screen."""
        from android_world.env.json_action import NAVIGATE_HOME, JSONAction

        action = JSONAction(action_type=NAVIGATE_HOME)
        self.env.execute_action(action)
        return True

    def _navigate_back(self) -> bool:
        """Navigate back."""
        from android_world.env.json_action import NAVIGATE_BACK, JSONAction

        action = JSONAction(action_type=NAVIGATE_BACK)
        self.env.execute_action(action)
        return True

    def _wait(self, duration_ms: int = 1000) -> bool:
        """Wait for a specified duration."""
        from android_world.env.json_action import WAIT, JSONAction

        action = JSONAction(action_type=WAIT)
        self.env.execute_action(action)
        return True

    def _open_app(self, app_name: str) -> bool:
        """Open an app by name."""
        from android_world.env.json_action import OPEN_APP, JSONAction

        action = JSONAction(action_type=OPEN_APP, app_name=app_name)
        self.env.execute_action(action)
        return True
