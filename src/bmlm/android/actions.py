"""Action execution for Android devices via AndroidWorld."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

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
    target_id: str | None = None
    coords: tuple[int, int] | None = None
    text: str | None = None
    direction: str | None = None  # up, down, left, right
    duration_ms: int | None = None  # for long_press


class ActionExecutor:
    """Executes actions on an Android device via AndroidWorld environment."""

    def __init__(self, env: Any):
        """Initialize with an AndroidWorld environment.

        Args:
            env: AndroidWorld environment instance
        """
        self.env = env
        self._ui_elements_cache: list[dict] = []

    def get_ui_elements(self) -> list[dict]:
        """Get current UI elements from the environment.

        Returns:
            List of UI element dictionaries with id, text, type, bounds
        """
        try:
            state = self.env.get_state(wait_to_stabilize=True)

            elements = []
            if hasattr(state, "ui_elements"):
                for elem in state.ui_elements:
                    elements.append({
                        "id": getattr(elem, "resource_id", None) or getattr(elem, "id", None),
                        "text": getattr(elem, "text", ""),
                        "content_desc": getattr(elem, "content_description", ""),
                        "type": getattr(elem, "class_name", "unknown"),
                        "bounds": getattr(elem, "bounds", None),
                        "clickable": getattr(elem, "clickable", False),
                        "enabled": getattr(elem, "enabled", True),
                    })

            self._ui_elements_cache = elements
            return elements
        except Exception as e:
            log.warning("failed_to_get_ui_elements", error=str(e))
            return self._ui_elements_cache

    def execute(self, action_dict: dict) -> bool:
        """Execute an action on the device.

        Args:
            action_dict: Action dictionary with keys:
                - action: str (tap, type, swipe, etc.)
                - target_id: str | None
                - target_coords: tuple[int, int] | None
                - input_text: str | None
                - direction: str | None

        Returns:
            True if action executed successfully
        """
        action_type = action_dict.get("action", "wait")
        target_id = action_dict.get("target_id")
        coords = action_dict.get("target_coords")
        text = action_dict.get("input_text")
        direction = action_dict.get("direction")

        try:
            # Map action to AndroidWorld action
            if action_type == "tap":
                return self._tap(target_id, coords)
            elif action_type == "long_press":
                return self._long_press(target_id, coords)
            elif action_type == "type":
                return self._type_text(text or "")
            elif action_type == "swipe":
                return self._swipe(direction or "up")
            elif action_type == "scroll":
                return self._scroll(direction or "down")
            elif action_type == "navigate_home":
                return self._navigate_home()
            elif action_type == "navigate_back":
                return self._navigate_back()
            elif action_type == "wait":
                return self._wait()
            elif action_type == "open_app":
                return self._open_app(text or "")
            else:
                log.warning("unknown_action_type", action_type=action_type)
                return False

        except Exception as e:
            log.error("action_execution_failed", action=action_type, error=str(e))
            return False

    def _find_element_coords(self, target_id: str | None) -> tuple[int, int] | None:
        """Find center coordinates of an element by ID."""
        if not target_id:
            return None

        for elem in self._ui_elements_cache:
            if elem.get("id") == target_id or target_id in str(elem.get("text", "")):
                bounds = elem.get("bounds")
                if bounds:
                    # Bounds format varies, handle common cases
                    if isinstance(bounds, dict):
                        x = (bounds.get("left", 0) + bounds.get("right", 0)) // 2
                        y = (bounds.get("top", 0) + bounds.get("bottom", 0)) // 2
                        return (x, y)
                    elif isinstance(bounds, (list, tuple)) and len(bounds) == 4:
                        x = (bounds[0] + bounds[2]) // 2
                        y = (bounds[1] + bounds[3]) // 2
                        return (x, y)
        return None

    def _tap(self, target_id: str | None, coords: tuple[int, int] | None) -> bool:
        """Execute a tap action."""
        if not coords:
            coords = self._find_element_coords(target_id)

        if not coords:
            log.warning("tap_no_coords", target_id=target_id)
            return False

        # Use AndroidWorld's tap action
        from android_world.env import interface

        action = interface.Tap(x=coords[0], y=coords[1])
        self.env.execute_action(action)
        return True

    def _long_press(self, target_id: str | None, coords: tuple[int, int] | None) -> bool:
        """Execute a long press action."""
        if not coords:
            coords = self._find_element_coords(target_id)

        if not coords:
            log.warning("long_press_no_coords", target_id=target_id)
            return False

        from android_world.env import interface

        action = interface.LongPress(x=coords[0], y=coords[1])
        self.env.execute_action(action)
        return True

    def _type_text(self, text: str) -> bool:
        """Type text into the focused field."""
        from android_world.env import interface

        action = interface.Type(text=text)
        self.env.execute_action(action)
        return True

    def _swipe(self, direction: str) -> bool:
        """Execute a swipe action."""
        from android_world.env import interface

        # Define swipe vectors based on direction
        # Assuming 1080x2400 screen
        center_x, center_y = 540, 1200
        distance = 500

        direction_map = {
            "up": (center_x, center_y, center_x, center_y - distance),
            "down": (center_x, center_y, center_x, center_y + distance),
            "left": (center_x, center_y, center_x - distance, center_y),
            "right": (center_x, center_y, center_x + distance, center_y),
        }

        if direction not in direction_map:
            direction = "up"

        start_x, start_y, end_x, end_y = direction_map[direction]
        action = interface.Swipe(
            start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y
        )
        self.env.execute_action(action)
        return True

    def _scroll(self, direction: str) -> bool:
        """Execute a scroll action (similar to swipe but typically slower)."""
        from android_world.env import interface

        # Scroll is often just a slower/shorter swipe
        scroll_direction = "down" if direction in ["down", "up"] else direction
        action = interface.Scroll(direction=scroll_direction)
        self.env.execute_action(action)
        return True

    def _navigate_home(self) -> bool:
        """Navigate to home screen."""
        from android_world.env import interface

        action = interface.NavigateHome()
        self.env.execute_action(action)
        return True

    def _navigate_back(self) -> bool:
        """Navigate back."""
        from android_world.env import interface

        action = interface.NavigateBack()
        self.env.execute_action(action)
        return True

    def _wait(self, duration_ms: int = 1000) -> bool:
        """Wait for a specified duration."""
        import time

        time.sleep(duration_ms / 1000)
        return True

    def _open_app(self, app_name: str) -> bool:
        """Open an app by name."""
        from android_world.env import interface

        action = interface.OpenApp(app_name=app_name)
        self.env.execute_action(action)
        return True
