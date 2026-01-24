"""Jeeves ContentProvider client for UI element retrieval."""

import json
import subprocess
from dataclasses import dataclass
from typing import Optional

import structlog

log = structlog.get_logger()


@dataclass
class JeevesElement:
    """A UI element from Jeeves with index and bounds."""

    index: int
    text: str
    content_desc: str
    class_name: str
    resource_id: str
    bounds: tuple[int, int, int, int]  # left, top, right, bottom
    clickable: bool
    enabled: bool
    children: list["JeevesElement"]

    @property
    def center(self) -> tuple[int, int]:
        """Calculate center coordinates of this element."""
        left, top, right, bottom = self.bounds
        return ((left + right) // 2, (top + bottom) // 2)

    def get_swipe_coords(
        self,
        direction: str,
        margin: int = 20,
        screen_width: int = 1080,
        screen_height: int = 2400,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get start and end coordinates for a swipe within this element.

        Args:
            direction: Swipe direction (up, down, left, right)
            margin: Pixels to inset from element edges
            screen_width: Screen width for edge safety calculations
            screen_height: Screen height for edge safety calculations

        Returns:
            ((start_x, start_y), (end_x, end_y)) coordinates for the swipe
        """
        left, top, right, bottom = self.bounds
        center_x, center_y = self.center

        # Inset bounds by margin
        safe_left = left + margin
        safe_right = right - margin
        safe_top = top + margin
        safe_bottom = bottom - margin

        # Edge safety zones - swipe START must not be in these zones
        # (Android gesture navigation triggers back when swiping from edges)
        EDGE_SAFE_MIN = 100  # Minimum distance from left/top edges
        edge_safe_max_x = screen_width - EDGE_SAFE_MIN   # Maximum x (distance from right)
        edge_safe_max_y = screen_height - EDGE_SAFE_MIN  # Maximum y (distance from bottom)

        if direction == "right":
            # Swipe from left to right - start not too close to LEFT edge
            start_x = max(safe_left, EDGE_SAFE_MIN)
            return ((start_x, center_y), (safe_right, center_y))
        elif direction == "left":
            # Swipe from right to left - start not too close to RIGHT edge
            start_x = min(safe_right, edge_safe_max_x)
            return ((start_x, center_y), (safe_left, center_y))
        elif direction == "down":
            # Swipe from top to bottom - start not too close to TOP edge
            start_y = max(safe_top, EDGE_SAFE_MIN)
            return ((center_x, start_y), (center_x, safe_bottom))
        elif direction == "up":
            # Swipe from bottom to top - start not too close to BOTTOM edge
            start_y = min(safe_bottom, edge_safe_max_y)
            return ((center_x, start_y), (center_x, safe_top))
        else:
            # Default: center to center (no movement)
            return ((center_x, center_y), (center_x, center_y))

    def to_dict(self) -> dict:
        """Convert to dictionary for model input."""
        return {
            "index": self.index,
            "text": self.text,
            "content_desc": self.content_desc,
            "type": self.class_name.split(".")[-1] if self.class_name else "Unknown",
            "clickable": self.clickable,
            "enabled": self.enabled,
        }


class JeevesClient:
    """Client for querying Jeeves ContentProvider via ADB."""

    CONTENT_URI = "content://com.jeeves"

    def __init__(self, device_serial: str = "emulator-5554"):
        self.device_serial = device_serial
        self._elements_cache: list[JeevesElement] = []
        self._elements_by_index: dict[int, JeevesElement] = {}

    def _run_adb(self, command: str, timeout: int = 10) -> str:
        """Execute an ADB command and return output."""
        full_cmd = ["adb"]
        if self.device_serial:
            full_cmd.extend(["-s", self.device_serial])
        full_cmd.extend(["shell", command])

        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            log.warning("adb_timeout", command=command)
            return ""
        except Exception as e:
            log.error("adb_error", command=command, error=str(e))
            return ""

    def get_ui_elements(self) -> list[JeevesElement]:
        """Query Jeeves for current UI elements.

        Returns:
            List of JeevesElement with indices matching overlay numbers
        """
        # Query the ContentProvider
        result = self._run_adb(f"content query --uri {self.CONTENT_URI}/a11y_tree")

        if not result or "No result found" in result:
            log.warning("jeeves_no_elements")
            return self._elements_cache

        # Parse the ContentProvider response
        elements = self._parse_content_provider_response(result)

        if elements:
            self._elements_cache = elements
            self._elements_by_index = {e.index: e for e in self._flatten_elements(elements)}
            log.info("jeeves_elements_loaded", count=len(self._elements_by_index))

        return elements

    def _parse_content_provider_response(self, response: str) -> list[JeevesElement]:
        """Parse ContentProvider query response into JeevesElements."""
        elements = []

        # ContentProvider returns rows like: Row: 0 result={"status":"success","message":"[...]"}
        try:
            # Try to find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                # Handle different response formats
                tree = None
                if "a11y_tree" in data:
                    tree = data["a11y_tree"]
                elif "message" in data:
                    # Message might be a JSON string that needs parsing
                    message = data["message"]
                    if isinstance(message, str):
                        try:
                            tree = json.loads(message)
                        except json.JSONDecodeError:
                            pass
                    elif isinstance(message, dict) and "a11y_tree" in message:
                        tree = message["a11y_tree"]
                    elif isinstance(message, list):
                        tree = message

                if tree is None:
                    tree = data if isinstance(data, list) else [data]

                if isinstance(tree, list):
                    elements = self._parse_element_tree(tree)

        except json.JSONDecodeError as e:
            log.warning("jeeves_json_parse_error", error=str(e))
            # Try alternative parsing for simple format
            elements = self._parse_simple_format(response)

        return elements

    def _parse_element_tree(self, tree: list[dict]) -> list[JeevesElement]:
        """Recursively parse element tree from JSON."""
        elements = []

        for item in tree:
            element = self._parse_single_element(item)
            if element:
                elements.append(element)

        return elements

    def _parse_single_element(self, item: dict) -> Optional[JeevesElement]:
        """Parse a single element from JSON dict."""
        try:
            # Parse bounds from string "left,top,right,bottom" or dict
            bounds = item.get("bounds", "0,0,0,0")
            if isinstance(bounds, str):
                parts = bounds.replace(" ", "").split(",")
                if len(parts) == 4:
                    bounds_tuple = tuple(int(p) for p in parts)
                else:
                    bounds_tuple = (0, 0, 0, 0)
            elif isinstance(bounds, (list, tuple)):
                bounds_tuple = tuple(bounds)
            elif isinstance(bounds, dict):
                bounds_tuple = (
                    bounds.get("left", 0),
                    bounds.get("top", 0),
                    bounds.get("right", 0),
                    bounds.get("bottom", 0),
                )
            else:
                bounds_tuple = (0, 0, 0, 0)

            # Parse children recursively
            children_data = item.get("children", [])
            children = self._parse_element_tree(children_data) if children_data else []

            return JeevesElement(
                index=item.get("index", item.get("overlayIndex", -1)),
                text=item.get("text", ""),
                content_desc=item.get("contentDescription", item.get("content_desc", "")),
                class_name=item.get("className", item.get("class_name", "")),
                resource_id=item.get("resourceId", item.get("resource_id", "")),
                bounds=bounds_tuple,
                clickable=item.get("clickable", False),
                enabled=item.get("enabled", True),
                children=children,
            )
        except Exception as e:
            log.warning("jeeves_element_parse_error", error=str(e), item=item)
            return None

    def _parse_simple_format(self, response: str) -> list[JeevesElement]:
        """Fallback parser for simpler response formats."""
        elements = []
        # Try to extract any JSON arrays from the response
        try:
            for line in response.split("\n"):
                if "[" in line and "]" in line:
                    start = line.find("[")
                    end = line.rfind("]") + 1
                    arr = json.loads(line[start:end])
                    elements.extend(self._parse_element_tree(arr))
                    break
        except:
            pass
        return elements

    def _flatten_elements(self, elements: list[JeevesElement]) -> list[JeevesElement]:
        """Flatten hierarchical elements into a flat list."""
        flat = []
        for elem in elements:
            flat.append(elem)
            if elem.children:
                flat.extend(self._flatten_elements(elem.children))
        return flat

    def get_element_by_index(self, index: int) -> Optional[JeevesElement]:
        """Look up an element by its Jeeves index.

        Args:
            index: The Jeeves overlay index

        Returns:
            JeevesElement if found, None otherwise
        """
        return self._elements_by_index.get(index)

    def get_elements_for_model(self) -> list[dict]:
        """Get UI elements formatted for model input.

        Returns:
            List of element dicts with index, text, type, etc.
        """
        elements = []
        for elem in self._elements_by_index.values():
            # Only include elements that are potentially interactive
            if elem.clickable or elem.text or elem.content_desc:
                elements.append(elem.to_dict())

        # Sort by index for consistent ordering
        elements.sort(key=lambda x: x["index"])
        return elements

    def get_tap_coordinates(self, index: int) -> Optional[tuple[int, int]]:
        """Get tap coordinates for an element by index.

        Args:
            index: The Jeeves element index

        Returns:
            (x, y) center coordinates, or None if element not found
        """
        element = self.get_element_by_index(index)
        if element:
            return element.center
        return None

    def get_swipe_coordinates(
        self, index: int, direction: str
    ) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
        """Get swipe start/end coordinates within an element's bounds.

        Args:
            index: The Jeeves element index
            direction: Swipe direction (up, down, left, right)

        Returns:
            ((start_x, start_y), (end_x, end_y)) or None if element not found
        """
        element = self.get_element_by_index(index)
        if element:
            return element.get_swipe_coords(direction)
        return None

    def refresh(self) -> list[dict]:
        """Refresh UI elements and return formatted for model.

        Returns:
            List of element dicts for model input
        """
        self.get_ui_elements()
        return self.get_elements_for_model()
