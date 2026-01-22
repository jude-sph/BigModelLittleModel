#!/usr/bin/env python3
"""
Automatic Jeeves Detection and Installation System
Ensures Jeeves accessibility service is available before benchmark runs
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class JeevesAutoSetup:
    """Automatic Jeeves detection, installation, and configuration"""

    def __init__(self, device_serial: str = None, jeeves_dir: Path = None):
        self.jeeves_package = "com.jeeves"
        self.jeeves_service = "com.jeeves/.JeevesAccessibilityService"
        self.apk_path = None
        self.device_serial = device_serial or ""
        # Default to the jeeves directory relative to this file
        self.jeeves_dir = jeeves_dir or Path(__file__).parent

    def _run_adb(self, command: str) -> str:
        """Execute ADB command"""
        full_cmd = ["adb"]
        if self.device_serial:
            full_cmd.extend(["-s", self.device_serial])
        full_cmd.extend(command.split())

        try:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, timeout=30
            )
            return result.stdout
        except Exception as e:
            logger.error(f"ADB command failed: {e}")
            return ""

    def find_apk_path(self) -> Optional[str]:
        """Find the built Jeeves APK"""
        # Check the actual build output location
        apk_path = self.jeeves_dir / "app/build/outputs/apk/debug/app-debug.apk"
        if apk_path.exists():
            return str(apk_path)

        # Fallback to checking jeeves directory for any APK
        if self.jeeves_dir.exists():
            for apk in self.jeeves_dir.rglob("*.apk"):
                return str(apk)

        return None

    def is_apk_installed(self) -> bool:
        """Check if Jeeves APK is installed"""
        try:
            result = self._run_adb(f"shell pm list packages {self.jeeves_package}")
            is_installed = self.jeeves_package in result
            logger.info(f"📦 Jeeves APK installed: {is_installed}")
            return is_installed
        except Exception as e:
            logger.error(f"Error checking APK installation: {e}")
            return False

    def is_accessibility_enabled(self) -> bool:
        """Check if Jeeves accessibility service is enabled"""
        try:
            result = self._run_adb(
                "shell settings get secure enabled_accessibility_services"
            )
            is_enabled = self.jeeves_service in result
            logger.info(f"🔐 Jeeves accessibility service enabled: {is_enabled}")
            return is_enabled
        except Exception as e:
            logger.error(f"Error checking accessibility service: {e}")
            return False

    def has_overlay_permission(self) -> bool:
        """Check if overlay permission is granted"""
        try:
            result = self._run_adb(
                f"shell appops get {self.jeeves_package} SYSTEM_ALERT_WINDOW"
            )
            has_permission = "allow" in result.lower()
            logger.info(f"🎨 Overlay permission granted: {has_permission}")
            return has_permission
        except Exception as e:
            logger.debug(f"Could not check overlay permission: {e}")
            return False

    def build_apk(self) -> bool:
        """Build Jeeves APK if needed"""
        if not self.jeeves_dir.exists():
            logger.error(f"❌ Jeeves directory not found at {self.jeeves_dir}")
            return False

        logger.info("🔨 Building Jeeves APK...")

        try:
            # Run gradle build in jeeves directory
            result = subprocess.run(
                ["./gradlew", "assembleDebug"],
                cwd=str(self.jeeves_dir),
                timeout=300,  # 5 minutes
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("✅ APK build successful")
                return True
            else:
                logger.error(f"❌ APK build failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error building APK: {e}")
            return False

    def install_apk(self) -> bool:
        """Install Jeeves APK"""
        apk_path = self.find_apk_path()

        if not apk_path:
            logger.info("📦 APK not found, building...")
            if not self.build_apk():
                return False
            apk_path = self.find_apk_path()

        if not apk_path:
            logger.error("❌ Could not locate APK file after build")
            return False

        logger.info(f"📱 Installing APK: {apk_path}")

        try:
            # First, uninstall existing package to avoid version/signature conflicts
            logger.info("🗑️ Uninstalling existing package...")
            uninstall_result = self._run_adb(f"uninstall {self.jeeves_package}")
            logger.debug(f"Uninstall result: {uninstall_result}")

            # Now install fresh (don't use -r flag since we uninstalled)
            result = self._run_adb(f"install {apk_path}")

            # Check for both old and new success indicators
            success_indicators = ["Success", "Performing Streamed Install"]
            failure_indicators = [
                "INSTALL_FAILED",
                "Failure [",
                "adb: failed to install",
            ]

            if any(indicator in result for indicator in failure_indicators):
                logger.error(f"❌ APK installation failed: {result}")
                return False
            elif any(indicator in result for indicator in success_indicators):
                logger.info("✅ APK installation successful")
                return True
            else:
                # If unclear, verify by checking if package is installed
                if self.is_apk_installed():
                    logger.info("✅ APK installation verified")
                    return True
                else:
                    logger.error(f"❌ APK installation unclear result: {result}")
                    return False

        except Exception as e:
            logger.error(f"Error installing APK: {e}")
            return False

    def grant_overlay_permission(self) -> bool:
        """Grant overlay permission to Jeeves"""
        try:
            logger.info("🎨 Granting overlay permission...")
            self._run_adb(
                f"shell appops set {self.jeeves_package} SYSTEM_ALERT_WINDOW allow"
            )
            logger.info("✅ Overlay permission granted")
            return True

        except Exception as e:
            logger.error(f"Error granting overlay permission: {e}")
            return False

    def enable_accessibility_service(self) -> bool:
        """Enable Jeeves accessibility service"""
        try:
            logger.info("🔐 Enabling accessibility service...")

            # Start the main activity first
            self._run_adb(f"shell am start -n {self.jeeves_package}/.MainActivity")

            # Get the current list of enabled services
            current_services = self._run_adb(
                "shell settings get secure enabled_accessibility_services"
            ).strip()

            # Add our service to the list
            if self.jeeves_service not in current_services:
                new_services = (
                    f"{current_services}:{self.jeeves_service}"
                    if current_services
                    else self.jeeves_service
                )
                self._run_adb(
                    f"shell settings put secure enabled_accessibility_services '{new_services}'"
                )

            self._run_adb("shell settings put secure accessibility_enabled 1")

            # Give it time to start
            import time

            time.sleep(3)

            # Verify it worked
            if self.is_accessibility_enabled():
                logger.info("✅ Accessibility service enabled")
                return True
            else:
                logger.warning(
                    "⚠️ Accessibility service command ran but not verified as enabled"
                )
                return False

        except Exception as e:
            logger.error(f"Error enabling accessibility service: {e}")
            return False

    def test_jeeves_functionality(self) -> bool:
        """Test basic Jeeves functionality via ContentProvider"""
        try:
            logger.info("🧪 Testing Jeeves functionality...")

            # Test ping endpoint
            result = self._run_adb(
                f"shell content query --uri content://{self.jeeves_package}/ping"
            )

            if "pong" in result.lower():
                logger.info("✅ Jeeves functionality test passed")
                return True

            # Alternative: test if we can get the phone state
            result = self._run_adb(
                f"shell content query --uri content://{self.jeeves_package}/phone_state"
            )

            if "packageName" in result or "appName" in result:
                logger.info("✅ Jeeves functionality test passed (phone_state)")
                return True

            logger.warning(
                "⚠️ Jeeves functionality test failed - service may need manual activation"
            )
            return False

        except Exception as e:
            logger.error(f"Error testing Jeeves functionality: {e}")
            return False

    def enable_overlay_visibility(self) -> bool:
        """Enable overlay visibility through ContentProvider or broadcast"""
        try:
            logger.info("🎯 Enabling overlay visibility...")

            # First try ContentProvider method (works with our updated ContentProvider)
            result = self._run_adb(
                f"shell content insert --uri content://{self.jeeves_package}/overlay_offset "
                "--bind visible:b:true"
            )

            if "success" in result.lower():
                logger.info("✅ Overlay visibility enabled via ContentProvider")
                return True

            # Fallback to broadcast method (works with our new BroadcastReceiver)
            logger.info("📡 Trying broadcast method...")
            result = self._run_adb(
                f"shell am broadcast -a com.jeeves.TOGGLE_OVERLAY --ez overlay_visible true"
            )

            if "Broadcast completed" in result or "result=0" in result:
                logger.info("✅ Overlay visibility enabled via broadcast")
                return True
            else:
                logger.warning(f"⚠️ Could not enable overlay: {result}")
                return False

        except Exception as e:
            logger.error(f"Error enabling overlay: {e}")
            return False

    def check_device_connected(self) -> bool:
        """Check if device is connected via ADB"""
        try:
            result = self._run_adb("shell echo connected")
            return "connected" in result
        except Exception as e:
            logger.error(f"Device not connected: {e}")
            return False

    def ensure_jeeves_ready(self, force_reinstall: bool = False) -> bool:
        """Complete Jeeves setup and verification"""
        logger.info("🎩 Ensuring Jeeves is ready for benchmark...")

        # Check device connection
        if not self.check_device_connected():
            logger.error("❌ No device connected")
            return False

        # Check if everything is already set up
        if not force_reinstall:
            apk_installed = self.is_apk_installed()
            accessibility_enabled = self.is_accessibility_enabled()
            overlay_granted = self.has_overlay_permission()

            if apk_installed and accessibility_enabled and overlay_granted:
                if self.test_jeeves_functionality():
                    logger.info("🎩 Jeeves already ready!")
                    return True

        # Install APK if needed
        if force_reinstall or not self.is_apk_installed():
            if not self.install_apk():
                return False

        # Grant permissions
        if not self.has_overlay_permission():
            if not self.grant_overlay_permission():
                logger.warning(
                    "⚠️ Overlay permission grant failed - may need manual intervention"
                )

        # Enable accessibility service
        if not self.is_accessibility_enabled():
            if not self.enable_accessibility_service():
                logger.error("❌ Failed to enable accessibility service automatically")
                logger.error("📋 MANUAL STEP REQUIRED:")
                logger.error("   1. Go to Settings > Accessibility")
                logger.error("   2. Find 'Jeeves' and turn it ON")
                logger.error("   3. Allow the permission when prompted")
                return False

        # Enable overlay visibility
        self.enable_overlay_visibility()

        # Final functionality test
        if self.test_jeeves_functionality():
            logger.info("🎩 Jeeves setup complete and verified!")
            return True
        else:
            logger.warning("⚠️ Jeeves installed but functionality test failed")
            logger.warning(
                "   This may require manual accessibility service activation"
            )
            return False


def main():
    """Main setup function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup Jeeves accessibility service")
    parser.add_argument("--force", action="store_true", help="Force reinstallation")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--device", type=str, default="", help="Device serial number")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(message)s", handlers=[logging.StreamHandler()]
    )

    setup = JeevesAutoSetup(args.device)
    success = setup.ensure_jeeves_ready(force_reinstall=args.force)

    if success:
        print("✅ Jeeves ready for benchmarks!")
        sys.exit(0)
    else:
        print("❌ Jeeves setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
