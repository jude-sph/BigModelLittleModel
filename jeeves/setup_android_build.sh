#!/bin/bash
set -e

echo "Installing Java 17..."
sudo apt update
sudo apt install -y openjdk-17-jdk

echo "Setting up Android SDK..."
SDK_PATH="/usr/lib/android-sdk"

# Create local.properties if it doesn't exist
if [ ! -f local.properties ]; then
    echo "sdk.dir=$SDK_PATH" > local.properties
fi

# Install Android command-line tools if not present
if [ ! -d "$SDK_PATH/cmdline-tools" ]; then
    echo "Installing Android command-line tools..."
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
    unzip -q commandlinetools-linux-11076708_latest.zip
    yes | sudo cmdline-tools/bin/sdkmanager --sdk_root="$SDK_PATH" --licenses
    sudo cmdline-tools/bin/sdkmanager --sdk_root="$SDK_PATH" "cmdline-tools;latest" "build-tools;35.0.0" "platforms;android-34"
    rm -rf commandlinetools-linux-11076708_latest.zip cmdline-tools
else
    echo "Accepting licenses and installing components..."
    yes | sudo "$SDK_PATH/cmdline-tools/latest/bin/sdkmanager" --licenses
    sudo "$SDK_PATH/cmdline-tools/latest/bin/sdkmanager" "build-tools;35.0.0" "platforms;android-34"
fi

echo "Setup complete! You can now run: ./gradlew assembleDebug"