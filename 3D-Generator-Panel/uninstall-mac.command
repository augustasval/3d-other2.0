#!/bin/bash
# 3D Generator Panel Uninstaller for macOS

echo "=========================================="
echo "  3D Generator Panel Uninstaller"
echo "=========================================="
echo ""

PANEL_NAME="3D-Generator-Panel"
DEST_DIR="$HOME/Library/Application Support/Adobe/CEP/extensions/$PANEL_NAME"

if [ -d "$DEST_DIR" ]; then
    echo "Removing panel from: $DEST_DIR"
    rm -rf "$DEST_DIR"
    echo "Panel removed successfully."
else
    echo "Panel not found at: $DEST_DIR"
fi

echo ""
echo "Uninstall complete. Restart After Effects."
echo ""
echo "Press any key to close..."
read -n 1
