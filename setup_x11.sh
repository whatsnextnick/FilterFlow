#!/bin/bash
# Setup X11 forwarding for WSL2

echo "Setting up X11 forwarding for WSL2..."

# Install X11 apps if not present
if ! command -v xclock &> /dev/null; then
    echo "Installing X11 apps..."
    sudo apt-get update
    sudo apt-get install -y x11-apps
fi

# Set DISPLAY variable for WSL2
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# Add to bashrc if not already there
if ! grep -q "DISPLAY=" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# X11 forwarding for WSL2" >> ~/.bashrc
    echo "export DISPLAY=\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}'):0" >> ~/.bashrc
    echo "Added DISPLAY to ~/.bashrc"
fi

echo ""
echo "X11 setup complete!"
echo ""
echo "Next steps:"
echo "1. Install VcXsrv or Xming on Windows"
echo "2. Launch with 'Disable access control' checked"
echo "3. Run: source ~/.bashrc"
echo "4. Test: xclock"
echo "5. Run app: python3 photo_filter_app.py test_image.jpg"
