#!/bin/bash

# Script to install NVM and Node.js 16

# Install NVM
echo "Installing NVM..."
export NVM_DIR="$HOME/.nvm"
mkdir -p $NVM_DIR
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh > $NVM_DIR/install.sh
bash $NVM_DIR/install.sh

# Load NVM
echo "Loading NVM..."
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install Node.js 16
echo "Installing Node.js 16..."
nvm install 16

# Set default Node.js version
echo "Setting default Node.js version..."
nvm alias default 16

# Display versions
echo "Node.js version:"
node -v
echo "NPM version:"
npm -v

echo "Installation complete!" 