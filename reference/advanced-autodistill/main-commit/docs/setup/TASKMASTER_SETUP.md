# TaskMaster Setup for WSL2 in Cursor

This guide will help you set up TaskMaster to work properly with WSL2 and Cursor.

## Prerequisites

- Windows 11 with WSL2 installed
- Ubuntu distro in WSL2
- Cursor IDE
- NodeJS and npm

## Setup Steps

### 1. Install NVM and Node.js 16+ in WSL2

Run the following commands in your WSL2 terminal:

```bash
# Install NVM
export NVM_DIR="$HOME/.nvm"
mkdir -p $NVM_DIR
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh > $NVM_DIR/install.sh
bash $NVM_DIR/install.sh

# Load NVM in current shell
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install Node.js 16
nvm install 16
nvm alias default 16
```

### 2. Install TaskMaster in WSL2

With Node.js 16+ active, install TaskMaster:

```bash
npm install -g task-master-ai
```

### 3. Create a TaskMaster Start Script

Create `start_taskmaster.sh` in your project root:

```bash
#!/bin/bash

# Load NVM environment
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  

# Use Node.js 16
nvm use 16

# Set API keys (replace with your actual keys)
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY_HERE"
export PERPLEXITY_API_KEY="YOUR_PERPLEXITY_API_KEY_HERE"
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export GOOGLE_API_KEY="YOUR_GOOGLE_KEY_HERE"
export MISTRAL_API_KEY="YOUR_MISTRAL_KEY_HERE"
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY_HERE"
export XAI_API_KEY="YOUR_XAI_KEY_HERE"
export AZURE_OPENAI_API_KEY="YOUR_AZURE_KEY_HERE"

# Set model configuration
export model="gemini-2.5-pro-exp-05-05"
export MAX_TOKENS="64000"
export TEMPERATURE="0.2"
export DEFAULT_SUBTASKS="5"
export DEFAULT_PRIORITY="medium"
export DEFAULT_MODEL="gemini-2.5-pro-exp-05-05"

# Run Task Master
npx task-master-ai "$@"
```

Make the script executable:

```bash
chmod +x start_taskmaster.sh
```

### 4. Configure MCP for Cursor

Create a file named `taskmaster-mcp.json` in the Windows environment (typically in your user directory):

```json
{
  "mcpServers": {
    "taskmaster-ai": {
      "command": "wsl",
      "args": ["-d", "Ubuntu-22.04", "bash", "-c", "cd /home/ml/project-agi && ./start_taskmaster.sh"],
      "projectRoot": "/home/ml/project-agi",
      "env": {
        "ANTHROPIC_API_KEY": "YOUR_ANTHROPIC_API_KEY_HERE",
        "PERPLEXITY_API_KEY": "PERPLEXITY_API_KEY",
        "OPENAI_API_KEY": "YOUR_OPENAI_KEY_HERE",
        "GOOGLE_API_KEY": "AIzaSyAP-CLweel23OirPckPR7bNgf0nGSowabY",
        "MISTRAL_API_KEY": "YOUR_MISTRAL_KEY_HERE",
        "OPENROUTER_API_KEY": "YOUR_OPENROUTER_KEY_HERE",
        "XAI_API_KEY": "YOUR_XAI_KEY_HERE",
        "AZURE_OPENAI_API_KEY": "YOUR_AZURE_KEY_HERE",
        "model": "gemini-2.5-pro-exp-03-25",
        "MAX_TOKENS": "64000",
        "TEMPERATURE": "0.2",
        "DEFAULT_SUBTASKS": "5",
        "DEFAULT_PRIORITY": "medium",
        "DEFAULT_MODEL": "gemini-2.5-pro-exp-03-25"
      }
    }
  }
}
```

Add this file to your Cursor settings (File > Preferences > Settings), search for "MCP config" and add the path to your `taskmaster-mcp.json` file.

### 5. Start TaskMaster

You can now start TaskMaster from the command palette in Cursor (Ctrl+Shift+P) by selecting "TaskMaster: Ask AI". This will start the TaskMaster server in your WSL2 environment through the MCP configuration.

## Troubleshooting

If you encounter any issues:

1. Make sure your WSL2 distribution name is correct in the MCP configuration file. You can check with `wsl -l -v` in a Windows command prompt.

2. If paths are not correct, make sure to use absolute WSL paths in your configuration.

3. Ensure Node.js 16+ is active when running TaskMaster. You can check with `node -v`.

4. If you see "nullish coalescing operator" errors, it means your Node.js version is too old. Make sure nvm is properly loading Node.js 16.

5. For issues with the MCP server, check the Cursor developer console (Help > Toggle Developer Tools) for error messages. 