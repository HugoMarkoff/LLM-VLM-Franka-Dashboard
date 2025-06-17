#!/bin/bash

# Setup script for LLM-VLM-Franka-Dashboard environment variables
echo "🔑 Setting up OpenRouter API Key..."

# Set the environment variable for current session
export OPENROUTER_API_KEY='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'

echo "✅ OPENROUTER_API_KEY set for current session"

# Check if the key is already in shell config files
if grep -q "OPENROUTER_API_KEY" ~/.bashrc 2>/dev/null; then
    echo "ℹ️  API key already found in ~/.bashrc"
elif grep -q "OPENROUTER_API_KEY" ~/.zshrc 2>/dev/null; then
    echo "ℹ️  API key already found in ~/.zshrc"
else
    # Determine which shell config file to use
    if [ -n "$ZSH_VERSION" ]; then
        CONFIG_FILE="$HOME/.zshrc"
        SHELL_NAME="zsh"
    elif [ -n "$BASH_VERSION" ]; then
        CONFIG_FILE="$HOME/.bashrc"
        SHELL_NAME="bash"
    else
        CONFIG_FILE="$HOME/.bashrc"
        SHELL_NAME="bash"
    fi
    
    echo "💾 Would you like to add this to your $CONFIG_FILE for permanent setup? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "" >> "$CONFIG_FILE"
        echo "# OpenRouter API Key for LLM-VLM-Franka-Dashboard" >> "$CONFIG_FILE"
        echo "export OPENROUTER_API_KEY='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'" >> "$CONFIG_FILE"
        echo "✅ API key added to $CONFIG_FILE"
        echo "🔄 Run 'source $CONFIG_FILE' or restart your terminal to make it permanent"
    else
        echo "⚠️  Environment variable set for current session only"
        echo "   To set it permanently, add this line to your shell config:"
        echo "   export OPENROUTER_API_KEY='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'"
    fi
fi

# Test the setup
echo ""
echo "🧪 Testing setup..."
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "✅ OPENROUTER_API_KEY is set: ${OPENROUTER_API_KEY:0:20}..."
    echo "🚀 Ready to run the dashboard!"
else
    echo "❌ OPENROUTER_API_KEY is not set"
    exit 1
fi 