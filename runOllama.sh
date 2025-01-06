#!/bin/bash
# Installation script for required Python packages

# Update pip to the latest version
cd C:\\Users\\dpazi\\AppData\\Local\\Programs\\Ollama

./ollama pull llama3
./ollama list
./ollama run llama3:latest
./ollama serve