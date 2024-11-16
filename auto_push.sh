#!/bin/bash

# Check the status
git status

# Add all changes
git add .

# Commit changes with a dynamic message (using date as an example)
git commit -m "Auto commit: $(date)"

# Push changes to GitHub (replace 'main' with your branch if necessary)
git push origin main

