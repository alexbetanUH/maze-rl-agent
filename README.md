# Maze RL Agent

A reinforcement learning agent that navigates a dynamic maze environment using Q-Learning, 
built as a group project for COSC 4353 at the University of Houston.

## Overview

The agent navigates a 64×64 grid maze with rotating fire hazards, teleporters, and 
confusion pads. It learns hazard avoidance purely through experience, without being 
given the rotation schedule or environment layout in advance.

## Features

- **Q-Learning** with a custom BFS-augmented reward heuristic
- **Dynamic hazards** — rotating fire pits, teleporters, and confusion pads
- **Zero-shot transfer** — agent trained on one maze navigates an unseen maze without retraining
- **Custom maze parser** — converts PNG maze images to navigable grids with automatic 
  start/goal detection and RGBA hazard extraction
- **Real-time visualization** — animated matplotlib display showing agent path, 
  hazard states, and exploration progress

## Tech Stack

- Python 3
- NumPy
- Matplotlib
- Pillow (PIL)

## Project Structure
- maze_agent.py       # Core Q-Learning agent and decision loop
- imageToMaze.py      # PNG-to-grid maze parser
- visualization.py    # Real-time animated visualization

## How It Works

The agent's state encodes both position and fire rotation phase, giving it awareness 
of hazard timing without being explicitly programmed with the rotation schedule. 
A BFS heuristic supplements the Q-table to guide exploration toward the goal.

Training runs epsilon-greedy exploration (ε = 0.35 → 0.02) across multiple episodes 
until the agent consistently solves the maze.

## Team

Group 11 — University of Houston, Spring 2026  
5 contributors over 7 weeks (Mar – Apr 2026)
