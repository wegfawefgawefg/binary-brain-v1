#!/usr/bin/env python3
"""CLI loop to run the brain on gym-like environments."""
from python.brain import Brain

def main():
    brain = Brain(256)
    for _ in range(10):
        brain.tick()
        print("Reward:", brain.reward)

if __name__ == "__main__":
    main()
