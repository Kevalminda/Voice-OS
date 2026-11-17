"""Backward-compatible entry point for the same manual hardware diagnostic."""

from audio.diagnostics import main

if __name__ == "__main__":
    raise SystemExit(main())
