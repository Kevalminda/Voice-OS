"""Manual hardware diagnostic entry point; safe for pytest to import."""

from audio.diagnostics import main

if __name__ == "__main__":
    raise SystemExit(main())
