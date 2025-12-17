"""
B-Decide AI - Streamlit Cloud entrypoint

Streamlit Community Cloud typically expects an `app.py` (or a selected file).
This file boots the Streamlit UI and can be used as the public deployment entry.

Notes:
- This app runs the Streamlit UI only. If you want the FastAPI backend publicly,
  deploy `backend/main_enhanced.py` separately (Render/Railway/Fly.io/etc.).
"""

from __future__ import annotations

import os

import streamlit as st


def main() -> None:
    """
    Entrypoint for Streamlit.

    Use env var `BDECIDE_DASHBOARD=basic` to run the basic dashboard instead of enhanced.
    """

    mode = os.getenv("BDECIDE_DASHBOARD", "enhanced").strip().lower()

    if mode == "basic":
        from frontend.dashboard import ChurnDashboard

        app = ChurnDashboard()
        app.run()
        return

    # Default: enhanced dashboard
    from frontend.dashboard_enhanced import EnhancedChurnDashboard

    app = EnhancedChurnDashboard()
    app.run()


# Streamlit executes the script top-to-bottom on each run.
main()


