"""Research-only utilities for model architecture and data flow observation.

This package must not be imported by production code under mvp_app/ or services/.
It exists to instrument models externally via PyTorch forward hooks and to
serialize observed tensors/metadata in a form consumed by observation reports.
"""
