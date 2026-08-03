# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Frozen-teacher probes for I3.

Everything under here lives outside the training and inference paths: probes read frozen teacher
features and fit a small head on geometry targets, and never construct the Qwen backbone, the action
model or the world predictor (`docs/implementation-plan.md` section 9).
"""
