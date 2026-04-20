#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gradio as gr

print(f"파이썬 인터프리터: {sys.executable}")
print(f"Gradio 버전: {gr.__version__}")
print(f"Gradio 라이브러리 경로: {os.path.dirname(gr.__file__)}")
print("\nGradio Update 함수 확인:")
print(f"gr.update 존재 여부: {hasattr(gr, 'update')}")
print(f"gr.Dropdown.update 존재 여부: {hasattr(gr.Dropdown, 'update')}") 