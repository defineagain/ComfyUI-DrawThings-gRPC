from __future__ import annotations

import flatbuffers
import numpy as np

import flatbuffers
import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class SeedMode(object):
  Legacy: int
  TorchCpuCompatible: int
  ScaleAlike: int
  NvidiaGpuCompatible: int

