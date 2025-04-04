from __future__ import annotations

import flatbuffers
import numpy as np

import flatbuffers
import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class SamplerType(object):
  DPMPP2MKarras: int
  EulerA: int
  DDIM: int
  PLMS: int
  DPMPPSDEKarras: int
  UniPC: int
  LCM: int
  EulerASubstep: int
  DPMPPSDESubstep: int
  TCD: int
  EulerATrailing: int
  DPMPPSDETrailing: int
  DPMPP2MAYS: int
  EulerAAYS: int
  DPMPPSDEAYS: int
  DPMPP2MTrailing: int
  DDIMTrailing: int

