from __future__ import annotations

import flatbuffers
import numpy as np

import flatbuffers
import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class ControlInputType(object):
  Unspecified: int
  Custom: int
  Depth: int
  Canny: int
  Scribble: int
  Pose: int
  Normalbae: int
  Color: int
  Lineart: int
  Softedge: int
  Seg: int
  Inpaint: int
  Ip2p: int
  Shuffle: int
  Mlsd: int
  Tile: int
  Blur: int
  Lowquality: int
  Gray: int

