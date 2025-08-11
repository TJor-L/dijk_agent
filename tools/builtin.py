from typing import Optional
from pydantic import BaseModel
from .base import Tool
import datetime as dt

class CalcIn(BaseModel):
    a: float
    b: float

class CalcOut(BaseModel):
    result: float

def _calc_impl(inp: CalcIn) -> CalcOut:
    return CalcOut(result=inp.a + inp.b)

class Calc(Tool):
    name = "calc"
    description = "Compute a + b."
    input_model = CalcIn
    output_model = CalcOut

    def __init__(self):
        super().__init__(func=_calc_impl)

class NowIn(BaseModel):
    pass

class NowOut(BaseModel):
    utc: str

def _now_impl(inp: NowIn) -> NowOut:
    return NowOut(utc=dt.datetime.utcnow().isoformat())

class Now(Tool):
    name = "now"
    description = "Return current UTC ISO time."
    input_model = NowIn
    output_model = NowOut

    def __init__(self):
        super().__init__(func=_now_impl)
