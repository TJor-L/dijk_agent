from typing import Dict, Any, Type, Callable
from pydantic import BaseModel

class Tool:

    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]

    def __init__(self, func: Callable[[BaseModel], Any]):
        self.func = func

    def run(self, **kwargs) -> Dict[str, Any]:
        inp = self.input_model(**kwargs)
        out = self.func(inp)
        if isinstance(out, BaseModel):
            return out.model_dump()
        
        return self.output_model(**out).model_dump()


    def input_json_schema(self) -> Dict[str, Any]:
        return self.input_model.model_json_schema()
