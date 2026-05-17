from pydantic import BaseModel
from typing import List, Dict, Any

class PipelineSpec(BaseModel):
    name: str
    input_types: List[str]
    entropy_estimator: str
    output_signature: Dict[str, str]