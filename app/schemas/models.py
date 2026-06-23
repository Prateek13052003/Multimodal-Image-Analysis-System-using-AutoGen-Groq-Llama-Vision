from pydantic import BaseModel, Field
from typing import Optional, List

class AnalysisRequest(BaseModel):
    image_url: Optional[str] = Field(
        None,
        description="S3 URL or base64 image"
    )
    model: str = Field(
        default="llama-4-vision",
        description="Model to use"
    )

class ImageAnalysisResponse(BaseModel):
    id: str
    classification: str
    confidence: float = Field(..., ge=0, le=1)
    objects: List[str]
    colors: List[str]
    scene_description: str
    insights: str
    created_at: str
    status: str = "completed"
    cost_usd: Optional[float] = None
    correlation_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str