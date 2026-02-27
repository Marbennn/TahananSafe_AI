"""
FastAPI Inference Server for TahananSafe AI
Provides REST API endpoints for incident report analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.analyzer import IncidentAnalyzer
from utils.validators import IncidentValidator


# Initialize FastAPI app
app = FastAPI(
    title="TahananSafe AI API",
    description="AI-powered incident report analysis system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer (lazy loading)
analyzer: Optional[IncidentAnalyzer] = None


def get_analyzer() -> IncidentAnalyzer:
    """Get or initialize analyzer instance"""
    global analyzer
    if analyzer is None:
        model_path = os.getenv("MODEL_PATH", "./models/fine_tuned")
        analyzer = IncidentAnalyzer(model_path=model_path)
        analyzer.load_model()
    return analyzer


# Request/Response Models
class IncidentReportRequest(BaseModel):
    """Request model for incident report analysis"""
    incident_description: str = Field(
        ...,
        description="Detailed description of the incident",
        min_length=10,
        max_length=5000
    )
    witness_name: Optional[str] = Field(None, description="Name of witness (optional)")
    witness_relationship: Optional[str] = Field(None, description="Relationship to victim")
    photo_urls: Optional[list[str]] = Field(default=[], description="URLs of uploaded photos (max 3)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "incident_description": "My neighbor was shouting and hitting their spouse. I heard loud noises and crying. The victim appeared injured.",
                "witness_name": "Juan Dela Cruz",
                "witness_relationship": "Neighbor",
                "photo_urls": []
            }
        }


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    incident_type: str = Field(..., description="Primary type of abuse detected")
    incident_types: list[str] = Field(..., description="All detected abuse types (multi-label)")
    language: str = Field(..., description="Detected language")
    risk_level: str = Field(..., description="Risk level classification")
    risk_percentage: float = Field(..., description="Risk percentage (0-100)")
    priority_level: str = Field(..., description="Priority level")
    children_involved: bool = Field(..., description="Whether children are involved")
    weapon_mentioned: bool = Field(..., description="Whether weapon is mentioned")
    confidence_score: float = Field(..., description="AI confidence score (0-100)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "incident_type": "Physical Abuse",
                "incident_types": ["Physical Abuse", "Psychological Abuse"],
                "language": "English",
                "risk_level": "High",
                "risk_percentage": 75.5,
                "priority_level": "Second Priority (P2)",
                "children_involved": False,
                "weapon_mentioned": False,
                "confidence_score": 85.0
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    message: str


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "TahananSafe AI API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        analyzer_instance = get_analyzer()
        model_loaded = analyzer_instance.model is not None
        
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            message="API is operational" + (" (Model loaded)" if model_loaded else " (Using rule-based analysis)")
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            model_loaded=False,
            message=f"Error: {str(e)}"
        )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_incident(request: IncidentReportRequest):
    """
    Analyze an incident report
    
    This endpoint analyzes the incident description and returns:
    - Incident type classification
    - Language detection
    - Risk assessment (level and percentage)
    - Priority level
    - Children involvement detection
    - Weapon mention detection
    - AI confidence score
    """
    try:
        # Validate input
        validator = IncidentValidator()
        valid, error = validator.validate_incident_description(request.incident_description)
        if not valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Validate photo URLs
        if request.photo_urls and len(request.photo_urls) > 3:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 3 photos allowed"
            )
        
        # Get analyzer and perform analysis
        analyzer_instance = get_analyzer()
        analysis_result = analyzer_instance.analyze(request.incident_description)
        
        # Return response
        return AnalysisResponse(**analysis_result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    try:
        analyzer_instance = get_analyzer()
        
        info = {
            "base_model": analyzer_instance.base_model_name,
            "model_path": analyzer_instance.model_path,
            "model_loaded": analyzer_instance.model is not None,
            "device": analyzer_instance.device,
            "using_fine_tuned": analyzer_instance.model is not None and hasattr(
                analyzer_instance.model, "peft_config"
            )
        }
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
