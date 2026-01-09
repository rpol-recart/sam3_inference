"""Application configuration using Pydantic BaseSettings."""
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Server Configuration
    server_host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    server_port: int = Field(default=8000, description="Port to bind the server to")
    server_workers: int = Field(default=1, description="Number of server workers")
    log_level: str = Field(default="INFO", description="Logging level")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    
    # Model Configuration
    sam3_checkpoint: str = Field(
        default="/app/server/sam_weights/sam3.pt", 
        description="Path to SAM3 model checkpoint"
    )
    sam3_bpe_path: str = Field(
        default="/app/server/sam_weights/bpe_simple_vocab_16e6.txt.gz", 
        description="Path to BPE tokenizer file"
    )
    
    # Image Model Configuration
    image_model_device: str = Field(default="cuda:0", description="Device for image model")
    image_model_compile: bool = Field(default=False, description="Enable torch.compile optimization")
    image_model_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence threshold for image segmentation"
    )
    image_model_resolution: int = Field(default=1008, description="Input image resolution")
    image_model_enabled: bool = Field(default=True, description="Enable image model")
    
    # Video Model Configuration
    video_model_enabled: bool = Field(default=True, description="Enable video model")
    video_model_required: bool = Field(
        default=False, description="Whether video model is required for startup"
    )
    video_gpu_list: List[int] = Field(default=[0, 1, 2, 3], description="List of GPU IDs for video processing")
    video_model_compile: bool = Field(default=False, description="Enable torch.compile for video model")
    video_model_temporal_disambiguation: bool = Field(
        default=True, description="Enable temporal disambiguation for video"
    )
    
    # Session Management
    max_concurrent_sessions: int = Field(default=10, description="Maximum concurrent video sessions")
    session_timeout_seconds: int = Field(default=3600, description="Session timeout in seconds")
    session_cleanup_interval_seconds: int = Field(default=300, description="Session cleanup interval")
    
    # Cache Configuration
    enable_feature_cache: bool = Field(default=True, description="Enable feature caching")
    feature_cache_ttl_seconds: int = Field(default=600, description="Feature cache TTL in seconds")
    max_cache_size_mb: int = Field(default=4096, description="Maximum cache size in MB")
    
    # API Security
    require_api_key: bool = Field(default=False, description="Require API key for requests")
    api_keys: List[str] = Field(default=["your-api-key-1", "your-api-key-2"], description="List of valid API keys")
    cors_origins_list: List[str] = Field(default=["*"], description="List of allowed CORS origins")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(default=100, description="Max requests per minute per client")
    
    # Storage
    upload_dir: str = Field(default="/tmp/sam3_uploads", description="Directory for uploaded files")
    output_dir: str = Field(default="/tmp/sam3_outputs", description="Directory for output files")
    max_upload_size_mb: int = Field(default=100, description="Maximum upload size in MB")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics endpoint")
    metrics_port: int = Field(default=9090, description="Port for metrics server")
    
    class Config:
        """Configuration class."""
        env_file = ".env"
        case_sensitive = True
        
    def ensure_directories(self):
        """Ensure required directories exist."""
        for dir_path in [self.upload_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)


# Create global settings instance
settings = Settings()
