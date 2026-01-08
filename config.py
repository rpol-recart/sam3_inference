"""Configuration management for SAM3 Inference Server."""
import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    server_host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")
    server_workers: int = Field(default=1, alias="SERVER_WORKERS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    reload: bool = Field(default=False, alias="RELOAD")

    # Model Configuration
    sam3_checkpoint: str = Field(default="facebook/sam3", alias="SAM3_CHECKPOINT")
    sam3_bpe_path: str = Field(
        default="../../sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        alias="SAM3_BPE_PATH",
    )

    # Image Model
    image_model_device: str = Field(default="cuda:0", alias="IMAGE_MODEL_DEVICE")
    image_model_compile: bool = Field(default=False, alias="IMAGE_MODEL_COMPILE")
    image_model_confidence_threshold: float = Field(
        default=0.5, alias="IMAGE_MODEL_CONFIDENCE_THRESHOLD"
    )
    image_model_resolution: int = Field(default=1008, alias="IMAGE_MODEL_RESOLUTION")
    image_model_enabled: bool = Field(default=True, alias="IMAGE_MODEL_ENABLED")

    # Video Model
    video_model_enabled: bool = Field(default=True, alias="VIDEO_MODEL_ENABLED")
    video_model_required: bool = Field(default=False, alias="VIDEO_MODEL_REQUIRED")
    video_model_gpus: str = Field(default="0,1,2,3", alias="VIDEO_MODEL_GPUS")
    video_model_compile: bool = Field(default=False, alias="VIDEO_MODEL_COMPILE")
    video_model_temporal_disambiguation: bool = Field(
        default=True, alias="VIDEO_MODEL_TEMPORAL_DISAMBIGUATION"
    )

    # Session Management
    max_concurrent_sessions: int = Field(
        default=10, alias="MAX_CONCURRENT_SESSIONS"
    )
    session_timeout_seconds: int = Field(
        default=3600, alias="SESSION_TIMEOUT_SECONDS"
    )
    session_cleanup_interval_seconds: int = Field(
        default=300, alias="SESSION_CLEANUP_INTERVAL_SECONDS"
    )

    # Cache Configuration
    enable_feature_cache: bool = Field(default=True, alias="ENABLE_FEATURE_CACHE")
    feature_cache_ttl_seconds: int = Field(
        default=600, alias="FEATURE_CACHE_TTL_SECONDS"
    )
    max_cache_size_mb: int = Field(default=4096, alias="MAX_CACHE_SIZE_MB")

    # API Security
    require_api_key: bool = Field(default=False, alias="REQUIRE_API_KEY")
    api_keys: str = Field(default="", alias="API_KEYS")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(
        default=100, alias="RATE_LIMIT_REQUESTS_PER_MINUTE"
    )

    # Storage
    upload_dir: str = Field(default="/tmp/sam3_uploads", alias="UPLOAD_DIR")
    output_dir: str = Field(default="/tmp/sam3_outputs", alias="OUTPUT_DIR")
    max_upload_size_mb: int = Field(default=100, alias="MAX_UPLOAD_SIZE_MB")

    # Monitoring
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")

    @property
    def video_gpu_list(self) -> List[int]:
        """Parse VIDEO_MODEL_GPUS into list of integers."""
        if not self.video_model_gpus:
            return [0]
        return [int(gpu.strip()) for gpu in self.video_model_gpus.split(",")]

    @property
    def api_key_list(self) -> List[str]:
        """Parse API_KEYS into list of strings."""
        if not self.api_keys:
            return []
        return [key.strip() for key in self.api_keys.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS into list of strings."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
