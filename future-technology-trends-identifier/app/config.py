
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )

    api_token: str = Field("", env="API_TOKEN")
    api_url: str = Field("http://localhost:3000", env="API_URL")
    model_name: str = Field("mistral:latest", env="MODEL_NAME")
    chunk_size: int = Field(6000, env="CHUNK_SIZE")
    overlap: int = Field(400, env="OVERLAP")
    max_pages: int = Field(0, env="MAX_PAGES")
    temperature: float = Field(0.0, env="TEMPERATURE")
    seed: int = Field(42, env="SEED")
    parallel_chunks: int = Field(1, env="PARALLEL_CHUNKS")
    skip_pages: str = Field("-1", env="SKIP_PAGES")
    timeout: int = Field(120, env="TIMEOUT")
    priority_keywords: str = Field("Horizon Europe,RIA,IA", env="PRIORITY_KEYWORDS")
    exclude_sections: str = Field("references,appendix", env="EXCLUDE_SECTIONS")

    # Data / files
    datasets_folder: str = Field("datasets", env="DATASETS_FOLDER")
    esco_occupations_csv: str = Field("datasets/all_occupations.csv", env="ESCO_OCCUPATIONS_CSV")
    esco_skills_csv: str = Field("datasets/all_skills.csv", env="ESCO_SKILLS_CSV")

settings = Settings()
