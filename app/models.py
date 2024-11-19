from pydantic import BaseModel

class VideoRequest(BaseModel):
    # id: str
    video_url: str
