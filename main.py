import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from app.generate_pdf import generate_pdf, generate_proctoring_report
from app.helpers import convert_to_mp4, make_prediction, cleanup_temp_files
from app.llm import generate_personality_report
from app.models import VideoRequest
from trackers.eye_tracker import track_eye
from trackers.head_pose_estimation import detect_head_pose
from trackers.mouth_opening_detector import mouth_opening_detector
from trackers.person_and_phone import detect_phone_and_person


log_format = "{time} | {level}: {message}"
logger.add("logs/app.log", format=log_format, level="INFO")


app = FastAPI()

TEMP_DIR = "tmp/uploaded_files"

load_dotenv()


@app.post("/analyse-proctoring")
def proctoring_analysis(video_url: str = None):
    start_time = time.time()
    logger.info("Starting video analysis")

    res_dict = {}
    res_dict["violated_frames"] = set()
    id = uuid.uuid4()
    video_temp_path = os.path.join(TEMP_DIR, f"{id}.mp4")
    pdf_temp_path = os.path.join(TEMP_DIR, f"{id}.pdf")
    os.makedirs(TEMP_DIR, exist_ok=True)
    convert_to_mp4(video_url, video_temp_path)

    if not video_url:
        raise HTTPException(status_code=400, detail="Video url required")

    with ThreadPoolExecutor() as executor:
        future_phone = executor.submit(detect_phone_and_person, video_temp_path, res_dict)
        future_eye = executor.submit(track_eye, video_temp_path, res_dict)
        future_head = executor.submit(detect_head_pose, video_temp_path, res_dict)
        future_mouth = executor.submit(mouth_opening_detector, video_temp_path, res_dict)

        res_dict = future_phone.result()
        res_dict = future_eye.result()
        res_dict = future_head.result()
        res_dict = future_mouth.result()

        cleanup_temp_files(video_temp_path)

    logger.info(f"Res dict: {res_dict}")

    generate_proctoring_report(pdf_temp_path, res_dict)
    logger.info(f"Proctoring completed in {round(time.time() - start_time, 2)} seconds")

    return FileResponse(pdf_temp_path, media_type="application/pdf", filename="Proctoring.pdf")


@app.post("/predict-personality")
def personality_prediction(request: VideoRequest):
    start = time.time()
    id = uuid.uuid4()
    logger.info("Start of predict-personality api")
    video_url = request.video_url

    """
    Temporary storage location to store extracted/converted video and audio data
    """
    video_temp_path = os.path.join(TEMP_DIR, f"{id}.mp4")
    audio_temp_path = os.path.join(TEMP_DIR, f"{id}.wav")
    audio_feature_path = os.path.join(TEMP_DIR, f"{id}.csv")
    os.makedirs(TEMP_DIR, exist_ok=True)

    convert_to_mp4(video_url, video_temp_path)
    ocean_traits = make_prediction(video_temp_path, audio_temp_path, audio_feature_path)
    cleanup_temp_files(video_temp_path, audio_temp_path, audio_feature_path)

    personality_report = generate_personality_report(ocean_traits)
    while not personality_report:
        personality_report = generate_personality_report(ocean_traits)
        logger.info(f"Finalized personality report: {personality_report}")

    content = [(k.capitalize(), v) for k, v in personality_report.items()]
    file_name = f"{id}.pdf"
    pdf_generated = generate_pdf(content, ocean_traits, file_name)

    logger.info(f"Time taken for predict-personality: {time.time() - start}")
    return FileResponse(
        pdf_generated, media_type="application/pdf", filename="Personality Report.pdf"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=os.getenv("HOST"), port=int(os.getenv("PORT")), reload=True)
