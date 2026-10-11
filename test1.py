import base64
from pathlib import Path

import requests

# 이미지 파일 읽기
image_bytes = Path("animals.jpg").read_bytes()
image_base64 = base64.b64encode(image_bytes).decode("ascii")

# 이미지와 질문을 Gemma 3에 전달
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "gemma3:12b",
        "messages": [
            {
                "role": "user",
                "content": (
                    "이 사진의 전체 내용을 한국어로 설명해 주세요. "
                    "전체 장면, 주요 인물이나 사물, 배경과 배치를 설명하고, "
                    "명확히 보이지 않는 부분은 추측하지 마세요."
                ),
                "images": [image_base64]
            }
        ],
        "stream": False
    },
    timeout=600
)

response.raise_for_status()
print(response.json()["message"]["content"])