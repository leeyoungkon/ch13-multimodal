import base64
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:12b"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "주주명부.jpg"
OUTPUT_DIR = BASE_DIR / "output"

PROMPT = """첨부한 문서 이미지에서 문자와 표 구조를 추출하여 텍스트로 변환하세요.
다음 규칙을 따르세요.
1. 요약하거나 번역하지 말고, 보이는 내용을 원문 그대로 옮기세요.
2. 제목, 문단, 목록, 주석, 표를 원래 읽는 순서대로 출력하세요.
3. 표는 [표 1], [표 2]처럼 구분하고 Markdown 표로 출력하세요.
4. 표의 행 순서, 열 순서, 머리글, 숫자, 날짜, 단위, 기호를 보존하세요.
5. 빈 셀은 비워 두고, 읽을 수 없는 글자나 셀은 [판독 불가]로 표시하세요.
6. 이미지에 없는 내용이나 수치를 추측하여 추가하지 마세요.
7. 병합 셀과 다단 머리글은 Markdown에서 완전히 표현할 수 없으므로,
   필요한 경우 대응하는 열에 머리글을 반복하고 표 아래 [구조 참고]에
   원본에서 확인되는 병합 범위를 설명하세요. 범위가 불확실하면 불확실하다고 쓰세요.
8. 머리글이 없는 표는 '열1', '열2' 등의 구분용 머리글을 사용하고,
   [구조 참고]에 원본에는 머리글이 없다고 표시하세요.
9. 표가 없으면 본문만 출력하세요.
10. 서문, 추출 방법 설명, 전체 내용을 감싸는 코드 블록은 출력하지 마세요.
문서 내용과 필요한 [구조 참고]만 출력하세요."""


def extract_document(image_path):
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"문서 이미지가 없습니다: {image_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("JPG, PNG, WEBP 이미지 파일을 입력하세요.")
    image_bytes = image_path.read_bytes()
    if not image_bytes:
        raise ValueError("이미지 파일이 비어 있습니다.")

    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        json={
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": PROMPT,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 8192},
        },
        timeout=(10, 600),
    )
    if not response.ok:
        raise RuntimeError(f"Ollama 오류 ({response.status_code}): {response.text}")
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    text = data.get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError("추출 결과가 비어 있습니다.")
    return text, data.get("done_reason") == "length"


def save_result(image_path, text, truncated):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"{Path(image_path).stem}_extracted_{stamp}.txt"
    if truncated:
        text = "[주의: 출력 길이 제한으로 추출 결과가 중간에 끊겼습니다.]\n\n" + text
    # 실행마다 새 파일을 생성하여 이전 결과를 보존합니다.
    with output_path.open("x", encoding="utf-8") as output:
        output.write(text + "\n")
    return output_path


def main():
    raw_path = input("문서 이미지 경로 (Enter: 같은 폴더의 주주명부.jpg): ").strip()
    image_path = Path(raw_path.strip('\"').strip("'")) if raw_path else DEFAULT_IMAGE
    try:
        print(f"문자와 표 추출 중... 모델: {MODEL}", flush=True)
        text, truncated = extract_document(image_path)
        print("\n[추출 결과]\n" + text)
        if truncated:
            print("\n주의: 출력이 중간에 끊겼습니다. 문서를 나누거나 num_predict를 조정하세요.")
        output_path = save_result(image_path, text, truncated)
        print(f"\nUTF-8 텍스트 저장 완료: {output_path}")
    except requests.exceptions.ConnectionError:
        print(f"연결 실패: {OLLAMA_URL} 주소와 포트포워딩을 확인하세요.")
        raise SystemExit(1)
    except requests.exceptions.Timeout:
        print("응답 시간 초과: Ollama 상태를 확인하세요.")
        raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, requests.exceptions.RequestException) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
