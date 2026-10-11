import base64
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:12b"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "animals.jpg"
MAX_HISTORY_TURNS = 6  # 최근 6회 질문·답변을 유지합니다.
INSTRUCTIONS = (
    "첨부한 같은 사진을 보면서 사용자의 질문에 한국어로 답하세요. "
    "위치는 사진을 보는 사람을 기준으로 왼쪽/오른쪽/위/아래/중앙으로 설명하세요. "
    "특정 물체의 위치를 물으면 주변 물체와의 관계도 설명하세요. "
    "이전 질문과 답변을 참고하되, 실제 사진을 근거로 답하고 "
    "보이지 않거나 불확실한 것은 추측하지 마세요."
)


class ImageChat:
    def __init__(self, image_path):
        image_path = Path(image_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        image_bytes = image_path.read_bytes()
        if not image_bytes:
            raise ValueError("이미지 파일이 비어 있습니다.")
        self.image_base64 = base64.b64encode(image_bytes).decode("ascii")
        self.history = []

    def clear(self):
        self.history.clear()

    def ask(self, question):
        if not question.strip():
            raise ValueError("질문을 입력하세요.")
        user_message = {"role": "user", "content": question}
        # API는 요청 간 대화를 자동 보관하지 않으므로 매번 이력을 보냅니다.
        messages = [dict(message) for message in self.history] + [dict(user_message)]
        # 매 요청에 사진을 한 번 포함합니다. 이전 이미지 메시지를 누적하지 않습니다.
        messages[0]["content"] = INSTRUCTIONS + "\n\n질문: " + messages[0]["content"]
        messages[0]["images"] = [self.image_base64]

        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={"model": MODEL, "messages": messages, "stream": False},
            timeout=(10, 600),
        )
        if not response.ok:
            raise RuntimeError(f"Ollama 오류 ({response.status_code}): {response.text}")
        data = response.json()
        if data.get("error"):
            raise RuntimeError(data["error"])
        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama가 빈 답변을 반환했습니다.")

        # 정상 응답을 받은 질문과 답변만 이력에 추가합니다.
        self.history.extend([user_message, {"role": "assistant", "content": answer}])
        self.history = self.history[-MAX_HISTORY_TURNS * 2:]
        return answer


def main():
    raw_path = input("사진 경로 (Enter: 같은 폴더의 animals.jpg): ").strip()
    image_path = Path(raw_path.strip('\"').strip("'")) if raw_path else DEFAULT_IMAGE
    try:
        chat = ImageChat(image_path)
    except (OSError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)

    print(f"\n사진: {image_path}\n모델: {MODEL}")
    print("같은 사진에 여러 번 질문할 수 있습니다. q: 종료 / clear: 대화 초기화")
    while True:
        try:
            question = input("\n질문: ").strip()
            if question.lower() in {"q", "quit", "exit", "종료"}:
                print("종료합니다.")
                break
            if question.lower() == "clear":
                chat.clear()
                print("대화를 초기화했습니다. 같은 사진에 계속 질문하세요.")
                continue
            if not question:
                continue
            print("사진 분석 중...", flush=True)
            print("\n[답변]\n" + chat.ask(question))
        except requests.exceptions.ConnectionError:
            print(f"연결 실패: {OLLAMA_URL} 주소와 포트포워딩을 확인하세요.")
        except requests.exceptions.Timeout:
            print("응답 시간 초과: Ollama 상태를 확인하고 다시 질문하세요.")
        except (ValueError, RuntimeError, requests.exceptions.RequestException) as exc:
            print(f"오류: {exc}")
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break


if __name__ == "__main__":
    main()
