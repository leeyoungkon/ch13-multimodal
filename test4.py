import base64
from pathlib import Path

import requests

# PC에서 11434 포트를 Kubernetes Ollama의 11434로 포워딩한 경우입니다.
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "gemma3:12b"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "ingress.png"
MAX_HISTORY_TURNS = 6

GUIDE = """첨부한 시스템 구성도를 분석하여 한국어로 답하세요.
분석 규칙:
- 이미지에 실제 보이는 구성요소 이름, 상자, 선, 화살표, 연결 라벨을 근거로 분석하세요.
- 구성요소의 역할은 표시된 설명과 이름을 참고하세요. 일반 지식으로 추론한 역할은 [추정]으로 표시하세요.
- 연결마다 출발 구성요소, 도착 구성요소, 화살표 방향, 라벨을 확인하세요.
- 모든 연결을 데이터 흐름으로 단정하지 마세요. 데이터 전달, 요청/응답, 제어 관계를 구분하고,
  구분할 근거가 없으면 관계 유형을 [확인 불가]로 표시하세요.
- 화살표 끝이 불분명하거나 선이 겹치면 방향이나 연결 대상을 추측하지 마세요.
- 이미지에 표시되지 않은 구성요소, 프로토콜, 처리 순서, 데이터 형식을 만들어내지 마세요.
- 병렬 분기, 합류, 양방향 연결이 보이면 설명하되, 실제 실행 순서는 그림의 근거가 있을 때만 설명하세요.
- 읽기 어려운 이름과 연결은 [판독 불가] 또는 [확인 불가]로 표시하세요.
- 이전 답변과 후속 질문을 참고하되, 이미지와 충돌하면 이미지의 근거를 우선하세요."""

OVERVIEW_QUESTION = """이 시스템 구성도의 구성요소와 데이터 흐름을 설명해 주세요.
다음 순서로 작성하세요.
1. 전체 구성: 그림에서 확인할 수 있는 시스템 구조 요약
2. 구성요소: 이름, 역할, 근거 또는 추정 여부
3. 연결 관계: '출발 → 도착 / 연결 라벨 / 관계 유형' 형식으로 정리
4. 데이터 흐름: 확인된 연결을 따라 데이터가 이동하는 경로 설명
5. 확인이 필요한 부분: 불명확한 글자, 화살표, 연결, 데이터 의미
질문에 답할 때 그림에 없는 내용을 보충해 확정하지 마세요."""


class DiagramChat:
    def __init__(self, image_path):
        image_path = Path(image_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"구성도 이미지가 없습니다: {image_path}")
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("PNG, JPG, WEBP 이미지 파일을 입력하세요.")
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
        # 사진과 최근 대화를 매 요청에 전달합니다.
        messages = [dict(message) for message in self.history] + [dict(user_message)]
        messages[0]["content"] = GUIDE + "\n\n질문:\n" + messages[0]["content"]
        messages[0]["images"] = [self.image_base64]

        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 4096},
            },
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
        if data.get("done_reason") == "length":
            answer += "\n\n[주의: 출력 길이 제한으로 답변이 중간에 끊겼습니다.]"
        # 실패한 요청은 대화 이력에 기록하지 않습니다.
        self.history.extend([user_message, {"role": "assistant", "content": answer}])
        self.history = self.history[-MAX_HISTORY_TURNS * 2:]
        return answer


def show_answer(chat, question):
    try:
        print("구성도 분석 중...", flush=True)
        print("\n[분석 결과]\n" + chat.ask(question))
    except requests.exceptions.ConnectionError:
        print(f"연결 실패: {OLLAMA_URL} 주소와 포트포워딩을 확인하세요.")
    except requests.exceptions.Timeout:
        print("응답 시간 초과: Ollama 상태를 확인하고 다시 질문하세요.")
    except (ValueError, RuntimeError, requests.exceptions.RequestException) as exc:
        print(f"오류: {exc}")


def main():
    raw_path = input("구성도 이미지 경로 (Enter: 같은 폴더의 architecture.png): ").strip()
    image_path = Path(raw_path.strip('\"').strip("'")) if raw_path else DEFAULT_IMAGE
    try:
        chat = DiagramChat(image_path)
    except (OSError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)

    print(f"\n이미지: {image_path}\n모델: {MODEL}\nOllama: {OLLAMA_URL}")
    show_answer(chat, OVERVIEW_QUESTION)
    print("\n추가 질문 가능 / overview: 전체 분석 / clear: 대화 초기화 / q: 종료")
    while True:
        try:
            question = input("\n질문: ").strip()
            if question.lower() in {"q", "quit", "exit", "종료"}:
                break
            if question.lower() == "clear":
                chat.clear()
                print("대화를 초기화했습니다. 같은 구성도에 계속 질문하세요.")
                continue
            if not question:
                continue
            show_answer(chat, OVERVIEW_QUESTION if question.lower() == "overview" else question)
        except (EOFError, KeyboardInterrupt):
            break
    print("\n종료합니다.")


if __name__ == "__main__":
    main()
