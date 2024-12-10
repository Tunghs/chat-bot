import gradio as gr  # 그라디오 라이브러리를 불러옵니다.
import ollama

# 챗봇에 채팅이 입력되면 이 함수를 호출합니다. 
# message는 유저의 채팅 메시지, history는 채팅 기록, additional_input_info는 additional_inputs안 블록의 정보를 받습니다.
def response(message, history, additional_input_info):
    # additional_input_info의 텍스트를 챗봇의 대답 뒤에 추가합니다.
    response = ollama.chat(model='jmpark333/exaone', messages=[
    {
       'role': 'user',
       'content': message,
    },
    ])
    return response['message']['content']

gr.ChatInterface(
        fn=response,
        textbox=gr.Textbox(placeholder="말걸어주세요..", container=False, scale=7),
        title="RAG 실습 챗봇?",
        description="물어보면 답하는 챗봇임미다.",
        theme="soft",
        examples=[["안뇽"], ["요즘 덥다 ㅠㅠ"], ["점심메뉴 추천바람, 짜장 짬뽕 택 1"]],
        retry_btn="다시보내기 ↩",
        undo_btn="이전 채팅 삭제 ❌",
        clear_btn="전체 채팅 삭제 💫",
        additional_inputs=[
            gr.Textbox("!!!", label="끝말잇기")
        ]
).launch()