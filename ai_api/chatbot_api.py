from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_json_chat_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


app = FastAPI()

# 사용자 입력문장
class question_txt(BaseModel):
    text: str

def chatbot_answer(executor, session_id, query):
#     session_id = ChatMessageHistory()
#     agent = RunnableWithMessageHistory(
#     executor,
#     lambda session_id: session_id,
#     # 프롬프트의 질문이 입력되는 key: "input"
#     input_messages_key="input",
#     # 프롬프트의 메시지가 입력되는 key: "chat_history"
#     history_messages_key="chat_history",
# )
    
    
    
    response = executor.invoke(
        {'input': query},
        #  config={'configurable': {'session_id': session_id}},
    )
    return response['output']


# chatbot 관련 변수 설정
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

vectorstore =  Chroma(persist_directory="./testdb", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 5})   

retriever_tool = create_retriever_tool(
    retriever,
    name="db_search",
    description="놀이 관련 정보를 연결된 데이터베이스에서 검색합니다.",
)
tools = [retriever_tool]

llama3 = ChatOpenAI(
    base_url="http://sionic.chat:8001/v1",
    api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
    model="xionic-ko-llama-3-70b",
    temperature=0.1,
)

json_prompt = hub.pull("teddynote/react-chat-json-korean")

llama3_agent = create_json_chat_agent(llama3, tools, json_prompt)

llama3_agent_executor = AgentExecutor(
    agent=llama3_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

gpt = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = hub.pull("hwchase17/openai-functions-agent")


gpt_agent = create_openai_functions_agent(gpt, tools, prompt)
gpt_agent_executor = AgentExecutor(agent=gpt_agent, tools=tools, verbose=True)


@app.post("/chat")
async def process_text(data: question_txt):
    response = chatbot_answer(gpt_agent_executor, '123', data.text)
    return {"chatbot_response": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
