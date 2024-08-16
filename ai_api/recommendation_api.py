import pandas as pd
import numpy as np
import re
import joblib
import io

# 맞춤법검사 PLM 관련
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

# 전처리 관련
from pykospacing import Spacing # 띄어쓰기 교정
from soynlp.normalizer import emoticon_normalize  # 반복 단어 제거
from konlpy.tag import Okt  # 품사 태깅

# VectorDB 관련
import chromadb

# 시각화 관련
import matplotlib.pyplot as plt
from PIL import Image
import base64

# API 관련
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    # allow_origins=["http://localhost:3000/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

RANDOM_SEED = 42
W2V_MODEL = joblib.load('./model/classification_vector_model.pkl')  # Word2Vec 학습 후 모델 불러오기
binary_classification_model = joblib.load('./model/binary_classification_model.pkl')    # logistic regression 학습 후 모델 불러오기

# 맞춤법 교정 모델 text2text GAN (Pretrained Language Model : PLM 모델 활용)
corrector_model = T5ForConditionalGeneration.from_pretrained('j5ng/et5-typos-corrector')
corrector_tokenizer = T5Tokenizer.from_pretrained('j5ng/et5-typos-corrector')

typos_corrector = pipeline(
    "text2text-generation",
    model = corrector_model,
    tokenizer = corrector_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework = "pt",
)

# ChromaDB Client 초기화
client = chromadb.PersistentClient(path="./recommendation_DB/")
w2v_embedded_collection = client.get_collection(name="recommendation_chroma_db")

# 데이터베이스 전체 가져오기
get_database = w2v_embedded_collection.get()
    
# 특정 키워드 가중치
weighted_keywords_list = ['숲','꽃', '한식', '중식', '탕수육', '자장면', '짬뽕', '중국집', '일식', '경양식', '한식'
                          '패스트푸드', '햄버거', '빵', '아시아푸드', '쌀국수', '중식', '카페', '양식', '민간', '키즈카페', '오락실',
                          '공연장', '문화예술회관', '공공형', '서울형', '공연', '문화', '문화원', '강의', '문화생활', '박물관', '도서관', 
                          '책', '미술관', '작품', '기념관', '전통', '숲체험']
keyword_weight_value = 6


# 사용자 자치구명, 입력문장
class TextData(BaseModel):
    gu_name: str
    text: str

# 추천에 사용될 사용자 자치구명, 전처리 문장,
class RecommendRequest(BaseModel):
    gu_name: str
    clean_text: str

# 확인용. base64인코딩된 이미지
class ImageData(BaseModel):
    base64_encode_str: str

# 맞춤법 교정
def spelling_check(text):
    """
    맞춤법 교정
    """
    # 텍스트를 문장 단위로 분할합니다.
    parts_of_text = re.split(r'([,.!?])', text)  # 구분자를 유지하기 위해 그룹화

    correct_text = []

    for i in range(0, len(parts_of_text), 2):
        sentence = parts_of_text[i]
        if sentence.strip():  # 문장이 비어있지 않은 경우에만 처리
            # 맞춤법 교정을 적용합니다.
            corrected = typos_corrector("맞춤법을 고쳐주세요: " + sentence,
                                        max_length=128,
                                        num_beams=5,
                                        early_stopping=True)[0]['generated_text']
            corrected = corrected.rstrip('.,!?')
            correct_text.append(corrected)
        
        # 구분자(특수문자)를 다시 추가합니다.
        if i + 1 < len(parts_of_text):
            correct_text.append(parts_of_text[i + 1])

    return ''.join(correct_text)

# 띄어쓰기 처리
def spacing_processing(text):
    """
    띄어쓰기 처리
    """

    spacing = Spacing()
    text = spacing(text)
    return text

# 반복 문자 처리
def repetitive_character_processing(text):
    """
    반복 문자 처리
    """
    text = emoticon_normalize(text, num_repeats=2)
    return text


# 사용자 입력 데이터 처리
def input_process_text(text):
    text = repetitive_character_processing(text)
    text = spacing_processing(text)
    text = spelling_check(text)
    return text

@app.post("/process_text")
async def process_text(data: TextData):
    processed_text = input_process_text(data.text)
    return {"processed_text": processed_text}

# 한글이 아닌 특수문자 제거
def extract_keywords(text):

    # Konlpy
    okt = Okt()

    # 불용어처리
    stopwords = ['한테', '모레', '로', '나', '있다', '이미', '많이', '하다', '것', '종일', '매우', '하며', '있는', '곳', '저', '그것', '내일', '과', '그렇다면', '너',
                '누구의', '언제', '있고', '에서', '무엇', '정말', '와', '혹은', '이따가', '가', '그녀', '어제', '진짜', '실제로', '무슨', '만일', '가장', '등', '몹시',
                '께', '하지만', '을', '아주', '이', '누구', '그리고', '에게서', '잠깐', '모두', '에', '으로', '위한', '너무', '여기', '어디', '그러나', '의', '계속', '누가',
                '또는', '한참', '하고', '한테서', '우리', '이것', '까지', '조금', '그', '오늘', '적게', '그렇지만', '만약', '수', '를', '부터', '무엇이', '그래서', '는',
                '및', '방금', '은', '등등', '그런데', '거기', '어떤', '에게', '도', '벌써', '당신', '지금', '다', '저것', '할', '그냥', '사실', '어느', '얼마나', '읽을',
                '잠시', '어떻게', '한', '금방', '저기', '가득', '추천', '알려줘', '해줘', '어린이', '아이', '먹다', '맛', '가다', '오다', '자다', '같다', '더', '넘다', '또', 
                '이다', '아쉽다', '바', '꼭', '보다', '되어다', '나오다', '요', '들다', '처', '다음', '때', '되다', '거', '싶다', '점', '주다', '리지', '널', '크리스', '크림', 
                '피', '드', '안', '역시', '받다', '글레이', '좀', '꾸다', '지점', '해', '덮다', '집', '굿', '시', '비', '없다', '나다', '생기다', '바로', '못', '날']
    
    parts_of_text = re.split(r'([,.!?])', text)  # 구분자를 유지하기 위해 그룹화
    processed_tokens = []

    # 한글 특성상 ,.!? 구분자 단위로, 문맥에 맞게 포함이 되어있기에, 구분자가 있다면, 그 내에서 토큰화 방식 구현 
    for part in parts_of_text:
        tokens = okt.pos(part.strip(), stem=True)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # 명사, 형용사, 동사인 경우에만 처리
            if token[1] in ['Noun', 'Adjective', 'Verb'] and token[0] not in stopwords:
                processed_tokens.append(token[0])

                # 명사이거나 형용사인 경우 -> 명사 + 명사 : 복합명사 ex) 숲체험 / 형용사 + 명사 : 꾸며주는 명사 ex) 친절한 직원
                if token[1] in ['Noun', 'Adjective']:
                    # 다음 토큰도 명사이고, 두 토큰을 합쳐서 하나의 토큰으로 처리
                    if i < len(tokens) - 1 and tokens[i + 1][1] in ['Noun'] and tokens[i + 1][0] not in stopwords:
                        combined_token = token[0] + tokens[i + 1][0]
                        processed_tokens.append(combined_token)
                        # 형용사 + 명사 + 명사 또는 명사 + 명사 + 명사 인 경우 앞에 두개의 토큰을 더한 값은 삭제 후 새로운 토큰 삽입
                        # ex) 넓은 키즈카페, ex)문화예술회관
                        if i < len(tokens) - 2 and tokens[i + 2][1] in ['Noun'] and tokens[i + 2][0] not in stopwords:
                            combined_token = token[0] + tokens[i + 1][0] + tokens[i + 2][0]
                            processed_tokens.pop(-1)
                            processed_tokens.append(combined_token)

                # 부정 표현 처리
                if i < len(tokens) - 1 and tokens[i + 1][0] == '않다':
                    combined_token = token[0] + '않다'
                    processed_tokens.pop(-1)  # 이전 토큰 제거
                    processed_tokens.append(combined_token)
                    i += 1  # '않다' 토큰 건너뛰기
            i += 1
    return processed_tokens

# input 문장에 대한 가중치 적용 word2vec 벡터화 함수 정의
def weight_sentence_to_vector(keywords, model):
    word_vectors = []
    for word in keywords:
        if word in model.wv:
            word_vector = model.wv[word] * (keyword_weight_value if word in weighted_keywords_list else 1)
            word_vectors.append(word_vector)
    if not word_vectors:  # 단어 벡터가 없는 경우 영벡터 반환
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# 유사한 콘텐츠 추천 함수 정의
def recommend_similar_content(input_gu_name, input_preprocessed_text, classification_model):
    """
    input_gu_name : 사용자 자치구 명
    input_preprocessed_text : 전처리 후 사용자 입력 문장
    classification_vector_model : 벡터화 모델 Word2Vec 사용
    classification_model : 먹기 or 나머지 로 이진분류를 위한 모델
    threshold : 0. / 유사도 임계값(threshold < 유사도 인것만 리턴)
    """
    threshold = 0.5
    input_preprocessed_text = input_process_text(input_preprocessed_text)
    input_keywords = extract_keywords(input_preprocessed_text)
    input_vectors = weight_sentence_to_vector(input_keywords, W2V_MODEL).reshape(1, -1)
        
    # 이진 분류 예측
    binary_category_pred = classification_model.predict(input_vectors)[0]
    
    # 이진분류 (0:먹기 / 1:배우기+놀기)를 통해 필터링된 ids 리스트 만들기
    filtered_binary_classification_ids = []
    if get_database and 'metadatas' in get_database:
        for i, metadata in enumerate(get_database['metadatas']):
            if binary_category_pred == 0 and metadata.get('large_category') == '먹기':
                filtered_binary_classification_ids.append(get_database['ids'][i])
            elif binary_category_pred != 0 and metadata.get('large_category') != '먹기':
                filtered_binary_classification_ids.append(get_database['ids'][i])


    # 사용자 입력 문장 벡터를 통한, 유사도 상위 n_results개 데이터베이스 검색
    similarity_database = w2v_embedded_collection.query(query_embeddings = input_vectors, n_results = 500)

    # distances 가 threshold 보다 작은 ids 리스트 만들기
    filtered_distances_ids = []
    for idx, dist in enumerate(similarity_database['distances'][0]):
        if dist <= threshold and similarity_database['ids'][0][idx] in filtered_binary_classification_ids:
            print(dist)
            filtered_distances_ids.append(similarity_database['ids'][0][idx])

    # filtered_distances_ids를 통한, 데이터베이스 가져오기
    filtered_threshold_database = w2v_embedded_collection.get(ids = filtered_distances_ids)

    # 자치구 명이 사용자 입력 자치구 명과 동일한 것 상위에 보여주기
    same_gu_name_results = []
    different_gu_name_results = []
    for idx, result in enumerate(filtered_threshold_database['metadatas']):
        if result['gu_name'] == input_gu_name:
            same_gu_name_results.append({
                'id': filtered_threshold_database['ids'][idx],
                'metadata': result,
            })
        else:
            different_gu_name_results.append({
                'id': filtered_threshold_database['ids'][idx],
                'metadata': result,
            })

    # 결과 병합(동일 자치구 상단, 다른 자치구 하단): 동일 자치구 결과 + 다른 자치구 결과
    
    sorted_results = same_gu_name_results + different_gu_name_results
    
    # # 확인용...
    # print(f"이진분류를 통한 예측 대분류 : {'먹기' if binary_category_pred == 0 else '배우기+놀기'}")
    # set_medium_categories = set()
    # for i in range(len(sorted_results)):
    #     set_medium_categories.add(sorted_results[i]['metadata']['medium_category'])
    # print(f"추천된 전체 데이터 개수 : {len(sorted_results)}")
    
    # print(f"같은 자치구 내 중분류 유니크 리스트 : {same_gu_name_results[0]['metadata']['medium_category'] if len(same_gu_name_results) > 0 else '같은 자치구내 검색X'}")
    # print(f"추천된 중분류 유니크 리스트 : {set_medium_categories}")
    # print(f"첫번째 추천된 중분류 : {sorted_results[0]['metadata']['medium_category']}")

    return sorted_results

@app.post("/recommend")
async def recommend(data: RecommendRequest):
    user_state = data.gu_name
    preprocessed_text = data.clean_text
    recommendation_json = recommend_similar_content(input_gu_name = user_state, input_preprocessed_text = preprocessed_text, classification_model = binary_classification_model)
    return {'recommendations': recommendation_json}

def decode_to_image_arr(encode_str):
    img_decode = base64.b64decode(encode_str)
    img = Image.open(io.BytesIO(img_decode))
    return np.array(img)

@app.post("/decode_image")
async def decode_image(img_data: ImageData):
    encode_str = img_data.base64_encode_str
    img_arr = decode_to_image_arr(encode_str)
    return {"img_arr" : img_arr}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)