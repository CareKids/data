import requests
import base64
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib


# FastAPI 서버 URL
process_text_url = "http://127.0.0.1:8000/process_text"
recommend_url = "http://127.0.0.1:8000/recommend"
decode_img_url = "http://127.0.0.1:8000/decode_image"

# # 요청에 보낼 데이터
input_payload = {
    "gu_name" : "강남구",
    "text": "맛있는 중국집을 알려줘"
}
# input_payload = {
#     "gu_name" : "강동구",
#     "text": "아이가 좋아할 만한 미술관이나 기념관을 알려줘"
# }
# input_payload = {
#     "gu_name" : "강서구",
#     "text": "아이가 체험을 하거나, 재밌게 놀 수 있는 곳을 알려줘"
# }

# 헤더 설정
headers = {
    "Content-Type": "application/json"
}

# 1. /process_text 호출
response_process_text = requests.post(process_text_url, json=input_payload, headers=headers)
processed_text = response_process_text.json()['processed_text']

# 2. /recommend 호출
recommend_payload = {
    "gu_name": input_payload['gu_name'],  # 사용자가 속한 자치구
    "clean_text": processed_text
}

response_recommend = requests.post(recommend_url, json=recommend_payload, headers=headers)

print(processed_text)
# print(recommend_payload)
# print(response_recommend)

# 응답 데이터 출력
# print("Processed Text Response:", response_process_text.json())
# print("Recommendation Response:", response_recommend.json())

def decode_to_image_arr(encode_str):
    img_decode = base64.b64decode(encode_str)
    img = Image.open(io.BytesIO(img_decode))
    return np.array(img)



# 이미지 인코딩 데이터 활용 예제
# 중요사항 : 대분류 - 음식 일때만 wordcloud를 만들었기에, 음식과 관련된 사용자 입력문장을 작성
def decode_to_image_arr(encode_str):
    img_decode = base64.b64decode(encode_str)
    img = Image.open(io.BytesIO(img_decode))
    return np.array(img)


# 추천 결과 중 하나의 이미지 데이터를 가져와서 표시
recommendations = response_recommend.json()['recommendations']
if recommendations:
    first_recommendation = recommendations[0]['metadata']  # 첫 번째 추천 결과 선택
    second_recommendation = recommendations[1]['metadata']  # 첫 번째 추천 결과 선택
    third_recommendation = recommendations[2]['metadata']  # 첫 번째 추천 결과 선택
    
    # 이미지 데이터가 있는지 확인하고 디코드하여 출력
    for metadata in [first_recommendation, second_recommendation, third_recommendation]:
        if 'wordcloud_img_arr' in metadata:
            img_data = metadata['wordcloud_img_arr']
            decode_img = decode_to_image_arr(img_data)
            img = Image.fromarray(decode_img)

            # 이미지 출력
            plt.imshow(img)
            plt.title(f"""콘텐츠명 : {metadata['content']}, 대분류 - 중분류 : {metadata['large_category']} - {metadata['medium_category']}
                        사용자 문장 전처리 전: {input_payload['text']}
                        사용자 문장 전처리 후: {processed_text}
                        사용자 자치구명 - 추천된 카테고리 자치구 명 : {input_payload['gu_name']}->{metadata['gu_name']}""")
            plt.axis('off')
            plt.show()
