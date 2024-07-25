1.naver_review.ipynb
네이버 플레이스 에서 각 장소에 대해 리뷰 수집

2.kakaomap_review.ipynb
카카오맵 에서 각 장소에 대해 리뷰 수집

3.API 좌표.ipynb
도로명주소를 이용해 좌표로 변환(kakao api 사용)

4.BERT활용 텍스트 분류.ipynb
사전 학습된 BERT(kykim/bert-kor-base  => 한국어 감정분석모델)모델을 fine tuning 
-> 학습에 사용한 데이터 출처 https://huggingface.co/datasets/leey4n/KR3

5.리뷰 전처리 및 긍부정 분류.ipynb
1) 전체 리뷰에 대해서 맞춤법 및 전처리(이모티콘, 기호 등등)수행
2) fine tuning한 커스텀 모델 불러오기
3) 전체 리뷰를 긍정 부정으로 분류
4) 긍정 리뷰 => 자체 토큰화 실시 => 긍정키워드 추출
5) 긍정키워드로 이루어진 워드클라우드 구현
