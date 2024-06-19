# Vector 데이터를 저장할 공간 할당
# V = (x, y, z, ......)
V = []


# Vector 데이터 추가하기
def add_vector(V, vector):
    V.append(vector)


# Vector 데이터 조회하기
def get_vectors(V):
    return V


# 내적을 이용한 유사도 측정 (cosine similarity)
def get_similarity(V, vector1, vector2):
    similarity = 0
    for i in range(len(vector1)):
        similarity += vector1[i] * vector2[i]
    return similarity


# 유클리디안거리를 이용한 유사도 측정 (euclidian distance)
def get_similarity_by_uclidian_distance(V, vector1, vector2):
    similarity = 0
    for i in range(len(vector1)):
        similarity += (vector1[i] - vector2[i]) ** 2
    return similarity


# 최대 유사도 측정
def get_max_similarity(V, vector):
    max_similarity = 0
    for v in get_vectors(V):
        similarity = get_similarity(V, vector, v)
        if similarity > max_similarity:
            max_similarity = similarity
    return max_similarity


# 최소 유사도 측정
def get_min_similarity(V, vector):
    min_similarity = get_max_similarity(V, vector)
    for v in get_vectors(V):
        similarity = get_similarity(V, vector, v)
        if similarity < min_similarity:
            min_similarity = similarity
    return min_similarity


# 문장을 토큰화 하기
def tokenize(sentence):
    # 조사 제거
    sentence = (
        sentence.replace("은", "")
        .replace("는", "")
        .replace("이", "")
        .replace("가", "")
        .replace("을", "")
        .replace("를", "")
        .replace("에", "")
        .replace("에서", "")
        .replace("와", "")
        .replace("과", "")
        .replace("있", "")
        .replace("하", "")
        .replace("것", "")
        .replace("들", "")
        .replace("의", "")
        .replace("로", "")
        .replace("으로", "")
        .replace("된", "")
        .replace("다", "")
        .replace("하", "")
    )
    return sentence.split()


## 수동(하드코딩) 임베딩 및 테스트 과정 ##
# dimension define
# 1d 종 군집 차원 (포유류 or 조류 1, -1)
# 2d 종 군집 차원 (어류 or 양서류 1, -1)
# 3d 기후별 차원 (열대 or 냉한 1, -1)
# 4d 특징별 차원 (비행 or 수중 1, -1)
# 5d 특징별 차원 (빠름 or 느림 1, -1)

# [동물 임베딩 참고자료]
# 1. 포유류: 고양이, 강아지, 사자, 호랑이, 곰
# 2. 조류: 독수리, 비둘기, 참새, 앵무새, 타조
# 3. 어류: 상어, 돌고래, 고래, 참치, 병어
# 4. 양서류: 개구리, 도롱뇽, 뱀, 거북, 독사
# 5. 열대: 원숭이, 악어, 코끼리, 코뿔소, 기린, 타조
# 6. 온난: 토끼, 다람쥐, 고슴도치, 두더지, 족제비
# 7. 냉한: 펭귄, 북극곰
# 8. 비행: 독수리, 참새, 앵무새, 악어
# 9. 수중: 상어, 돌고래, 고래, 참치, 병어
# 10. 빠름: 독수리, 타조, 상어, 돌고래
# 11. 보통속도: 고양이, 강아지, 사자, 호랑이, 곰
# 11. 느림: 거북이, 독사, 코끼리, 코뿔소, 기린

# 동물 임베딩
allAnimalsEmbedding = [
    ["고양이", (1, 0, 0, 0, 0)],  # index: 0
    ["강아지", (1, 0, 0, 0, 0)],  # index: 1
    ["사자", (1, 0, 0, 0, 0)],  # index: 2
    ["호랑이", (1, 0, 0, 0, 0)],  # index: 3
    ["곰", (1, 0, 0, 0, 0)],  # index: 4
    ["독수리", (-1, 0, 0, 1, 0)],  # index: 5
    ["비둘기", (-1, 0, 0, 1, 0)],  # index: 6
    ["참새", (-1, 0, 0, 1, 0)],  # index: 7
    ["앵무새", (-1, 0, 0, 1, 0)],  # index: 8
    ["타조", (-1, 0, 1, 0, 1)],  # index: 9
    ["상어", (0, 1, 0, -1, 1)],  # index: 10
    ["돌고래", (0, 1, 0, -1, 1)],  # index: 11
    ["고래", (0, 1, 0, -1, 0)],  # index: 12
    ["참치", (0, 1, 0, -1, 1)],  # index: 13
    ["병어", (0, 1, 0, -1, 0)],  # index: 14
    ["개구리", (0, -1, 0, -1, 1)],  # index:15
    ["도롱뇽", (0, 0, 1, 0, 0)],  # index: 16
    ["뱀", (0, 0, 1, 0, 0)],  # index: 17
    ["거북", (0, 0, 1, 0, -1)],  # index: 18
    ["독사", (0, 0, 1, 0, 0)],  # index: 19
    ["원숭이", (1, 0, 1, 0, 1)],  # index: 20
    ["악어", (0, 0, 1, 1, -1)],  # index: 21
    ["코끼리", (1, 0, 1, 0, 0)],  # index: 22
    ["코뿔소", (1, 0, 1, 0, 0)],  # index: 23
    ["기린", (1, 0, 1, 0, 0)],  # index: 24
    ["토끼", (1, 0, 0, 0, 1)],  # index: 25
    ["다람쥐", (1, 0, 0, 0, 0)],  # index: 26
    ["고슴도치", (1, 0, 0, 0, 0)],  # index: 27
    ["두더지", (1, 0, 0, 0, -1)],  # index: 28
    ["족제비", (1, 0, 0, 0, 0)],  # index: 29
    ["펭귄", (-1, 0, -1, -1, -1)],  # index: 30
    ["북극곰", (1, 0, -1, 0, 0)],  # index: 31
    ## 이후는 문장을 인식하기 위한 vector 데이터 유사도를 측정해서 값을 도출 할때는 제외하고 있음 문장을 인식하기 위한 데이터로만 사용하도록함 (실제 AI는 이런식으로 동작하지 않을 것임)
    ["춥다", (0, 0, -1, 0, 0)],  # index: 32
    ["뜨겁다", (0, 0, 1, 0, 0)],  # index: 33
    ["빠르다", (0, 0, 0, 0, 1)],  # index: 34
    ["느리다", (0, 0, 0, 0, -1)],  # index: 35
    ["새", (-1, 0, 0, 0, 0)],  # index: 36
    ["포유류", (1, 0, 0, 0, 0)],  # index: 37
    ["조류", (-1, 0, 0, 1, 0)],  # index: 38
    ["어류", (0, 1, 0, -1, 1)],  # index: 39
    ["양서류", (0, 0, 1, 0, 0)],  # index: 40
    ["열대", (1, 0, 1, 0, 1)],  # index: 41
    ["온난", (1, 0, 1, 0, 0)],  # index: 42
    ["냉한", (0, 0, -1, 0, 0)],  # index: 43
    ["비행", (-1, 0, 0, 1, 0)],  # index: 44
    ["수중", (0, 1, 0, -1, 1)],  # index: 45
    ["빠름", (0, 0, 0, 0, 1)],  # index: 46
    ["느림", (0, 0, 0, 0, -1)],  # index: 47
    ["추운", (0, 0, -1, 0, 0)],  # index: 48
    ["수영", (0, 1, 0, -1, 1)],  # index: 49
    ["날지", (0, 0, 0, 0, 0)],  # index: 50
]

# 동물 임베딩 데이터 벡터에 추가하기
for animal in allAnimalsEmbedding:
    add_vector(V, animal[1])

# 벡터 데이터 조회하기
# print(get_vectors(V))

# 유사도 측정하기
# 독수리와 참새의 유사도
# print(get_similarity(V, get_vectors(V)[5], get_vectors(V)[7]))

# 독수리가 가질수 있는 최대 유사도
# print(get_max_similarity(V, get_vectors(V)[5]))

# 펭귄과 북극곰의 유사도
# print(get_similarity(V, get_vectors(V)[30], get_vectors(V)[31]))

# 북극곰이 가질수 있는 최대 유사도
# print(get_max_similarity(V, get_vectors(V)[31]))

# 춥다와 뜨겁다의 유사도
# print(get_similarity(V, get_vectors(V)[32], get_vectors(V)[33]))

###################### 테스트 할 문장 ######################
# 추운 곳에 살고 있는 새는 무엇인가요?
# 빠른 새는 무엇인가요?
# 빠른 어류는 무엇인가요?
sentence = "추운 곳에 살고 있는 새는 무엇인가요?"
#########################################################

# 문장 토큰화
tokens = tokenize(sentence)
print(tokens)

# 유효토큰 index 값으로 저장
validTokens = []

# 토큰을 순회 하면서 유사도 측정
_sumSimilarity_ = 0
for token in tokens:
    for animal in allAnimalsEmbedding:
        if token == animal[0]:
            validTokens.append(allAnimalsEmbedding.index(animal))
            similarity = get_max_similarity(V, animal[1])
            print(f"{token}의 최대 유사도: {similarity}")
            _sumSimilarity_ += similarity
            break
print(f"유효한 토큰인덱스: {validTokens}")


# 유효한 토큰을 순회하면서 모든 벡터데이터들의 유사도를 측정하여 가장 높은 유사도를 가진 동물을 찾아내기

# 동물별 유사도 결과 저장 공간
animalSimilarityResult = {}
bestSimilarityAnimal = ""

# 유사도 측정
for animal in allAnimalsEmbedding[0:32]:
    similarity = 0
    _sumSimilarity_ = 0
    for token in validTokens:
        sim = get_similarity(V, animal[1], get_vectors(V)[token])
        _sumSimilarity_ += sim
    # 유사도 결과 출력
    print(f"{animal[0]}의 유사도: {_sumSimilarity_}")
    animalSimilarityResult[animal[0]] = _sumSimilarity_

# 가장 높은 유사도를 가진 동물 찾기
bestSimilarityAnimal = max(animalSimilarityResult, key=animalSimilarityResult.get)
print(f"\"{sentence}\" 의 질문과 가장 유사한 동물: {bestSimilarityAnimal}")

