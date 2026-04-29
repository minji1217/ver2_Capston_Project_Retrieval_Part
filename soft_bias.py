import numpy as np
import utils 
import config 

class SoftBiasScorer:
    '''
    def __init__(self, embedding_db_path = config.EMBEDDING_DB_PATH):
        # 딕셔너리 형태의 임베딩 DB를 메모리에 로드 
        # 예시 : {'paper_123': [3.0, 1.2, ... ](768 dim)}
        self.embedding_db = utils.load_pickle(embedding_db_path)
    '''

    def soft_bias(self, candidate_list, user_bib_ids, embedding_db, decay_factor = 0.5):
        '''
        [soft bias 계산]
        -> 각 후보 논문과 user bib내 모든 논문들간 유사도 계산 후, 지수감쇠 적용해 user bib 반영한 후보 논문 리스트 
        - candidate_list : FAISS가 찾은 100개 후보 논문 리스트 
        - user_bib_ids : 유저의 bib 논문 ID 리스트 
        - decay_factor : 유사도 높은 순으로 점수 누적할 때, 후순위 논문들의 영향력 조절하는 값 (지수)
        '''        

        # 1. 유저의 bib가 없는 경우 그대로 반환
        if not user_bib_ids:
            for item in candidate_list:
                item['bib_score'] = 0.0
            return candidate_list
        
        # 2. 유저의 bib 논문 벡터들을 DB에서 조회 (using embedding_db)
        # DB에 존재하는 ID만 가져옴 
        bib_embs = [embedding_db[bid] for bid in user_bib_ids if bid in embedding_db]

        # 2-1. 필터링 후 남은 논문벡터가 없다면 0점 처리 
        if not bib_embs:
            for item in candidate_list:
                item['bib_score'] = 0.0
            return candidate_list
        
        # For 유사도 계산, 유저 bib 행렬 생성 (N X 768) : N은 유저 bib 내 논문 개수 
        bib_matrix = np.array(bib_embs)

        # 3. 각 후보 논문에 대해 bib 반영한 최종 점수 계산 
        for item in candidate_list:
            p_id = item['paper_id']

            # 후보 논문이 DB에 없으면 0점 처리
            if p_id not in self.embedding_db:
                item['bib_score'] = 0.0
                continue    

            # 후보 논문의 768차원 벡터를 embedding_db에서 바로 가져옴
            p_vec = self.embedding_db[p_id]

            # 행렬 내적 연산 : 후보 논문 1개 - 유저 bib N개 -> N개의 유사도 점수 배열 생성 
            # SPECTER2 벡터를 L2 정규화하였기에 내적이 곧 코사인 유사도 
            sims = np.dot(bib_matrix, p_vec)

            # 유사도순으로 내림차순 정렬
            sorted_sims = np.sort(sims)[::-1]

            # Decayed Sum 적용 : 1등은 1배, 2등은 0.5배, 3등은 0.25배, ... 
            weights = decay_factor ** np.arange(len(sorted_sims)) # 0,1,2,...
            
            # [추가] 정규화 로직 : 가중 평균 계산 
            # 이때 분모가 0이 되는 것을 방지하기 위해 아주 작은 값 (1e-9) 더해줌
            raw_sum = np.sum(sorted_sims * weights)
            weight_sum = np.sum(weights) # <-
            
            item['bib_score'] = float(raw_sum / (weight_sum + 1e-9)) # <-

            
        return candidate_list