import utils
import numpy as np
import config 
import time
import pickle 
from embedder import SpecterEmbedder
from query_builder import QueryBuilder
from retriever import FaissRetriever
from soft_bias import SoftBiasScorer



class UnifiedOnlinePaperProcess:
    def __init__(self):
        print("[No Fusion] Online 모듈 로딩 중...")
        start_time = time.time()

        self.query_builder = QueryBuilder()
        self.embedder = SpecterEmbedder()
        self.retriever = FaissRetriever()
        self.bib_scorer = SoftBiasScorer()
        with open(config.EMBEDDING_DB_PATH, "rb") as f:
            self.embedding_db = pickle.load(f)

        print(f"[초기화 완료] 소요 시간 : {time.time() - start_time:.2f}초\n")
    
    def run_pipeline(self, user_input):
        '''
        프론트엔드에서 온 데이터 받아서 최종 피처 반환 (Early Fusion 방식)
        - user_input : { "title" : str, "abstract" : str, "context" : str, "bib_ids" : list}
        '''
        req_id = "req_" + utils.get_timestamp()

        # 1. 토큰 제한 및 쿼리 결합
        # context first 전략 : 가장 중요한 현재 문맥을 맨 앞에 배치
        # 약 195/39/260 토큰 (토큰 = 1.3배*단어)

        # 1. QueryBuilder 통해 context는 정확하게 100토큰 가져옴 
        _, safe_context = self.query_builder.build_online_query(
            user_input_text=user_input.get('context', ''),
            title=user_input.get('title', ''),
            abstract=user_input.get('abstract', '')
        )

        # 2. Title과 Abstract는 512토큰 방어용으로 단어 수 자르기
        safe_title = utils.truncate_words(user_input.get('title', ''), 30)
        safe_abstract = utils.truncate_words(user_input.get('abstract', ''), 280)

        unified_text = f"{safe_context} [SEP] {safe_title} [SEP] {safe_abstract}"

        # 2. 단일 임베딩 연산
        u_vec = self.embedder.encode([unified_text])

        # 3. FAISS 검색 
        u_res = self.retriever.search(u_vec, [req_id], source = ["unified"], top_k = 100)

        # FAISS 결과는 [[]] 형태이므로, 첫 번째 결과(해당 쿼리의 TOP-K)만 추출 
        candidates = u_res[0]
        print(f"[Debug] FAISS에서 뽑힌 후보 개수: {len(candidates)}")

        # 4. Soft bias 계산 (RRF Fusion 생략)
        bib_ids = user_input.get('bib_ids', [])
        biased = self.bib_scorer.soft_bias(candidates, bib_ids, self.embedding_db)
        print(f"[Debug] 각 후보 논문에 Bib score 점수 추가 완료 ...")

        # 5. 정규화
        raw_sims = [c['sim'] for c in biased]
        raw_bibs = [c.get('bib_score', 0.0) for c in biased]

        norm_sims = utils.normalize(raw_sims)
        norm_bibs = utils.normalize(raw_bibs)

        # 6. 최종 데이터 형식 가공
        clean_candidates = []
        for idx, cand in enumerate(biased):
            clean_candidates.append({
                "paper_id" : str(cand['paper_id']),
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            })

        return {
            "query_id": req_id,
            'context': user_input.get('context', ''),
            "candidates": clean_candidates
        }
    
if __name__ == "__main__":
    # 서버 객체 생성
    engine = UnifiedOnlinePaperProcess()
    
    # 프론트엔드에서 받은 가상의 JSON 데이터
    sample = {
    "title": "Towards Autonomous LLM Agents for Recommendation",
    "abstract": "This paper investigates the potential of Large Language Models as autonomous agents in recommendation systems. We focus on overcoming existing controller limitations during autonomous interaction...",
    "context": "LLM 에이전트를 활용한 추천 시스템에서 가장 큰 과제 중 하나는 프로파일 기반 결정 로직을 넘어선 진정한 의미의 자율적 상호작용(autonomous interaction)을 구현하는 것이다. 특히 기존 컨트롤러의 한계를 지적한 연구들을 살펴보면 \cite{",
    "bib_ids": ["Wang2023_AgentCF", "ToolRec2024"]
    }
    
    print("\n 실시간 피처 추출 시작...")
    start_time = time.time()
    
    # 파이프라인 가동
    result = engine.run_pipeline(sample)
    
    print(f"처리 완료! 소요 시간: {time.time() - start_time:.4f}초")
    print(f"반환된 후보 개수: {len(result['candidates'])}개")
    
    # 리스트가 비어있을 때 IndexError가 나지 않도록 조건문 추가
    if len(result['candidates']) > 0:
        print(f"1번째 후보 샘플: {result['candidates'][0]}")
        print(f"2번째 후보 샘플: {result['candidates'][1]}")
        print(f"3번째 후보 샘플: {result['candidates'][2]}")
        print(f"4번째 후보 샘플: {result['candidates'][3]}")
        print(f"5번째 후보 샘플: {result['candidates'][4]}")
        
    else:
        print("후보가 없습니다! 위쪽 Debug 로그에서 어디서 0이 되었는지 확인하세요.")

    utils.save_json(result, "unified_online_output.json")
    print("'unified_online_output.json'에 저장 완료")
        