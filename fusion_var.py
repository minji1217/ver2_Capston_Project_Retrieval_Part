import numpy as np
import config 

def rank_fusion_var(paper_query_results, context_query_results, p_vecs, c_vecs, embedding_db):
    # 실제 온라인 디버그용
    # print(f"[Debug] FAISS가 가져온 p_res 길이: {len(paper_query_results[0])}")
    # print(f"[Debug] FAISS가 가져온 c_res 길이: {len(context_query_results[0])}")
    '''
    [RRF 기반 배치 단위 Fusion Logic]
    1. paper_query_results : Paper query 검색 결과 top-k (배치)
    2. context_query_results : Context query 검색 결과 top-k (배치) 


    *RRF : 순위 기준으로 합침 
    -> 이때, 원본 코사인 유사도 score는 paper_sim, context_sim으로 각각 보존
    -> 최종 출력 형태 : RRF 점수로 재정렬된 TOP-K 논문 리스트 
    '''
    final_results = []
    k_val = config.RRF_K
    
    # 한 논문의 한 placeholder에서의 paper/context query에 대해 수행
    for p_res, c_res, p_v, c_v in zip(paper_query_results, context_query_results, p_vecs, c_vecs):
        fusion_result = {}
        # 1. 모든 후보(PAPER/CONTEXT 결과 top-75)의 paper id를 합집합으로 모음
        # 예: {'1975812', '124346', ...}
        all_pids = set([item['paper_id'] for item in p_res] + [item['paper_id'] for item in c_res])

        # p_res, c_res를 dict로 변환
        # 기존 p_res, c_res는 리스트이기에 O(1)으로 찾기 위해 dict로 변환
        # -> 변환 후 : { '1975812': {"paper_id":, "score":, ...}, '124346':{ }, ...}
        p_lookup = {item['paper_id']: item for item in p_res}
        c_lookup = {item['paper_id']: item for item in c_res}

        
        # 1차원 벡터로 변환 (내적 연산 준비)
        p_vec_1d = np.squeeze(p_v)
        c_vec_1d = np.squeeze(c_v)

        for pid in all_pids:
            # 2. RRF 점수 계산 
            p_item = p_lookup.get(pid)
            c_item = c_lookup.get(pid)

            rrf_score = 0.0
            # rrf 점수 누적합 
            if p_item: rrf_score += 1.0 / (k_val + p_item['rank'])
            if c_item: rrf_score += 1.0 / (k_val + c_item['rank'])

            fusion_result[pid] = {
                "rrf_score": rrf_score,
                "p_sim": p_item['score'] if p_item else None,
                "c_sim": c_item['score'] if c_item else None
            }

        # 3. RRF 기준 1차 필터링 (Top-100 선발)
        sorted_by_rrf = sorted(fusion_result.items(), key=lambda x: x[1]['rrf_score'], reverse= True)
        top_candidates = sorted_by_rrf[:config.TOP_K_FINAL]

        # 4. Top-100 후보들 각각에 대해 missing sim 계산 및 최종 점수 계산 
        refined_list = []
        for pid, info in top_candidates:
            p_sim = info['p_sim']
            c_sim = info['c_sim']


            # SPECTER2 L2 정규화하기에, 없는 유사도 점수만 내적으로 계산
            if p_sim is None or c_sim is None:
                target_vector = embedding_db[pid] # 후보 논문의 임베딩
                if target_vector is None: 
                    continue

                if p_sim is None:
                    p_sim = float(np.dot(p_vec_1d, target_vector))
                if c_sim is None:
                    c_sim = float(np.dot(c_vec_1d, target_vector))
            
            # 가중합 계산 
            final_sim = (config.PAPER_SIM_WEIGHT * p_sim) + (config.CONTEXT_SIM_WEIGHT * c_sim)
            
            refined_list.append({
                "paper_id": pid,
                "sim": final_sim 
            })

        # 5. 최종 sim 점수 기준으로 최종 재정렬
        refined_list.sort(key=lambda x: x['sim'], reverse=True)

        # 6. 결과 생성
        placeholder_results = []
        q_id = c_res[0]['query_id'] if c_res else p_res[0]['query_id'] if p_res else "UNKNOWN"

        for new_rank, cand in enumerate(refined_list):
            placeholder_results.append({
                "query_id": q_id,  
                "rank": new_rank + 1, # Sim 기준 최종 순위
                "paper_id": cand['paper_id'],
                "sim": cand['sim']
            })

        final_results.append(placeholder_results)

    return final_results