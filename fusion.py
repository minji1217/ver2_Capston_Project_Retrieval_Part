import numpy as np
import config 

def rank_fusion(paper_query_results, context_query_results):
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
    for p_res, c_res in zip(paper_query_results, context_query_results):
        fusion_result = {}

        # 1. Papery query 결과 먼저 해시맵에 저장
        for item in p_res:
            pid = item['paper_id']
            fusion_result[pid] = {
                "rrf_score" : 1.0 / (k_val + item['rank']),
                "paper_sim" : item['score'],
                "context_sim" : 0.0
            }

        

        # 2. Context query 결과 저장 
        for item in c_res:
            pid = item['paper_id']
            # paper_query top-k에 있던 논문이 context_query top-k에도 있는 경우 
            if pid in fusion_result: 
                fusion_result[pid]["rrf_score"] += (1.0 / (k_val + item['rank']))
                fusion_result[pid]["context_sim"] = item['score']
            else:
                fusion_result[pid] = {
                    "rrf_score" : 1.0 / (k_val + item['rank']),
                    "paper_sim" : 0.0,
                    "context_sim" : item['score']
                }
        # fusion_result : {"pid": {"rrf_score":, "paper_sim": ,...}, "pid": ''}

        # 3. rrf_score 기준으로 정렬
        # items() : dict에 있는 key, value를 한 쌍(튜플)으로 묶어서 리스트처럼 
        # items 예시 : [(pid, {"rrf_score: 0, "paper_sim: 0", "context_sim: 0"}), ...]
        # x[0] : pid, x[1] : {...}
        # 결과 : 튜플들 저장된 리스트 
        sorted_items = sorted(fusion_result.items(), key = lambda x: x[1]['rrf_score'], reverse = True)

        # 4. 최종 Top-K(150->100) 추출 (Paper/Context Query top-k에서 중복된 논문 처리 -> 가중합)
        # pid : paper id, info : "rrf_score", "paper_sim", "context_sim"
        placeholder_results = []
        for new_rank, (pid, info) in enumerate(sorted_items[:config.TOP_K_FINAL]):
            # p_res[0]['query_id']도 가능 
            # 현재 묶음의 query_id 가져오기 
            q_id = (
                c_res[0]['query_id'] if c_res 
                else p_res[0]['query_id'] if p_res
                else "UNKNOWN"
            )

            # 4-1. 중복 논문 처리 (가중합)
            # 만약, 중복 논문이 아니라면 항상 하나의 sim 값은 0이됨 
            paper = info['paper_sim']
            context = info['context_sim']

            if paper > 0 and context > 0:
                final_sim = config.PAPER_SIM_WEIGHT * info['paper_sim'] + config.CONTEXT_SIM_WEIGHT * info['context_sim']
            elif paper > 0: 
                final_sim = paper
            else:
                final_sim = context

            placeholder_results.append({
                "query_id" : q_id,  # 한 논문의 한 placeholder에서
                "rank" : new_rank + 1,
                "paper_id" : pid,
                "rrf_score" : round(info['rrf_score'], 6), # Fusion ranking시 사용한 점수
                "sim" : final_sim # 1개의 대표 점수로, 나중에 sim(q,p)계산시 사용됨 
            })

        final_results.append(placeholder_results)

    return final_results