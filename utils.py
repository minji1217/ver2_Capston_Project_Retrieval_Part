import json
import numpy as np
import pickle # 파이썬 객체(리스트, 딕셔너리 등)를 있는 그대로 파일로 저장 or 다시 불러올 수 있게 해주는 lib
import datetime

# pickle 파일은 이진데이터이므로 read binary 모드로 
# 이진파일을 원래의 파이썬 딕셔너리 형태로 변환 
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# 파이썬 객체를 Pickle 파일(이진 데이터)로 저장 
def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    
# data : 저장하고 싶은 실제 내용(리스트, 딕셔너리 등)
# file_path : 어디에 어떤 이름으로 저장할지 알려주는 주소 (예: C:/.../a.json)
def save_json(data, file_path):
    with open(file_path, "w", encoding = "utf-8") as f: # utf-8 : 한글깨짐 방지
        # data를 f에 적을 때, 4칸씩 띄우고, 한글 깨지지 않게 저장 
        json.dump(data, f, indent = 4) # ensure_ascii = False는 한글 깨지지 않게 저장 

# JSON 파일을 파이썬 딕셔너리/리스트로 불러오기     
def load_json(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        return json.load(f)
    

def normalize(values):
    """
    Min-Max Scaling을 수행하여 모든 값을 [0, 1] 범위로 변환.
    """
    values = np.array(values, dtype=float) 
    
    if values.size == 0:
        return values
        
    v_min = np.min(values)
    v_max = np.max(values)
    diff = v_max - v_min
    
    # 모든 값이 같거나 차이가 없는 경우 (Zero Division 방지)
    if diff < 1e-8:
        return np.zeros_like(values, dtype=float)
        
    # 벡터화된 Min-Max 정규화 연산
    return (values - v_min) / diff



def get_timestamp():
    """
    온라인 요청 추적을 위한 고유 타임스탬프 생성
    형식: YYYYMMDD_HHMMSS_밀리초 (예: 20260423_143154_123)
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

def truncate_words(text, max_words):
    """단어 단위로 텍스트를 제한"""
    if not text: return ""
    words = text.split()
    return " ".join(words[:max_words])