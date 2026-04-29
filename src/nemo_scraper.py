import requests
import sqlite3
import os
import json
import time

def scrape_nemo():
    # 1) API 요청 정보 설정
    url = "https://www.nemoapp.kr/api/store/search-list"
    
    # 1) HTTP 헤더 설정
    headers = {
        "referer": "https://www.nemoapp.kr/store",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    }

    # 4) SQLite DB 저장 설정
    db_path = os.path.join("data", "nemo_data.db")
    os.makedirs("data", exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 테이블 생성 (주요 컬럼 위주)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stores (
        id TEXT PRIMARY KEY,
        number INTEGER,
        title TEXT,
        businessLargeCodeName TEXT,
        businessMiddleCodeName TEXT,
        priceTypeName TEXT,
        deposit INTEGER,
        monthlyRent INTEGER,
        premium INTEGER,
        maintenanceFee INTEGER,
        floor TEXT,
        size REAL,
        nearSubwayStation TEXT,
        viewCount INTEGER,
        favoriteCount INTEGER,
        raw_json TEXT
    )
    """)

    page_index = 0
    total_processed = 0

    while True:
        # 2) Payload 정보 (Query Parameters)
        params = {
            "CompletedOnly": "false",
            "NELat": "37.507252311764425",
            "NELng": "127.04528546334437",
            "SWLat": "37.48707710832999",
            "SWLng": "127.01777462563194",
            "Zoom": "15",
            "SortBy": "29",
            "PageIndex": str(page_index),
            "Subway": "222",
            "Radius": "1000"
        }

        print(f"Scraping PageIndex: {page_index}...")
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
        except Exception as e:
            print(f"HTTP 요청 실패 (PageIndex {page_index}): {e}")
            break

        data = response.json()
        items = data.get("items", [])
        
        # 3) 응답이 없을 때까지 (items가 비어있을 때까지) 수행
        if not items:
            print(f"PageIndex {page_index}에서 더 이상 수집할 데이터가 없습니다.")
            break

        print(f"수집된 아이템 수: {len(items)}")

        for item in items:
            cursor.execute("""
            INSERT OR REPLACE INTO stores (
                id, number, title, businessLargeCodeName, businessMiddleCodeName,
                priceTypeName, deposit, monthlyRent, premium, maintenanceFee,
                floor, size, nearSubwayStation, viewCount, favoriteCount, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.get("id"),
                item.get("number"),
                item.get("title"),
                item.get("businessLargeCodeName"),
                item.get("businessMiddleCodeName"),
                item.get("priceTypeName"),
                item.get("deposit"),
                item.get("monthlyRent"),
                item.get("premium"),
                item.get("maintenanceFee"),
                str(item.get("floor")),
                item.get("size"),
                item.get("nearSubwayStation"),
                item.get("viewCount"),
                item.get("favoriteCount"),
                json.dumps(item, ensure_ascii=False)
            ))

        total_processed += len(items)
        conn.commit()
        
        page_index += 1
        time.sleep(0.5) # 서버 부하 방지 및 차단 예방을 위한 대기

    conn.close()
    print(f"\n전체 수집 완료.")
    print(f"- 처리된 총 페이지 수: {page_index}")
    print(f"- 이번 세션에서 처리된 아이템 수: {total_processed}")
    print(f"- 데이터베이스 경로: {db_path}")

if __name__ == "__main__":
    scrape_nemo()
