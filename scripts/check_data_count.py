import os
import sys
import socket
import psycopg2
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

raw_url = os.getenv("DATABASE_URL")

# Try to connect with retry
conn = None
for attempt in range(5):
    try:
        conn = psycopg2.connect(raw_url, connect_timeout=10)
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed ({e}), retrying...")
        import time
        time.sleep(1)

if not conn:
    print("Could not connect to Supabase.")
    sys.exit(1)

cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
tables = [r[0] for r in cur.fetchall()]

print("=" * 65)
print("DANH SÁCH TẤT CẢ CÁC BẢNG VÀ SỐ DÒNG DỮ LIỆU TRÊN SUPABASE:")
print("=" * 65)
for t in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{t}";')
        cnt = cur.fetchone()[0]
        print(f"  {t:<35}: {cnt:>6} dòng")
    except Exception as e:
        print(f"  {t:<35}: Lỗi ({e})")
        conn.rollback()

conn.close()
