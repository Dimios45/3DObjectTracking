import requests, sys, time, os

BASE  = 'http://medialabnas.synology.me:5000'
SHARE = 'MHa7WjzLK'
FILE  = 'OPT424p_3D.zip'
OUT   = '/mnt/data/mritunjoyh/datasets/opt/opt_3d.zip'

os.makedirs(os.path.dirname(OUT), exist_ok=True)

s = requests.Session()
s.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0'})

print('Logging in...', flush=True)
r = s.post(BASE + '/webapi/entry.cgi', data={
    'api': 'SYNO.Core.Sharing.Login', 'version': '1',
    'method': 'login', 'sharing_id': SHARE, 'password': ''
}, timeout=10)
print('Login:', r.json(), flush=True)

print('Starting download...', flush=True)
r = s.get(f'{BASE}/fsdownload/{SHARE}/{FILE}', stream=True, timeout=60)
total = int(r.headers.get('content-length', 0))
print(f'Total size: {total/1024/1024/1024:.2f} GB', flush=True)

downloaded = 0
t0 = time.time()
with open(OUT, 'wb') as f:
    for chunk in r.iter_content(chunk_size=4*1024*1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed = downloaded / elapsed / 1024 / 1024
            pct = downloaded / total * 100 if total else 0
            print(f'{pct:.1f}%  {downloaded/1024/1024/1024:.2f}/{total/1024/1024/1024:.2f} GB  {speed:.1f} MB/s', flush=True)

print(f'Done: {OUT}', flush=True)
