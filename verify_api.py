import requests
r = requests.post('http://127.0.0.1:5000/api/predict', json={'review_text':'Great product','rating':5,'user_id':'u1'}, timeout=10)
print(r.status_code)
print(r.text)
