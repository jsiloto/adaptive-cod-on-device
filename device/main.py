import requests
with open('./resource/000000000161.jpg', 'rb') as f:
    data = f.read()

res = requests.post(url='http://0.0.0.0:5000/remote',
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'})
