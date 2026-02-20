import os, requests
APP=os.getenv('APP_BASE_URL','http://localhost:9060')
fid='HLSR-2024'
resp=requests.post(f'{APP}/summarize', json={'filing_id':fid}).json()
hi,risks,tone=resp.get('highlights',[]),resp.get('risks',[]),resp.get('tone','')
print(f'Coherence proxy passed={(len(hi)==2) and (1<=len(risks)<=3) and (tone in {"positive","neutral","cautious"})}')
