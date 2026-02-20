import sys, json, os, requests
APP=os.getenv('APP_BASE_URL','http://localhost:9060')
gold=json.load(open('evaluation/gold_labels.json'))
fid=sys.argv[1] if len(sys.argv)>1 else 'HLSR-2024'
resp=requests.post(f'{APP}/summarize', json={'filing_id':fid}).json()
pred=resp.get('tone','neutral'); gold_tone=gold.get(fid,{}).get('tone','neutral')
print(f'Tone predicted={pred} gold={gold_tone} agreement={int(pred==gold_tone)}')
