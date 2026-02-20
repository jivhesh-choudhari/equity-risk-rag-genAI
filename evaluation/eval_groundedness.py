import os, requests, re
APP=os.getenv('APP_BASE_URL','http://localhost:9060')
fid='HLSR-2024'
resp=requests.post(f'{APP}/summarize', json={'filing_id':fid}).json()
summary_text=' '.join(resp.get('highlights',[])+resp.get('risks',[]))
with open(os.path.join('corpus','filings',f'{fid}.md'),'r') as f: source=f.read()
tokens=re.findall(r'\w+',summary_text.lower()); hit=sum(1 for t in tokens if t in source.lower())
print(f'Groundedness proxy: {hit/len(tokens) if tokens else 0.0:.2f}')
