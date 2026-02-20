import re
POS={'grew','growth','improve','higher','strong','profit','margin','cash','expects'}
NEG={'risk','breach','penalties','volatility','churn','decline','delay','uncertain','cautious','litigation'}
UNC={'may','could','might','uncertain','expects','cautious'}

def score_sentiment(text):
    toks=re.findall(r'\w+',text.lower()); p=sum(t in POS for t in toks); n=sum(t in NEG for t in toks); u=sum(t in UNC for t in toks); return p,n,u

def label_from_scores(p,n,u):
    if n>p and n>=u: return 'cautious'
    if p>n and p>=u: return 'positive'
    return 'neutral'

def extract_risk_snippets(chunks,k=3):
    scored=[]
    for ch in chunks:
        t=ch['page_content']
        w=sum(t.lower().count(wd) for wd in ['risk','breach','regulatory','volatility','churn','penalties'])
        if w>0: scored.append((w,ch))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [c for _,c in scored[:k]]
