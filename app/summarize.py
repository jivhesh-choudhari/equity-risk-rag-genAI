import os
from .loader import load_markdown_as_documents
from .chunker import chunk_documents
from .sentiment_risk import score_sentiment,label_from_scores,extract_risk_snippets
CORPUS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'corpus','filings')

def summarize_filing(filing_id:str):
    path=os.path.join(CORPUS_DIR,f'{filing_id}.md')
    if not os.path.exists(path): raise FileNotFoundError('Unknown filing_id: '+filing_id)
    docs=load_markdown_as_documents(path); chunks=chunk_documents(docs,800)
    hi=[]
    for ch in chunks:
        sec=ch['metadata'].get('section','')
        if sec in {'Business','Results'} and len(hi)<2:
            s=ch['page_content'].split('.')[0].strip()
            if s: hi.append(f"{s}. ({sec})")
    risks=[]
    for ch in extract_risk_snippets(chunks,3):
        sec=ch['metadata'].get('section',''); s=ch['page_content'].split('.')[0].strip()
        if s: risks.append(f"{s}. ({sec})")
    p=n=u=0
    for ch in chunks:
        sp,sn,su=score_sentiment(ch['page_content']); p+=sp; n+=sn; u+=su
    tone=label_from_scores(p,n,u)
    return {'filing_id':filing_id,'highlights':hi[:2],'risks':risks[:3],'tone':tone}
