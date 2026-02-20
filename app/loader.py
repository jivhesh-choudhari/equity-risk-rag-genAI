from typing import List, Dict
import re

def load_markdown_as_documents(path:str)->List[dict]:
    with open(path,'r') as f: text=f.read()
    parts=re.split(r'\n(?=## )', text)
    title=parts[0].splitlines()[0].replace('# ','')
    docs=[]; current='Preamble'; buf=[]
    for p in parts:
        if p.startswith('## '):
            if buf:
                docs.append({'page_content':'\n'.join(buf).strip(),'metadata':{'title':title,'section':current}})
            lines=p.splitlines(); current=lines[0].replace('## ','').strip(); buf=lines[1:]
        else:
            buf.extend(p.splitlines()[1:])
    if buf:
        docs.append({'page_content':'\n'.join(buf).strip(),'metadata':{'title':title,'section':current}})
    return docs
