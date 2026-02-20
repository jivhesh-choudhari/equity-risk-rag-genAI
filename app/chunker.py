def chunk_documents(docs, max_chars=900):
    out=[]
    for d in docs:
        t=d['page_content']; m=d.get('metadata',{})
        if len(t)<=max_chars: out.append(d)
        else:
            for i in range(0,len(t),max_chars):
                out.append({'page_content':t[i:i+max_chars],'metadata':{**m,'chunk_index':i//max_chars}})
    return out
