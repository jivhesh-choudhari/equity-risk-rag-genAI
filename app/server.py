from fastapi import FastAPI
from pydantic import BaseModel
from .summarize import summarize_filing
app=FastAPI(title='Financial Research Analyst Agent',version='0.1.0')
class SumReq(BaseModel): filing_id:str
@app.post('/summarize')
def summarize(req:SumReq):
    return summarize_filing(req.filing_id)
