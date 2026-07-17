import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GraphMind Systems GraphRAG API (Groq Edition)")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Llama 3.3 70B is highly capable for reasoning and JSON instruction following
GROQ_MODEL = "llama-3.3-70b-versatile"

# ==========================================
# Pydantic Models for Validation
# ==========================================

# 1. Extract Endpoint Models
class ExtractRequest(BaseModel):
    chunk_id: str
    text: str

class Entity(BaseModel):
    name: str
    type: str = Field(description="Must be Person, Organization, Product, or Framework")

class Relationship(BaseModel):
    source: str
    target: str
    relation: str = Field(description="Must be FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, or AUTHORED")

class ExtractResponse(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

# 2. Graph Query Models
class GraphQueryRequest(BaseModel):
    question: str
    graph: Dict[str, Any]

class GraphQueryResponse(BaseModel):
    answer: str
    reasoning_path: List[str]
    hops: int

# 3. Community Summary Models
class CommunityRequest(BaseModel):
    community_id: str
    entities: List[str]
    relationships: List[Dict[str, Any]]

class CommunityResponse(BaseModel):
    community_id: str
    summary: str


# ==========================================
# Endpoints
# ==========================================

@app.post("/extract-graph", response_model=ExtractResponse)
async def extract_graph(request: ExtractRequest):
    """Extracts entities and relationships from unstructured text."""
    schema_str = json.dumps(ExtractResponse.model_json_schema())
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a precise knowledge graph extraction system. Extract entities and relationships from the text based on the strictly provided types. Return ONLY valid JSON matching this exact schema:\n{schema_str}"
                },
                {"role": "user", "content": request.text}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        # Parse the JSON string directly back into the Pydantic model
        return ExtractResponse.model_validate_json(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph-query", response_model=GraphQueryResponse)
async def graph_query(request: GraphQueryRequest):
    """Answers a multi-hop question using the provided graph context."""
    schema_str = json.dumps(GraphQueryResponse.model_json_schema())
    system_prompt = f"""
    You are a GraphRAG reasoning agent. Use ONLY the provided knowledge graph to answer the question.
    Graph Context: {json.dumps(request.graph)}
    
    Determine the shortest path of entities used to answer the question. Count the number of relationships traversed as 'hops'.
    Return ONLY valid JSON matching this exact schema:
    {schema_str}
    """
    
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return GraphQueryResponse.model_validate_json(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/community-summary", response_model=CommunityResponse)
async def community_summary(request: CommunityRequest):
    """Generates a summary of a specific sub-community."""
    system_prompt = f"""
    Summarize the following connected community in the knowledge graph. 
    Focus on the main entities and how they interrelate. 
    Keep the summary concise but informative (1-3 sentences).
    
    Entities: {request.entities}
    Relationships: {json.dumps(request.relationships)}
    """
    
    try:
        # Standard text completion is safer and cheaper for summary generation
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt}
            ],
            temperature=0.3,
        )
        summary = completion.choices[0].message.content.strip()
        return CommunityResponse(community_id=request.community_id, summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
    return {"status": "Active", "message": "GraphMind Systems GraphRAG API (Groq) is running."}
