from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client
from typing import List
from openai import AsyncOpenAI
load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm )

logfire.configure(send_to_logfire='if-token-present')
db_table = os.getenv("DB_TABLE", "hilan_docs")
docs_locations = os.getenv("DOCS_DIRECTORY", "./parsed")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

@dataclass
class AIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
 אני מומחה בנושא טופס 101 ומדריכי משתמש. יש לי גישה למידע מקיף על 
 - מבנה הטופס ומטרתו 
 - הנחיות למילוי כל סעיף 
 - דגשים חשובים ונקודות לתשומת לב 
 - פתרון בעיות נפוצות 
 - עדכונים ושינויים בטופס

התפקיד שלי הוא:

1. לספק מידע מדויק ועדכני על טופס 101
2. להסביר בבהירות כל סעיף וסעיף
3. לענות על שאלות ספציפיות
4. לעזור בפתרון בעיות במילוי הטופס

הנחיות עבודה:

- אני אספק תשובות ישירות וברורות בעברית
- אשתמש בדוגמאות מעשיות כשנדרש
- אציין מקורות מידע רלוונטיים
- אעדכן על שינויים ועדכונים בטופס
- אודיע אם אין לי מידע מספק בנושא מסוים

אני מתמקד אך ורק בנושאים הקשורים לטופס 101 ולא אענה על שאלות שאינן קשורות לנושא זה.
אהיה הוגן ולא אמציא שום מידע שאינו נמצא בקבצים.
חשוב, תמיד אענה בעברית למרות שחלק מהמידע הוא באנגלית
"""

ai_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=AIDeps,
    retries=2,
    model_settings={"temperature": 0.3}
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 3072  # Return zero vector on error

@ai_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[AIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_docs',
            {
                'table_name': db_table,
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source':docs_locations}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
                # {doc['name']} - Chunk {doc['chunk_number']}
                
                ##  Summary:
                {doc['summary']}
                
                ## Content:
                    {doc['content']}
            """
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@ai_agent.tool
async def list_documentation_pages(ctx: RunContext[AIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique pages for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs

        result = ctx.deps.supabase.from_(db_table) \
            .select('name') \
            .eq('metadata->>source', docs_locations) \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['name'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@ai_agent.tool
async def get_page_content(ctx: RunContext[AIDeps], name: str) -> str:
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_(db_table) \
            .select('content, summary, chunk_number') \
            .eq('name', name) \
            .eq('metadata->>source', docs_locations) \
            .order('chunk_number') \
            .execute()

        if not result.data:
            return f"לא נמצא תוכן עבור: {name}"

        # Get the main title from the first chunk
        page_title = result.data[0][name].split(' - ')[0]

        # Start with document overview
        formatted_content = [
            f"# {page_title}",
            "\n## תקציר המסמך",
            "נקודות מפתח מכל חלק:"
        ]

        # Add summaries first as context
        summaries = []
        for chunk in result.data:
            if chunk['summary']:
                summaries.append(f"- חלק {chunk['chunk_number']}: {chunk['summary']}")

        if summaries:
            formatted_content.extend(summaries)
            formatted_content.append("\n## התוכן המלא")

        # Add the full content of each chunk
        for chunk in result.data:
            formatted_content.append(chunk['content'])

        # Join everything together with proper spacing
        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"שגיאה בשליפת תוכן העמוד: {str(e)}"