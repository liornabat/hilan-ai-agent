import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

db_table = os.getenv("DB_TABLE", "hilan_docs")
docs_locations = os.getenv("DOCS_DIRECTORY", "./parsed")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

@dataclass
class ProcessedDocument:
    name: str
    chunk_number: int
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

async def get_embedding(text: str) -> List[float]:
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

async def process_document(file_path: Path, max_retries: int = 3) -> ProcessedDocument:
    """Process a single JSON document with retries."""
    for attempt in range(max_retries):
        try:
            # Read and parse JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract required fields
            content = data.get('text', '')
            # Get summary from first entity of type 'summary' if available
            summary = next(
                (entity['text'] for entity in data.get('entities', [])
                 if entity.get('type') == 'summary'),
                ''  # Default empty string if no summary found
            )

            # Create metadata
            metadata = {
                "source": docs_locations,
                "original_file_path": data.get('file_path', ''),
                "pages": data.get('pages', 1),
                "is_error": data.get('is_error', False),
                "error_message": data.get('error_message'),
                "processed_at": datetime.now(timezone.utc).isoformat()
            }

            # Get embedding for the content
            embedding = await get_embedding(content)

            return ProcessedDocument(
                name=str(file_path),
                chunk_number=0,  # Since we're processing whole documents
                content=content,
                summary=summary,
                metadata=metadata,
                embedding=embedding
            )
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to process document after {max_retries} attempts: {str(e)}")
                raise
            await asyncio.sleep(1)  # Wait before retry

async def insert_document(doc: ProcessedDocument):
    """Insert a processed document into Supabase."""
    try:
        data = {
            "name": doc.name,
            "chunk_number": doc.chunk_number,
            "summary": doc.summary,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": doc.embedding
        }

        result = supabase.table(db_table).insert(data).execute()
        print(f"Inserted document {doc.name}")
        return result
    except Exception as e:
        print(f"Error inserting document: {e}")
        return None

def get_json_files(directory: str) -> List[Path]:
    """Get all JSON files from the specified directory and its subdirectories."""
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"Directory not found: {directory}")
        return []

    json_files = []
    for file_path in directory_path.rglob("*.json"):
        if file_path.is_file():
            json_files.append(file_path)

    return json_files

async def get_unprocessed_files(files: List[Path]) -> List[Path]:
    """Filter out files that have already been processed in the database."""
    try:
        # Query Supabase for existing file paths
        response = supabase.table(db_table) \
            .select("name") \
            .execute()

        # Extract unique file paths from the response
        processed_files = set(item['name'] for item in response.data)

        # Filter out files that are already in the database
        unprocessed_files = [file for file in files if str(file) not in processed_files]

        print(f"Total files: {len(files)}")
        print(f"Already processed: {len(processed_files)}")
        print(f"New files to process: {len(unprocessed_files)}")

        return unprocessed_files
    except Exception as e:
        print(f"Error checking processed files: {str(e)}")
        return files  # Return all files if there's an error

async def process_files_parallel(files: List[Path], max_concurrent: int = 5):
    """Process multiple files in parallel with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_file(file_path: Path):
        try:
            async with semaphore:
                print(f"Processing file: {file_path}")
                doc = await process_document(file_path)
                await insert_document(doc)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    try:
        # Process files in smaller batches
        batch_size = 10
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} files)")
            await asyncio.gather(*[process_file(file) for file in batch])

            # Add a small delay between batches
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")

async def main():
    try:
        # Get all JSON files
        json_files = get_json_files(docs_locations)
        if not json_files:
            print("No JSON files found")
            return

        print(f"Found {len(json_files)} JSON files")

        # Filter out already processed files
        unprocessed_files = await get_unprocessed_files(json_files)
        if not unprocessed_files:
            print("All files have already been processed")
            return

        # Process files in parallel
        await process_files_parallel(unprocessed_files, max_concurrent=10)

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())