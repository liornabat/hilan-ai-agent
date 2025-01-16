import os

import asyncio
from typing import List, Dict, Any

from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from openai import AsyncOpenAI
from supabase import create_client, Client

from document import Document

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

db_table = os.getenv("DB_TABLE", "hilan_docs")
docs_locations = os.getenv("DOCS_DIRECTORY", "./documents")


async def insert_document(doc: Document):
    """Insert a processed document into Supabase."""
    try:
        data = {
            "file_name": doc.file_name,
            "page": doc.page,
            "title": doc.title,
            "summary": doc.summary,
            "content": doc.content,
            "url": doc.url,
            "metadata": doc.metadata,
            "embedding": doc.embedding,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        result = supabase.table(db_table).insert(data).execute()
        print(f"Inserted document {doc.file_name}_{doc.page}")
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


async def process_file(file_path: Path) -> None:
    """Process a single JSON file and insert it into Supabase."""
    try:
        # Load the document using Document class
        doc = Document.load(str(file_path))

        # Insert into Supabase
        await insert_document(doc)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


async def process_files_parallel(files: List[Path], max_concurrent: int = 10):
    """Process multiple files in parallel with a concurrency limit."""
    try:
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(file: Path):
            async with semaphore:
                await process_file(file)

        # Process files in smaller batches
        batch_size = 10
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} files)")

            # Process batch with semaphore
            tasks = [process_with_semaphore(file) for file in batch]
            await asyncio.gather(*tasks)

            # Add a small delay between batches to avoid overwhelming the database
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")


async def main():
    try:
        # Get list of JSON files
        files = get_json_files(docs_locations)
        if not files:
            print("No JSON files found.")
            return

        print(f"Found {len(files)} JSON files to process")

        # Process files in parallel
        await process_files_parallel(files)

        print("Processing completed!")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())