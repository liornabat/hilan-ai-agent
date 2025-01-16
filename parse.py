from typing import List
import os
import json
from dotenv import load_dotenv
import asyncio
import aiofiles
import re
from pathlib import Path
from openai import AsyncOpenAI
import logging

from document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
client = AsyncOpenAI()


async def get_embedding(text: str) -> List[float]:
    """
    Get embedding for the text using OpenAI's API.
    """
    try:
        response = await client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise


async def translate_text(text: str) -> str:
    """
    Translate English markdown text to Hebrew using OpenAI API.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the following English markdown text to Hebrew. Maintain all markdown formatting in the translation."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise


def extract_file_info(file_path: Path) -> tuple[str, int]:
    """
    Extract file name and page number from the file path.
    """
    file_name = file_path.stem
    match = re.search(r'(.+)_(\d+)$', file_name)
    if match:
        base_name = match.group(1)
        page_num = int(match.group(2))
        return base_name, page_num
    else:
        return file_name, 0


async def process_file(file_path: Path, output_directory: Path, markdown_directory: Path) -> None:
    """
    Process a single JSON file and create a Document object.
    """
    try:
        # Read JSON file
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

        # Extract file name and page number
        file_name, page_number = extract_file_info(file_path)

        # Extract English text from entities and translate
        english_text = ""
        if data.get('entities') and len(data['entities']) > 0:
            english_text = data['entities'][0].get('text', '')

        if english_text:
            # Translate the text
            hebrew_translation = await translate_text(english_text)
        else:
            hebrew_translation = None
            logger.warning(f"No text to translate in {file_path.name}")

        # Create Document object
        doc = Document(
            file_name=file_name,
            page=page_number,
            title=None,
            summary=hebrew_translation,
            content=data.get('text', ''),
            url=None,
            metadata={},
            embedding=[]  # Initialize empty embedding
        )

        # Generate markdown content and get its embedding
        markdown_content = doc.to_markdown()
        doc.embedding = await get_embedding(markdown_content)

        # Save both JSON and markdown versions
        doc.save(str(output_directory))
        doc.save_to_markdown(str(markdown_directory))

        logger.info(f"Successfully processed {file_path.name}")

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")


async def process_directory(
        input_directory: str,
        output_directory: str,
        markdown_directory: str,
        batch_size: int = 10
) -> None:
    """
    Process all JSON files in the specified directory with parallel execution.
    """
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    markdown_path = Path(markdown_directory)

    # Get all JSON files in the directory
    json_files = list(input_path.glob('**/*.json'))

    if not json_files:
        logger.warning(f"No JSON files found in {input_directory}")
        return

    # Create output directories if they don't exist
    output_path.mkdir(parents=True, exist_ok=True)
    markdown_path.mkdir(parents=True, exist_ok=True)

    # Process files in batches
    for i in range(0, len(json_files), batch_size):
        batch = json_files[i:i + batch_size]
        tasks = [process_file(file, output_path, markdown_path) for file in batch]
        await asyncio.gather(*tasks)

        logger.info(f"Processed batch {i // batch_size + 1} of {(len(json_files) + batch_size - 1) // batch_size}")


def main():
    """
    Main entry point of the script.
    """
    # Directory paths
    input_directory = "./parsed"
    output_directory = "./documents"
    markdown_directory = "./documents"

    # Create event loop and run the async process
    asyncio.run(process_directory(input_directory, output_directory, markdown_directory))

    logger.info("Processing completed!")


if __name__ == "__main__":
    main()