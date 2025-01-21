from typing import List
import os
import json
from dotenv import load_dotenv
import asyncio
import aiofiles
import logging
from pathlib import Path
from openai import AsyncOpenAI
import tiktoken

from document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
client = AsyncOpenAI()
encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    """
    return len(encoding.encode(text))


def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks based on token count rather than character count.

    Args:
        text (str): Text to split
        chunk_size (int): Target size for each chunk in tokens

    Returns:
        List[str]: List of text chunks
    """
    if not text:
        return []

    # Split text into sentences
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Add period back if it was removed during split
        sentence = sentence + '.'
        sentence_tokens = count_tokens(sentence)

        # If this single sentence is already too large, split it into words
        if sentence_tokens > chunk_size:
            # If we have a current chunk, add it to chunks first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split long sentence into words and create chunks
            words = sentence.split()
            word_chunk = []
            word_chunk_tokens = 0

            for word in words:
                word_tokens = count_tokens(word + ' ')
                if word_chunk_tokens + word_tokens > chunk_size:
                    if word_chunk:
                        chunks.append(' '.join(word_chunk))
                    word_chunk = [word]
                    word_chunk_tokens = word_tokens
                else:
                    word_chunk.append(word)
                    word_chunk_tokens += word_tokens

            if word_chunk:
                chunks.append(' '.join(word_chunk))
            continue

        # Check if adding this sentence would exceed the chunk size
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Verify no chunk exceeds the token limit (for debugging)
    for i, chunk in enumerate(chunks):
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens > chunk_size:
            logger.warning(f"Chunk {i} still has {chunk_tokens} tokens despite splitting!")

    return chunks


class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.total_files = 0
        self.lock = asyncio.Lock()

    async def add_tokens(self, tokens: int):
        async with self.lock:
            self.total_tokens += tokens
            self.total_files += 1


token_counter = TokenCounter()


async def get_embedding(text: str) -> List[float]:
    """
    Get embedding for the text using OpenAI's API.
    """
    try:
        response = await client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise


async def process_file(file_path: Path, output_directory: Path, markdown_directory: Path, chunk_size: int) -> None:
    """
    Process a single JSON file, split its content into chunks, and create Document objects.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

        content_chunks = split_text_into_chunks(data.get('content', ''), chunk_size)

        if not content_chunks:
            content_chunks = ['']

        for chunk_idx, chunk_content in enumerate(content_chunks, 1):
            base_filename = Path(data['file_name']).stem

            doc = Document(
                file_name=f"{base_filename}_chunk_{chunk_idx}",
                page=chunk_idx,
                title=data.get('title'),
                summary='',
                content=chunk_content,
                url=data.get('url'),
                metadata={
                    "original_filename": data['file_name'],
                    "total_chunks": len(content_chunks),
                    "chunk_size": chunk_size,
                    "tokens": count_tokens(chunk_content)  # Add token count to metadata
                },
                embedding=[]
            )

            markdown_content = doc.to_markdown()
            doc.embedding = await get_embedding(markdown_content)

            # Count tokens for this chunk
            chunk_tokens = count_tokens(chunk_content)
            if chunk_tokens > 8192:
                logger.warning(f"---------------- Chunk {chunk_idx} has {chunk_tokens} tokens")

            await token_counter.add_tokens(chunk_tokens)

            doc.save(str(output_directory))
            doc.save_to_markdown(str(markdown_directory))

        logger.info(f"Successfully processed {file_path.name} into {len(content_chunks)} chunks")

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")


async def process_directory(
        input_directory: str,
        output_directory: str,
        markdown_directory: str,
        chunk_size: int = 1000,
        batch_size: int = 10
) -> None:
    """
    Process all JSON files in the specified directory with parallel execution.
    """
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    markdown_path = Path(markdown_directory)

    json_files = list(input_path.glob('**/*.json'))

    if not json_files:
        logger.warning(f"No JSON files found in {input_directory}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    markdown_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(json_files), batch_size):
        batch = json_files[i:i + batch_size]
        tasks = [process_file(file, output_path, markdown_path, chunk_size) for file in batch]
        await asyncio.gather(*tasks)

        logger.info(f"Processed batch {i // batch_size + 1} of {(len(json_files) + batch_size - 1) // batch_size}")

    # Log final statistics
    logger.info(f"Total files processed: {token_counter.total_files}")
    logger.info(f"Total tokens: {token_counter.total_tokens}")
    logger.info(f"Average tokens per chunk: {token_counter.total_tokens / token_counter.total_files:.2f}")


def main():
    """
    Main entry point of the script.
    """
    input_directory = "./documents_kol_zchut"
    output_directory = "./documents_kol_zchut_json"
    markdown_directory = "./documents_kol_zchut_md"

    chunk_size = 5000  # Characters per chunk

    asyncio.run(process_directory(
        input_directory=input_directory,
        output_directory=output_directory,
        markdown_directory=markdown_directory,
        chunk_size=chunk_size
    ))

    logger.info("Processing completed!")


if __name__ == "__main__":
    main()