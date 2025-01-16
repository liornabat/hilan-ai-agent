from pathlib import Path

from typing import Optional

from pydantic import BaseModel, Field

class Document(BaseModel):
    file_name: str = Field(..., description="Name of the document")
    page: int = Field(..., description="Page number of the document")
    title: Optional[str] = Field(..., description="Title of the document")
    summary: Optional[str] = Field(..., description="Summary of the document")
    content: Optional[str] = Field(..., description="Content of the document")
    url: Optional[str] = Field(..., description="URL of the document")
    metadata: Optional[dict] = Field(default_factory=dict, description="Metadata of the document")
    embedding: Optional[list[float]] = Field(default_factory=list, description="Embedding of the document")

    def save(self, output_path: str) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        json_filename = f"{self.file_name}_{self.page}.json"
        file_path = Path(output_path) / json_filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, file_path: str) -> 'Document':
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            return cls.model_validate_json(f.read())

    def to_markdown(self) -> str:
        """
        Convert the document to markdown format.
        Only includes fields that have values (not None or empty collections).

        Returns:
            str: Markdown formatted string
        """
        markdown_parts = []
        data = self.model_dump()

        for field, value in data.items():
            # Skip empty values
            if value is None:
                continue
            if isinstance(value, (list, dict)) and not value:
                continue

            # Convert field name to title case for header
            header = field.replace('_', ' ').title()
            markdown_parts.append(f"# {header}\n")

            # Format the value based on its type
            if isinstance(value, dict):
                formatted_value = "\n".join(f"- {k}: {v}" for k, v in value.items())
            elif isinstance(value, list):
                if all(isinstance(x, float) for x in value):  # Handle embeddings
                    formatted_value = f"Vector with {len(value)} dimensions"
                else:
                    formatted_value = "\n".join(f"- {item}" for item in value)
            else:
                formatted_value = str(value)

            markdown_parts.append(f"{formatted_value}\n\n")

        return "".join(markdown_parts)

    def save_to_markdown(self, output_path: str) -> None:
        """
        Save the document as a markdown file.

        Args:
            output_path (str): Directory path where the markdown file will be saved
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        markdown_filename = f"{self.file_name}_{self.page}.md"
        file_path = Path(output_path) / markdown_filename

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_markdown())