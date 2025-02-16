from tree_sitter import Node

def chunk_node(node: Node, text: str, max_chars: int = 1000, debug: bool = False) -> list:
    """
    Recursively extracts meaningful code chunks from a syntax tree node.
    Ignores small/uninformative chunks.
    """
    new_chunks = []
    current_chunk = ""

    for child in node.children:
        chunk_text = text[child.start_byte : child.end_byte].strip()

        if len(chunk_text) > max_chars:
            new_chunks.extend(chunk_node(child, text, max_chars, debug))
        elif len(current_chunk) + len(chunk_text) > max_chars:
            new_chunks.append(current_chunk)
            current_chunk = chunk_text
        else:
            current_chunk += "\n" + chunk_text

    if debug:
        # Open in append mode instead of overwrite mode
        with open("chunks_debug.txt", "a", encoding="utf-8") as f:
            f.write("\n===== New File Processed =====\n\n")
            for i, chunk in enumerate(new_chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk)
                f.write("\n\n")

    return new_chunks
