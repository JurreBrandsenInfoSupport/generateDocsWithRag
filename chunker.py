from tree_sitter import Node


def chunk_node(node: Node, text: str, max_chars: int = 1000) -> list:
    new_chunks = []
    current_chunk = ""
    for child in node.children:
        if child.end_byte - child.start_byte > max_chars:
            new_chunks.append(current_chunk)
            current_chunk = ""
            new_chunks.extend(chunk_node(child, text))
        elif len(current_chunk) + child.end_byte - child.start_byte > max_chars:
            new_chunks.append(current_chunk)
            current_chunk = text[child.start_byte : child.end_byte]
        else:
            current_chunk += text[child.start_byte : child.end_byte]
    if current_chunk:
        new_chunks.append(current_chunk)
    return new_chunks