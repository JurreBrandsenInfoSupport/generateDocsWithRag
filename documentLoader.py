from dataclasses import dataclass
from typing import List, Tuple
import json
from pathlib import Path
from os import path
from importlib import import_module
from tree_sitter import Language, Parser, Tree

@dataclass
class LanguageDefinition:
    name: str
    package: str
    extensions: List[str]


def load_language_definitions():
    languages_path = Path(__file__).parent / "languages.json"

    with open(languages_path, "r") as f:
        data = json.load(f)
        return [LanguageDefinition(**d) for d in data]


def detect_language(filename: str) -> Tuple[Language, str]:
    for language in load_language_definitions():
        name, extension = path.splitext(filename)

        if any(extension == ext for ext in language.extensions):
            language_pack = import_module(language.package)
            language_spec = Language(language_pack.language())

            return language_spec, language.name

    return None, None


def parse_tree(filename: str, content: str) -> Tuple[Tree, str]:
    language, language_name = detect_language(filename)

    if language is None:
        raise ValueError(f"Language not found for file {filename}")

    parser = Parser(language)
    tree = parser.parse(bytes(content, "utf-8"))

    return tree, language_name