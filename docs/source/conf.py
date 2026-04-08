# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "vllm-metal"
copyright = "2025, vLLM Team"
author = "vLLM Team"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_book_theme"
html_title = "vllm-metal"
html_theme_options = {
    "repository_url": "https://github.com/vllm-project/vllm-metal",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
}
