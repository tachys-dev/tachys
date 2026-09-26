project = "tachys"
# No trailing period: the theme footer renders "© Copyright {copyright}."
copyright = "2026, The Simons Foundation, Inc"
author = "Riccardo Rende"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# README.md holds build instructions for contributors, not site content.
exclude_patterns = ["_build", "README.md"]

# The documentation hub holds the master toctree; the site root (index.html) is a
# standalone animated landing page rendered from _templates/landing.html.
root_doc = "contents"

templates_path = ["_templates"]
# benchmarks.html is likewise standalone: it reads its whole dataset from
# _static/benchmarks-data.js at runtime, so new systems are a data-only edit.
# papers.html is standalone too; its list is plain HTML in the template.
html_additional_pages = {
    "index": "landing.html",
    "benchmarks": "benchmarks.html",
    "papers": "papers.html",
}

# The theme of the JAX and Flax docs: the left sidebar holds the logo, the
# search field and the whole site navigation, grouped by the captions of the
# toctrees in contents.md; there are no navigation links in a top bar.
html_theme = "sphinx_book_theme"
html_title = "tachys"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False

html_theme_options = {
    "logo": {
        "text": "tachys",
        "image_light": "_static/logo-mark.svg",
        "image_dark": "_static/logo-mark.svg",
    },
    "repository_url": "https://github.com/tachys-dev/tachys",
    "use_repository_button": True,
    "use_download_button": False,
    # Unless set, pydata-sphinx-theme adds a top bar holding a second search field.
    "navbar_persistent": [],
    "show_toc_level": 2,
    "footer_content_items": ["copyright.html", "sphinx-version.html"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}
