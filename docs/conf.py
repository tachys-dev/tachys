project = "tachys"
copyright = "2024, Riccardo Rende"
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
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "pydata_sphinx_theme"
html_title = "tachys"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False

html_theme_options = {
    "logo": {"text": "tachys"},
    "github_url": "https://github.com/riccardo-rende/tachys",
    "navbar_align": "left",
    "show_nav_level": 2,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "navbar_end": ["navbar-icon-links"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}
