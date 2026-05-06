"""
Inject <link rel="modulepreload"> hints for Gradio 6 component bundles.

Gradio 6 lazy-loads each component type's JS bundle via dynamic import().
Each bundle load triggers a Svelte reactive flush → Blocks.postprocess()
→ 5 full tree traversals via traverse(). With ~20 distinct component types,
this runs 20 times on initial page load before the UI is interactive.

Preloading all component bundles causes the browser to fetch them in parallel
before Svelte initializes, so all ensure() calls resolve synchronously. The
Svelte scheduler can then batch them into a single flush → postprocess() runs
once instead of ~20 times.

The patch modifies Gradio's index.html template (idempotent — skips if already
patched). The modification adds a Jinja2 loop over a preload_bundles global
that enumerates current .js files at serve time, so it survives Gradio upgrades
that rename bundles.
"""

_LOCALE_PREFIXES = (
    "ar-", "ca-", "ckb-", "de-", "es-", "eu-", "fa-", "fi-", "fr-",
    "he-", "hi-", "id-", "ja-", "ko-", "lt-", "nb-", "nl-", "pl-",
    "pt-", "ro-", "ru-", "sv-", "ta-", "th-", "tr-", "uk-", "ur-",
    "uz-", "zh-",
)

_JINJA_MARKER = "preload_component_bundles()"


def _find_manifest_bundles(assets_path):
    """
    Parse the i18n bundle manifest (i18n-*.js) to extract only the JS files
    that are actually loaded via dynamic import() — these are the ones that
    trigger ensure() calls and cascade into postprocess()+traverse().

    Falls back to all non-locale .js files if no manifest is found.
    """
    import re
    manifests = sorted(assets_path.glob("i18n-*.js"))
    if not manifests:
        return None

    src = manifests[0].read_text(encoding="utf-8", errors="replace")
    names = set(re.findall(r"""import\(["']\./([^"']+\.js)["']\)""", src))
    return names


def apply(real_gr):
    from pathlib import Path
    try:
        from gradio.routes import templates, STATIC_TEMPLATE_LIB
    except ImportError:
        return  # routes not yet initialised

    assets_path = Path(STATIC_TEMPLATE_LIB) / "frontend" / "assets"
    html_path = Path(STATIC_TEMPLATE_LIB) / "frontend" / "index.html"

    if not assets_path.exists() or not html_path.exists():
        return

    # Main entry point — already in index.html as <script type="module">
    _ENTRY = "index-DWyDLubb.js"

    # Parse once at patch time to build the candidate set
    manifest_names = _find_manifest_bundles(assets_path)

    def preload_component_bundles():
        """Return sorted list of component JS bundle filenames to preload."""
        result = []
        for f in sorted(assets_path.glob("*.js")):
            name = f.name
            if name == _ENTRY:
                continue
            if any(name.startswith(p) for p in _LOCALE_PREFIXES):
                continue
            # If we parsed a manifest, only preload bundles that are actually
            # dynamically imported (skip static dependency chunks).
            if manifest_names is not None and name not in manifest_names:
                continue
            result.append(name)
        return result

    templates.env.globals["preload_component_bundles"] = preload_component_bundles

    # Patch the HTML template file (idempotent)
    content = html_path.read_text(encoding="utf-8")
    if _JINJA_MARKER in content:
        return  # already patched

    preload_block = (
        "\t\t{% for _pf in preload_component_bundles() %}\n"
        '\t\t<link rel="modulepreload" as="script" crossorigin href="./assets/{{ _pf }}">\n'
        "\t\t{% endfor %}\n"
    )
    patched = content.replace(
        '\t\t<script type="module"',
        preload_block + '\t\t<script type="module"',
    )
    if patched == content:
        return  # insertion point not found — skip rather than corrupt

    html_path.write_text(patched, encoding="utf-8")
    bundle_count = len(preload_component_bundles())
    print(
        f"[gradio_compat] Injected modulepreload hints for {bundle_count} "
        "component bundles into Gradio HTML template"
    )
