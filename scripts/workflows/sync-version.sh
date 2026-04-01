#!/usr/bin/env bash
# Sync version from packages/models/lactationcurve/pyproject.toml to citation files.
# Called by semantic-release via build_command.
set -euo pipefail

# Cross-platform sed in-place (macOS requires '' arg, Linux does not)
sedi() {
    if [[ "$OSTYPE" == darwin* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

PKG="packages/models/lactationcurve/pyproject.toml"
README="packages/models/lactationcurve/README.md"
INIT="packages/models/lactationcurve/src/lactationcurve/__init__.py"
CITATION="CITATION.cff"

# Verify all required files exist
for f in "$PKG" "$README" "$INIT" "$CITATION"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

# Extract version from pyproject.toml
NEW=$(grep '^version' "$PKG" | head -1 | sed 's/version = "\(.*\)"/\1/')
if [ -z "$NEW" ]; then
    echo "ERROR: could not extract version from $PKG"
    exit 1
fi
TODAY=$(date +%Y-%m-%d)
echo "Syncing citation version to $NEW (updated $TODAY)..."

# Pattern: version number in the form X.Y.Z (e.g. 0.1.0, 1.2.3)
V='[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*'

# CITATION.cff — both top-level and indented version fields
sedi "s|version: \"$V\"|version: \"$NEW\"|" "$CITATION"
sedi "s|lactationcurve: v\.$V|lactationcurve: v.$NEW|g" "$CITATION"

# README.md — inline citation, BibTeX title (with escaped underscores), and BibTeX version
sedi "s|lactationcurve: v\.$V|lactationcurve: v.$NEW|" "$README"
sedi "s|lactationcurve: v\.$V|lactationcurve: v.$NEW|" "$README"
sedi "s|(v\.$V)|(v.$NEW)|" "$README"
sedi "s|version      = {$V}|version      = {$NEW}|" "$README"

# __init__.py docstring — citation line and "Current version" section
sedi "s|v\.$V\. (v\.$V)|v.$NEW. (v.$NEW)|" "$INIT"
D='[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
sedi "s#\*\*Version:\*\* v\.$V | \*\*Updated:\*\* $D#**Version:** v.$NEW | **Updated:** $TODAY#" "$INIT"

# Stage the updated files so semantic-release includes them in the release commit
git add "$CITATION" "$README" "$INIT"

echo "Done — synced CITATION.cff, README.md, and __init__.py to v$NEW"
