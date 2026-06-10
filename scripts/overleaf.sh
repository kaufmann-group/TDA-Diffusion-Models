#!/bin/bash

# Script pulls LaTeX files from Overleaf, compiles with pdfLatex, pushes .tex files, respective .pdf's and figures to remote under notes/

set -e

REPO=$(git rev-parse --show-toplevel)
TMPDIR=$(mktemp -d)
BRANCH_NAME="tmp-overleaf-sync-$(date +%s)"

cd "$REPO"

trap 'rm -rf "$TMPDIR"; git -C "$REPO" worktree prune >/dev/null 2>&1; git -C "$REPO" branch -D "$BRANCH_NAME" >/dev/null 2>&1' EXIT

git -C "$REPO" fetch origin main
git -C "$REPO" fetch overleaf master

git -C "$REPO" checkout -b "$BRANCH_NAME" origin/main
git -C "$REPO" worktree add "$TMPDIR" "$BRANCH_NAME"

rm -rf "$TMPDIR/notes"
mkdir -p "$TMPDIR/notes/pdfs"

git -C "$REPO" archive overleaf/master | tar -x -C "$TMPDIR/notes"

cd "$TMPDIR/notes"

for file in "Notes & Helpful Files"/*.tex; do
    [ -e "$file" ] || continue

    if grep -Fxq '% git ignore this file' "$file"; then
        echo "Ignoring $file"
        rm -f "$file"
        continue
    fi

    if grep -q '\\documentclass' "$file"; then
        echo "Compiling $file..."
        
        base_name=$(basename "$file")
        
        pdflatex -shell-escape -interaction=nonstopmode -halt-on-error "$file"
        
        if [ -f "${base_name%.tex}.pdf" ]; then
            mv "${base_name%.tex}.pdf" pdfs/
        fi
        
        rm -rf _minted-${base_name%.tex}
        rm -f "${base_name%.tex}".{aux,log,toc,out,snm,nav,vrb,fls,fdb_latexmk}
    else
        echo "Skipping non-document file: $file"
    fi
done

cd "$TMPDIR"

git add notes

if git diff --cached --quiet; then
    echo "No changes to commit."
else
    git commit -m "sync notes and PDFs from Overleaf"
    git push origin HEAD:main
fi