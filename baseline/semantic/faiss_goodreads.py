import json
import os
import time
import argparse
from typing import Dict, Optional

_SKIP_SHELVES = {
    'to-read', 'currently-reading', 'owned', 'favorites', 'books',
    'read', 'default', 'kindle', 'library', 'my-library', 'owned-books',
    'ebook', 'e-book', 'audio', 'audiobook', 'wish-list', 'book-club',
    'reviewed', 'ocpl-hold', 'hp', 'summer-15', 'gr-authors',
    'read-goodreads-authors', 'goodreads-author', 'thebookfairy',
    'book-fairy', 'real-men-read-books', 'read-reading-libraries',
}


def build_item_text(item: dict, author_map: Dict[str, str] = None) -> str:
    parts = []

    # ── Title (boost 3×) ──────────────────────────────────────────────────
    title = (
        item.get('title_without_series', '').strip()
        or item.get('title', '').strip()
    )
    if title:
        # Lặp 3 lần để title có weight cao trong embedding
        parts.extend([title, title, title])

    # ── Authors ────────────────────
    if author_map:
        authors_raw = item.get('authors', [])
        author_names = []
        for a in authors_raw[:2]:   # max 2 authors
            if isinstance(a, dict):
                aid  = str(a.get('author_id', ''))
                role = a.get('role', '').strip()
                name = author_map.get(aid, '')
                if name and role.lower() not in {'illustrator', 'editor',
                                                 'translator', 'photographer'}:
                    author_names.append(name)
        if author_names:
            parts.append("by " + " and ".join(author_names))

    # ── Popular shelves → genres ──────────────────────────────────────────
    shelves_raw = item.get('popular_shelves', [])
    genre_names = []
    for shelf in shelves_raw:
        if isinstance(shelf, dict):
            name  = shelf.get('name', '').strip().lower()
            count = int(shelf.get('count', 0))
            if name and count >= 2 and name not in _SKIP_SHELVES:
                genre_names.append(name.replace('-', ' '))
        elif isinstance(shelf, str):
            name = shelf.strip().lower()
            if name not in _SKIP_SHELVES:
                genre_names.append(name.replace('-', ' '))

    if genre_names:
        parts.append("genres: " + ", ".join(genre_names[:8]))
    desc = item.get('description', '').strip()
    if desc:
        words = desc.split()
        if words:
            snippet = ' '.join(words[:120])
            parts.append(snippet)

    # ── Format ───────────────────────────────────────────────────────────
    fmt = item.get('format', '').strip()
    is_ebook = str(item.get('is_ebook', '')).lower()
    if is_ebook == 'true':
        parts.append("ebook")
    elif fmt and fmt.lower() not in {'', 'unknown'}:
        parts.append(fmt.lower())   # "hardcover", "paperback", etc.

    # ── Publication year ──────────────────────────────────────────────────
    pub_year = item.get('publication_year', '').strip()
    if pub_year and pub_year.isdigit() and 1800 <= int(pub_year) <= 2030:
        parts.append(f"published {pub_year}")

    # ── Rating ───────────────────────────────────────────────────────────
    rating = item.get('average_rating', '').strip()
    if rating:
        try:
            r = float(rating)
            if 0 < r <= 5:
                parts.append(f"rated {r:.1f} stars")
        except ValueError:
            pass

    # ── Fallback ──────────────────────────────────────────────────────────
    if not parts:
        return item.get('title', '') or str(item.get('item_id', 'unknown'))

    return " | ".join(filter(None, parts))


# ─────────────────────────────────────────────────────────────────────────────
# Load author map (optional)
# ─────────────────────────────────────────────────────────────────────────────

def load_author_map(author_file: str) -> Dict[str, str]:
    if not author_file or not os.path.exists(author_file):
        return {}

    author_map = {}
    with open(author_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
                aid  = str(a.get('author_id') or a.get('id') or '')
                name = a.get('name') or a.get('author_name') or ''
                if aid and name:
                    author_map[aid] = name.strip()
            except json.JSONDecodeError:
                pass

    print(f"[Build] Loaded {len(author_map)} authors")
    return author_map


# ─────────────────────────────────────────────────────────────────────────────
# Dry run
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(data_path: str, author_file: str = None, n: int = 5):
    author_map = load_author_map(author_file)

    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rich = build_item_text(item, author_map)

            print(f"\n--- Item {count+1} ---")
            print(f"  item_id : {item.get('item_id', '?')}")
            print(f"  title   : {item.get('title', '?')}")
            print(f"  n_shelves: {len(item.get('popular_shelves', []))}")
            print(f"  desc_len: {len(item.get('description', ''))} chars")
            print(f"\n  Rich text ({len(rich.split())} words):")
            # In từng phần để dễ đọc
            for part in rich.split(' | '):
                print(f"    • {part[:120]}")

            count += 1
            if count >= n:
                break


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save(
    data_path:   str,
    save_path:   str,
    embed_model: str,
    batch_size:  int,
    author_file: str = None,
):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    author_map = load_author_map(author_file)

    print(f"\n[Build] Loading embedding model: {embed_model}")
    embedding_function = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'batch_size': batch_size},
    )

    vector_store = None
    batch_count  = 0
    total_docs   = 0
    empty_rich   = 0
    start_time   = time.time()

    print(f"[Build] Reading {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        while True:
            # Đọc batch_size dòng
            batch_lines = []
            for _ in range(batch_size):
                line = f.readline()
                if not line:
                    break
                batch_lines.append(line)

            if not batch_lines:
                break

            batch_count  += 1
            documents_batch = []

            for line in batch_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rich_text = build_item_text(item, author_map)

                if not rich_text.strip():
                    empty_rich += 1
                    rich_text = item.get('title', '') or f"item_{item.get('item_id', '')}"

                doc = Document(
                    page_content=rich_text,
                    metadata={
                        'item_id':   str(item.get('item_id', '')),
                        'item_name': (
                            item.get('title_without_series', '')
                            or item.get('title', '')
                        ).strip(),
                    }
                )
                documents_batch.append(doc)

            if not documents_batch:
                continue

            total_docs += len(documents_batch)
            if vector_store is None:
                vector_store = FAISS.from_documents(
                    documents=documents_batch,
                    embedding=embedding_function,
                    distance_strategy="COSINE",
                )
            else:
                batch_vs = FAISS.from_documents(
                    documents=documents_batch,
                    embedding=embedding_function,
                    distance_strategy="COSINE",
                )
                vector_store.merge_from(batch_vs)

            elapsed = time.time() - start_time
            print(f"[Build] Batch {batch_count} | "
                  f"docs={total_docs} | "
                  f"elapsed={elapsed:.0f}s")

    if vector_store is None:
        print("[Build] ERROR: No documents built. Check data_path.")
        return

    elapsed = time.time() - start_time
    print(f"\n[Build] Done: {total_docs} docs | "
          f"{empty_rich} empty rich texts | "
          f"{elapsed:.0f}s total")

    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"[Build] Saved → {save_path}")
    print(f"[Build] Files: {os.listdir(save_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────

def verify_only(save_path: str, embed_model: str):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"\n[Verify] Loading index from {save_path}...")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vs = FAISS.load_local(
        save_path, embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy="COSINE",
    )

    queries = [
        "User read: The Name of the Wind | A Wise Man's Fear | The Slow Regard of Silent Things",
        "User read: Harry Potter and the Sorcerer's Stone | Eragon | The Hobbit",
        "User read: Gone Girl | The Girl on the Train | Big Little Lies",
        "User read: Sapiens | Thinking Fast and Slow | The Power of Habit",
        "fantasy magic adventure epic",
        "mystery thriller crime detective",
    ]

    print("\n" + "=" * 65)
    print("  VERIFY — Similarity search results")
    print("=" * 65)

    for query in queries:
        print(f"\nQuery: '{query[:70]}...'")
        results = vs.similarity_search(query, k=5)
        for j, doc in enumerate(results):
            name    = doc.metadata.get('item_name', '')
            item_id = doc.metadata.get('item_id', '')
            snippet = doc.page_content[:80]
            print(f"  {j+1}. [{item_id}] {name}")
            print(f"     {snippet}...")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Build rich content FAISS index for Goodreads"
    )
    p.add_argument('--data_path',   required=True,
                   help='Path to item_goodreads.json (JSONL)')
    p.add_argument('--save_path',   required=True,
                   help='Output directory for FAISS index')
    p.add_argument('--embed_model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--batch_size',  type=int, default=256)
    p.add_argument('--author_file', default=None,
                   help='Optional')
    p.add_argument('--dry_run',     action='store_true',
                   help='')
    p.add_argument('--n_dry',       type=int, default=5)
    p.add_argument('--verify_only', action='store_true',
                   help='')
    return p.parse_args()


def main():
    args = get_args()

    if args.dry_run:
        dry_run(args.data_path, args.author_file, args.n_dry)
        return

    if args.verify_only:
        verify_only(args.save_path, args.embed_model)
        return

    print("=" * 65)
    print("  Goodreads Rich Content FAISS Builder")
    print(f"  data_path  : {args.data_path}")
    print(f"  save_path  : {args.save_path}")
    print(f"  embed_model: {args.embed_model}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  author_file: {args.author_file or 'none (authors skipped)'}")
    print("=" * 65)

    build_and_save(
        data_path   = args.data_path,
        save_path   = args.save_path,
        embed_model = args.embed_model,
        batch_size  = args.batch_size,
        author_file = args.author_file,
    )

    print("\n[Build] Verifying index...")
    verify_only(args.save_path, args.embed_model)

    print("\n[Build] Update run script:")
    print(f"  --faiss_db_path {args.save_path}")


if __name__ == '__main__':
    main()