"""
Offline index builder. Run once before starting the server.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --corpus data/corpus/ --output data/indexes/
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.data_loader import load_all_protocols, build_icd_lookup
from src.indexer import (
    build_chunks_from_protocol,
    build_tree_from_protocol,
    treenode_to_dict,
    lemmatize,
    TREES_DIR,
    INDEX_DIR,
)


def main():
    parser = argparse.ArgumentParser(description="Build BM25 + FAISS + tree indexes")
    parser.add_argument(
        "--corpus",
        default="data/corpus/",
        help="Path to corpus directory or JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/indexes/",
        help="Output directory for indexes",
    )
    parser.add_argument(
        "--trees-dir",
        default="data/trees/",
        help="Output directory for tree JSON files",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Override EMBED_MODEL env var for this run only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if indexes already exist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    trees_dir = Path(args.trees_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = output_dir / "bm25.pkl"
    faiss_path = output_dir / "faiss.index"
    chunks_path = output_dir / "chunks.pkl"

    if not args.force and bm25_path.exists() and faiss_path.exists() and chunks_path.exists():
        print("Indexes already exist. Use --force to rebuild.")
        return

    # -----------------------------------------------------------------------
    # 1. Load protocols
    # -----------------------------------------------------------------------
    print(f"Loading protocols from {args.corpus}...")
    protocols = load_all_protocols(args.corpus)
    print(f"Loaded {len(protocols)} protocols")

    # -----------------------------------------------------------------------
    # 2. Build tree structures
    # -----------------------------------------------------------------------
    print("Building tree structures...")
    for i, (pid, protocol) in enumerate(protocols.items()):
        tree_path = trees_dir / f"{pid}.json"
        if tree_path.exists() and not args.force:
            continue
        tree = build_tree_from_protocol(protocol)
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(treenode_to_dict(tree), f, ensure_ascii=False, indent=2)
        if (i + 1) % 100 == 0:
            print(f"  Trees: {i + 1}/{len(protocols)}")
    print(f"Trees saved to {trees_dir}")

    # -----------------------------------------------------------------------
    # 3. Build chunks
    # -----------------------------------------------------------------------
    print("Building chunks...")
    all_chunks = []
    for protocol in protocols.values():
        all_chunks.extend(build_chunks_from_protocol(protocol))
    print(f"Total chunks: {len(all_chunks)}")

    # -----------------------------------------------------------------------
    # 4. Tokenize for BM25
    # -----------------------------------------------------------------------
    print("Lemmatizing chunks for BM25...")
    tokenized = []
    for i, chunk in enumerate(all_chunks):
        tokenized.append(lemmatize(chunk.text))
        if (i + 1) % 500 == 0:
            print(f"  Tokenized: {i + 1}/{len(all_chunks)}")

    # -----------------------------------------------------------------------
    # 5. Build BM25 index
    # -----------------------------------------------------------------------
    print("Building BM25 index...")
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized)

    # -----------------------------------------------------------------------
    # 6. Build dense embeddings + FAISS index
    # -----------------------------------------------------------------------
    import numpy as np
    import faiss
    from src.indexer import encode_texts, load_embed_model
    from src.config import EMBED_PROVIDER, EMBED_MODEL, EMBED_MODEL_PATH

    # --embed-model CLI flag overrides EMBED_MODEL env var for this run
    if args.embed_model:
        os.environ["EMBED_MODEL"] = args.embed_model

    effective_model = args.embed_model or EMBED_MODEL
    # For llama-cpp the actual file is EMBED_MODEL_PATH, not EMBED_MODEL
    display_model = EMBED_MODEL_PATH if EMBED_PROVIDER == "llama-cpp" else effective_model
    print(f"Building FAISS dense index | provider={EMBED_PROVIDER} model={display_model}")
    embed_model = load_embed_model()

    print(f"Encoding {len(all_chunks)} chunks...")
    embeddings = encode_texts(
        embed_model,
        [c.text for c in all_chunks],
        is_query=False,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,
    )

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings.astype(np.float32))
    print(f"FAISS index: {faiss_index.ntotal} vectors, dim={embeddings.shape[1]}")

    # -----------------------------------------------------------------------
    # 7. Save everything
    # -----------------------------------------------------------------------
    import pickle

    print("Saving BM25 index...")
    with open(bm25_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "tokenized": tokenized,
            "embed_model_name": effective_model,
        }, f)

    print("Saving FAISS index...")
    faiss.write_index(faiss_index, str(faiss_path))

    print("Saving chunks metadata...")
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)

    # Metadata summary
    meta = {
        "num_protocols": len(protocols),
        "num_chunks": len(all_chunks),
        "embed_provider": EMBED_PROVIDER,
        "embed_model": effective_model,
        "faiss_dim": int(embeddings.shape[1]),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nIndex build complete!")
    print(f"  Protocols:  {meta['num_protocols']}")
    print(f"  Chunks:     {meta['num_chunks']}")
    print(f"  FAISS dim:  {meta['faiss_dim']}")
    print(f"  Output dir: {output_dir}")
    print(f"  Trees dir:  {trees_dir}")


if __name__ == "__main__":
    main()
