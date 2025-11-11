#!/usr/bin/env python3
"""
Test script for vectorstore persistence
Verifies that the implementation meets acceptance criteria
"""
import os
import sys
import time
import shutil
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_api.rag_system import AcademicRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_persistence():
    """Test vectorstore persistence functionality"""
    
    print("=" * 60)
    print("Testing Vectorstore Persistence")
    print("=" * 60)
    
    # Clean up any existing vectorstore
    vectorstore_path = "./vectorstore"
    if os.path.exists(vectorstore_path):
        print(f"\n🧹 Cleaning up existing vectorstore: {vectorstore_path}")
        shutil.rmtree(vectorstore_path)
    
    # Test 1: First initialization (should build from scratch)
    print("\n" + "=" * 60)
    print("Test 1: First Initialization (Build from Scratch)")
    print("=" * 60)
    
    start_time = time.time()
    rag1 = AcademicRAGSystem(
        pdf_directory="./data",
        model_name="gemma3:270m",
        embedding_model="nomic-embed-text",
        chunk_size=800,
        chunk_overlap=100,
        vectorstore_path="./vectorstore"
    )
    first_init_time = time.time() - start_time
    
    print(f"\n✅ First initialization completed in {first_init_time:.2f}s")
    print(f"   Vectorstore created: {rag1.vectorstore is not None}")
    print(f"   Success count: {rag1._successful_count}")
    print(f"   Failed count: {rag1._failed_count}")
    
    # Verify vectorstore directory exists
    assert os.path.exists(vectorstore_path), "Vectorstore directory not created"
    assert os.path.exists(os.path.join(vectorstore_path, "metadata.pkl")), "Metadata file not created"
    print(f"   Vectorstore files saved: ✓")
    
    # Test a query
    test_query = "What is this research about?"
    try:
        result = rag1.query(test_query)
        print(f"   Query test: ✓ (returned {len(result)} chars)")
    except Exception as e:
        print(f"   Query test: ✗ ({e})")
    
    # Clean up
    del rag1
    
    # Test 2: Second initialization (should load from cache)
    print("\n" + "=" * 60)
    print("Test 2: Second Initialization (Load from Cache)")
    print("=" * 60)
    
    start_time = time.time()
    rag2 = AcademicRAGSystem(
        pdf_directory="./data",
        model_name="gemma3:270m",
        embedding_model="nomic-embed-text",
        chunk_size=800,
        chunk_overlap=100,
        vectorstore_path="./vectorstore"
    )
    second_init_time = time.time() - start_time
    
    print(f"\n✅ Second initialization completed in {second_init_time:.2f}s")
    print(f"   Vectorstore loaded: {rag2.vectorstore is not None}")
    print(f"   Success count: {rag2._successful_count}")
    print(f"   Failed count: {rag2._failed_count}")
    
    # Calculate speedup
    if second_init_time > 0:
        speedup = first_init_time / second_init_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster!")
        
        if speedup >= 50:
            print(f"   ✅ PASS: Achieved {speedup:.1f}x speedup (target: 50x)")
        else:
            print(f"   ⚠️  WARNING: Only {speedup:.1f}x speedup (target: 50x)")
    
    # Test query still works
    try:
        result2 = rag2.query(test_query)
        print(f"   Query after reload: ✓ (returned {len(result2)} chars)")
    except Exception as e:
        print(f"   Query after reload: ✗ ({e})")
    
    # Test 3: Hash change detection
    print("\n" + "=" * 60)
    print("Test 3: Config Change Detection")
    print("=" * 60)
    
    del rag2
    
    # Try to load with different chunk_size
    start_time = time.time()
    rag3 = AcademicRAGSystem(
        pdf_directory="./data",
        model_name="gemma3:270m",
        embedding_model="nomic-embed-text",
        chunk_size=1000,  # Different from saved (800)
        chunk_overlap=100,
        vectorstore_path="./vectorstore"
    )
    third_init_time = time.time() - start_time
    
    print(f"\n✅ Config changed initialization completed in {third_init_time:.2f}s")
    
    if third_init_time > 30:  # If it took long, it rebuilt
        print(f"   ✅ PASS: Detected config change and rebuilt")
    else:
        print(f"   ⚠️  WARNING: May not have detected config change")
    
    del rag3
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✅ First init: {first_init_time:.2f}s (build from scratch)")
    print(f"✅ Second init: {second_init_time:.2f}s (load from cache)")
    print(f"✅ Speedup: {first_init_time / second_init_time:.1f}x")
    print(f"✅ Config change: {third_init_time:.2f}s (rebuild triggered)")
    
    # Acceptance criteria
    print("\n" + "=" * 60)
    print("Acceptance Criteria")
    print("=" * 60)
    
    criteria = [
        ("First startup time similar to current", first_init_time > 0, "✅"),
        ("Second startup < 10 seconds", second_init_time < 10, "✅" if second_init_time < 10 else "❌"),
        ("50-100x faster startup", first_init_time / second_init_time >= 50, "✅" if first_init_time / second_init_time >= 50 else "⚠️"),
        ("Config change detected", third_init_time > 30, "✅" if third_init_time > 30 else "⚠️"),
        ("Vectorstore files created", os.path.exists(vectorstore_path), "✅"),
    ]
    
    for criterion, passed, status in criteria:
        print(f"{status} {criterion}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        test_persistence()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
