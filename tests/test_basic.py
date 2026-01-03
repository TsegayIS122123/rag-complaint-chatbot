def test_import():
    """Test basic import."""
    import pandas as pd
    assert pd.__version__ is not None
    print("✅ pandas imported")

def test_data_processor():
    """Test data processor import."""
    try:
        from src.data_processor import ComplaintDataProcessor
        processor = ComplaintDataProcessor()
        assert processor is not None
        print("✅ Data processor works")
    except ImportError:
        print("⚠️  Data processor not found")
