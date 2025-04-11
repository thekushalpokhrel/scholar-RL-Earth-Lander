import sys
print("Python Path:")
for p in sys.path:
    print(p)

print("\nTrying to import...")
try:
    import earth_lander
    print("SUCCESS: Import worked!")
except ImportError as e:
    print(f"FAILED: {e}")
