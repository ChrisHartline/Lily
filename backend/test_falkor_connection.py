"""
Quick test script to verify FalkorDB Cloud connection.
Run this to test your credentials before deploying to Modal.

Usage:
    python backend/test_falkor_connection.py
"""

import os
import sys

# Your FalkorDB Cloud credentials
FALKOR_HOST = "r-6jissuruar.instance-i4awinx1v.hc-20vidasdi.us-central1.gcp.f2e0a955bb84.cloud"
FALKOR_PORT = 55041
FALKOR_PASSWORD = os.environ.get("FALKORDB_HOSTED_KEY", "Lily26!")
FALKOR_SSL = True
FALKOR_GRAPH = "claralilymem"

def test_connection():
    print("=" * 60)
    print("FalkorDB Cloud Connection Test")
    print("=" * 60)
    print(f"Host: {FALKOR_HOST}")
    print(f"Port: {FALKOR_PORT}")
    print(f"SSL: {FALKOR_SSL}")
    print(f"Graph: {FALKOR_GRAPH}")
    print(f"Password: {'*' * len(FALKOR_PASSWORD)}")
    print()

    import socket
    
    try:
        from falkordb import FalkorDB
        print("[1/5] OK - FalkorDB library imported")
    except ImportError:
        print("[1/5] FAIL - FalkorDB not installed. Run: pip install falkordb")
        return False

    # Test basic TCP connectivity first
    try:
        print(f"[2/5] Testing TCP connectivity to {FALKOR_HOST}:{FALKOR_PORT}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((FALKOR_HOST, FALKOR_PORT))
        sock.close()
        if result == 0:
            print(f"[2/5] OK - TCP port {FALKOR_PORT} is reachable")
        else:
            print(f"[2/5] FAIL - Cannot reach {FALKOR_HOST}:{FALKOR_PORT} (error: {result})")
            print(f"      Check: firewall, VPN, or if the instance is running")
            return False
    except socket.timeout:
        print(f"[2/5] FAIL - TCP connection timed out")
        return False
    except socket.gaierror as e:
        print(f"[2/5] FAIL - DNS resolution failed: {e}")
        return False
    except Exception as e:
        print(f"[2/5] FAIL - TCP test failed: {type(e).__name__}: {e}")
        return False

    try:
        print(f"[3/5] Connecting to FalkorDB Cloud...")
        print(f"      (This may take a few seconds...)")
        
        socket.setdefaulttimeout(30)  # 30 second timeout
        
        db = FalkorDB(
            host=FALKOR_HOST,
            port=FALKOR_PORT,
            password=FALKOR_PASSWORD,
            ssl=FALKOR_SSL,
        )
        print("[3/5] OK - Connected to FalkorDB")
    except socket.timeout:
        print(f"[3/5] FAIL - Connection timed out after 30 seconds")
        print(f"      SSL handshake may have failed")
        return False
    except Exception as e:
        print(f"[3/5] FAIL - Connection failed: {type(e).__name__}: {e}")
        return False

    try:
        print(f"[4/5] Selecting graph '{FALKOR_GRAPH}'...")
        graph = db.select_graph(FALKOR_GRAPH)
        print(f"[4/5] OK - Graph selected")
    except Exception as e:
        print(f"[4/5] FAIL - Graph selection failed: {e}")
        return False

    try:
        print("[5/5] Testing query (node count)...")
        result = graph.query("MATCH (n) RETURN count(n) as count")
        count = result.result_set[0][0] if result.result_set else 0
        print(f"[5/5] OK - Query successful - {count} nodes in graph")
    except Exception as e:
        print(f"[5/5] FAIL - Query failed: {e}")
        return False

    print()
    print("=" * 60)
    print("SUCCESS! FalkorDB Cloud connection is working.")
    print("=" * 60)
    print()
    print("For Modal deployment, create the secret with:")
    print()
    print(f'modal secret create falkordb-secret \\')
    print(f'  FALKOR_HOST="{FALKOR_HOST}" \\')
    print(f'  FALKOR_PORT="{FALKOR_PORT}" \\')
    print(f'  FALKOR_PASSWORD="{FALKOR_PASSWORD}" \\')
    print(f'  FALKOR_SSL="true"')
    print()
    return True


if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    success = test_connection()
    sys.exit(0 if success else 1)
