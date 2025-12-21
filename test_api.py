"""
Test script for Oil Leakage & Corrosion Detection API
"""
import requests
import json
import sys
import time
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_root():
    """Test root endpoint"""
    print("Testing root endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_list_issues():
    """List all detectable issues"""
    print("Listing all detectable issues...")
    response = requests.get(f"{API_URL}/issues")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total issues: {data['total_issues']}\n")
        
        for issue in data['issues']:
            print(f"  - {issue['name']} ({issue['severity']})")
    print()

def test_detect_leakage_async(image_path: str):
    """Test async leakage detection with image upload"""
    print(f"\n{'='*60}")
    print(f"Testing ASYNC Oil & Corrosion detection with: {image_path}")
    print(f"{'='*60}\n")
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Step 1: Upload image
    print("Step 1: Uploading image to API...")
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        # Endpoint: POST /detect (see API root docs)
        response = requests.post(f"{API_URL}/detect", files=files)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return
    
    data = response.json()
    print(f"✅ Success!")
    print(f"   Request ID: {data['request_id']}")
    print(f"   Status URL: {data['status_url']}")
    print(f"   Message: {data['message']}")
    
    request_id = data['request_id']
    
    # Step 2: Poll for results
    print(f"\nStep 2: Waiting for worker to process...")
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        time.sleep(1)  # Wait 1 second between polls
        attempt += 1
        
        response = requests.get(f"{API_URL}/result/{request_id}")
        result = response.json()
        status = result['status']
        
        print(f"   [{attempt}/{max_attempts}] Status: {status}", end='\r')
        
        if status == 'completed':
            print(f"\n\n✅ Processing completed in ~{attempt} seconds!")
            print(f"\n{'='*60}")
            print("DETECTION RESULTS")
            print(f"{'='*60}\n")
            
            print(f"Request ID: {result['request_id']}")
            print(f"Processing Time: {result.get('processing_time', 'N/A')}s")
            
            issues = result.get('detected_issues', [])
            print(f"\nDetected Issues ({len(issues)}):")
            
            for issue in issues:
                print(f"\n  🏭 Issue: {issue['issue_name'].upper()}")
                print(f"     Confidence: {issue['confidence']}%")
                print(f"     Severity: {issue['severity']}")
                
                if issue.get('demo_mode'):
                    print(f"     ⚠️  DEMO MODE (model not loaded)")
                
                print(f"\n     Description:")
                print(f"     {issue.get('description', '')[:200]}...")
                
                print(f"\n     Action Plan:")
                for step in issue.get('action_plan', [])[:3]:
                     print(f"     - {step}")
            
            print(f"\n{'='*60}\n")
            return
            
        elif status == 'failed':
            print(f"\n\n❌ Processing failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return
    
    print(f"\n\n⏱️  Timeout: Result not ready after {max_attempts} seconds")
    print("   The worker might be processing a large queue or not running")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Oil Leakage & Corrosion API - Test Script")
    print("=" * 60)
    print()
    
    # Check if API is running
    try:
        test_health()
        test_root()
        # test_list_issues()
        
        # Test with image if provided
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            test_detect_leakage_async(image_path)
        else:
            print("=" * 60)
            print("No image provided. To test async detection:")
            print(f"  python {Path(__file__).name} /path/to/transformer_image.jpg")
            print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to API at {API_URL}")
        print("\nMake sure the services are running:")
        print("  docker-compose up")
        sys.exit(1)

if __name__ == "__main__":
    main()
