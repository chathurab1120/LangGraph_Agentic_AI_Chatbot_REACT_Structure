import subprocess
import sys

def run_tests():
    """Run tests with coverage reporting"""
    print("Running tests with coverage...")
    
    # Run pytest with coverage using python -m pytest
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "test_enhanced_chatbot.py",
        "-v",
        "--cov",
        "--cov-report=term-missing",
        "--cov-report=html"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.stderr:
            print("Errors:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        
        # Check if tests passed
        if result.returncode != 0:
            print("Tests failed!")
            sys.exit(1)
        
        print("\nTests completed successfully!")
        print("Coverage report has been generated in htmlcov/index.html")
        
    except Exception as e:
        print(f"Error running tests: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_tests() 