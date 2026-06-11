import sys
import os
import streamlit.web.cli as stcli

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Programmatically run: streamlit run app.py --server.port 8000
    sys.argv = ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "127.0.0.1"]
    print("Launching Streamlit chatbot app on http://127.0.0.1:8000")
    sys.exit(stcli.main())
