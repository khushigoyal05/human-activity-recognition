import os
import sys

print("Starting Kinetics AI Streamlit Dashboard...")
print("If the browser does not open automatically, look for the Network URL below.")

try:
    os.system("streamlit run app.py")
except KeyboardInterrupt:
    print("\nShutdown complete.")
