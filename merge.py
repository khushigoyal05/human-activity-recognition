"""Safe merger: reads all part files and appends them to har_pipeline.py."""
import os

parts = ["p2.py", "p3.py", "p4.py"]

with open("har_pipeline.py", "a", encoding="utf-8", newline="\n") as out:
    for fname in parts:
        with open(fname, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        out.write("\n\n")
        out.write(content)
        out.write("\n")
        print(f"Appended {fname}")

for fname in parts:
    os.remove(fname)
    print(f"Removed {fname}")

print("Done.")
