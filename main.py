import os
import sys

scripts = [
    "generate_processed_datasets.py",
    "run_baseline_inference.py",
    "run_processed_inference.py",
    "compare_with_baseline.py",
    "metrics.py",
    "create_visual_panels.py"
]

if __name__ == "__main__":
    for script in scripts:
        print(f"\n{'='*50}\nRUNNING: {script}\n{'='*50}")
        ret = os.system(f'"{sys.executable}" {script}')
        if ret != 0:
            print(f"Error running {script}. Exiting.")
            sys.exit(1)
    print("\nSUCCESS: Complete pipeline execution finished.")
