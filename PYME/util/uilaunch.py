import sys
import os
import shutil

def ensure_macos_framework_build():
    """
    Ensures the script is running with a Framework build of Python (python.app)
    on macOS. If not, it attempts to find one and relaunches the process.
    """
    if sys.platform != "darwin":
        return

    # If we are already running inside a .app bundle (standard Framework build)
    # or the specific python.app executable, we are good.
    if ".app/Contents/MacOS/python" in sys.executable:
        return

    current_exe = sys.executable
    env_base = sys.prefix  # The root of the current environment (venv or conda env)
    

    # 1. Look for Conda's specific python.app (The most common need for this)
    # Structure: <env>/python.app/Contents/MacOS/python
    conda_app = os.path.join(env_base, "python.app", "Contents", "MacOS", "python")

    if os.path.exists(conda_app) and os.access(conda_app, os.X_OK):
        target_python = conda_app
    else:
        target_python = None
        print("Warning: Could not find a python.app executable in the current conda environment.")
        return
    

    # 4. The Relaunch
    # Only relaunch if we found a better python AND we aren't already running it.
    if target_python and os.path.abspath(current_exe) != os.path.abspath(target_python):
        # We use execv to replace the current process (keeping PID same-ish)
        # We pass the new executable as argv[0] and all original args
        os.execv(target_python, [target_python] + sys.argv)