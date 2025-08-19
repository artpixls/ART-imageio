import subprocess, sys

subprocess.run(['gpr_tools', '-i', sys.argv[1], '-o', sys.argv[2]], check=True)
