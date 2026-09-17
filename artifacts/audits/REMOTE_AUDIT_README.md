# Remote environment audit

Run from a host with functional SSH:

```bash
bash scripts/remote_environment_audit.sh main 12956 artifacts/audits/remote
bash scripts/remote_environment_audit.sh gpu 14170 artifacts/audits/remote
```

Each output records hostname, working directory, Python executable/version, GPU state, candidate git/data directories, commit hashes, and a local SHA-256. No file transfer is performed.

Current Windows workspace result: `ssh` resolves to `C:\Users\liujunqing\.sbx-denybin\ssh.bat` and exits before network connection; remote facts remain unverified.
