"""Single GPU queue. usage: queue_runner.py <jobs file> <max running> <gap seconds>

Jobs file lines: `<need_gb> <tag> <command...>`; finished tags go to logs/jobs/done.txt, so restarts skip them.
The file is re-read each cycle, so jobs can be appended. Exit code 75 requeues.
"""
import os
import subprocess
import sys
import time

jobs_file, max_run, gap = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
DONE = "logs/jobs/done.txt"
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0",
           PYTORCH_ALLOC_CONF="expandable_segments:True", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
running, last, fails = {}, 0.0, {}


def gpus():
    o = subprocess.run("nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits",
                       shell=True, capture_output=True, text=True).stdout
    return sorted(((int(f), int(i), int(u)) for i, f, u in (l.split(", ") for l in o.strip().splitlines())), key=lambda t: (t[2] > 90, -t[0]))


def done():
    return set(open(DONE).read().split()) if os.path.exists(DONE) else set()


while True:
    for tag, (p, f) in list(running.items()):
        rc = p.poll()
        if rc is None:
            continue
        f.close()
        del running[tag]
        if rc == 0:
            open(DONE, "a").write(tag + "\n")
        else:
            fails[tag] = fails.get(tag, 0) + (rc != 75)
            print(time.strftime("%H:%M"), tag, "rc", rc, flush=True)
    jobs = [l.split(None, 2) for l in open(jobs_file) if l.strip()]
    d = done()
    todo = [j for j in jobs if j[1] not in d and j[1] not in running and fails.get(j[1], 0) < 3]
    if not todo and not running:
        break
    if todo and len(running) < max_run and time.time() - last >= gap:
        g = gpus()
        for need, tag, cmd in todo:
            fit = [x for x in g if x[0] / 1024 >= float(need) + 0.4]
            if fit:
                f = open(f"logs/jobs/{tag}.log", "a")
                e = dict(env, CUDA_VISIBLE_DEVICES=str(fit[0][1]))
                running[tag] = (subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=e), f)
                last = time.time()
                print(time.strftime("%H:%M"), "start", tag, "gpu", fit[0][1], "free", fit[0][0], flush=True)
                break
    time.sleep(3)
print("QUEUE EMPTY", time.strftime("%H:%M"), flush=True)
