"""Friendly live dashboard for G1 get-up training. Owned by the G1 app.

Canonical home (moved from legacy dashboard.py). Prefer:

    g1 dashboard
    g1 dashboard --port 6006
"""
import argparse
import glob
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from core.config import resolve_videos_dir

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(LAB_DIR)
WORKSPACE = os.path.dirname(APP_DIR)
LOG_ROOT = os.path.join(WORKSPACE, "unitree_rl_mjlab", "logs")
VIDEOS_DIR = resolve_videos_dir()

# tag -> (friendly name, plain-language explanation)
METRICS = {
  "Episode_Reward/stand_success": (
    "Stand-up success",
    "Bonus earned for ending an episode standing tall and upright (max 3.0). "
    "This is the money chart — when it climbs toward 3, the robot has learned to get up.",
  ),
  "Episode_Reward/stand_height": (
    "Standing height",
    "How close the torso gets to full standing height (max ~2.0). "
    "Rises first — the robot learns to get tall before it learns to be steady.",
  ),
  "Episode_Reward/upright": (
    "Torso upright",
    "How level the torso is (max 1.0). Rises as face-down/face-up starts end upright.",
  ),
  "Episode_Reward/stand_pose": (
    "Standing pose",
    "How closely the joints match a natural standing pose (max 0.5). "
    "Fine-tuning that comes last.",
  ),
  "Episode_Reward/feet_force": (
    "Pushing through feet",
    "Foot-ground force while recovering. The dense, honest gradient toward "
    "standing — head-bridging can't fake it.",
  ),
  "Episode_Reward/stand_on_feet": (
    "Standing on feet",
    "Bonus for tall + upright + supported by both feet. This is the "
    "correct-way-up detector.",
  ),
  "Episode_Reward/bad_support": (
    "Bad support penalty",
    "Negative — torso/arms touching the ground while the body is high. "
    "Lying flat is free; propping your head is not.",
  ),
  "Episode_Reward/symmetry": (
    "Asymmetry cost",
    "Tiny penalty for lopsided left/right actions. Keeps recovery natural.",
  ),
  "Episode_Reward/joint_torques": (
    "Effort penalty",
    "Motor torque squared — gentle, low-effort motion scores better. "
    "Protects the robot and anyone near it.",
  ),
  "Episode_Reward/joint_vel": (
    "Speed penalty",
    "Joint velocity squared — no whipping or slamming.",
  ),
  "Episode_Reward/action_rate_l2": (
    "Jerkiness penalty",
    "Negative — punishes twitchy motion. Should settle near zero, not plunge.",
  ),
  "Train/mean_reward": (
    "Total reward",
    "Everything added together per episode. Overall direction of learning.",
  ),
  "Episode_Reward/no_head_contact": (
    "Head contact penalty",
    "Negative — head/neck touching the ground while the body is high. The G1 model has no separate head_link; this sensor matches torso and blocks head-bridging.",
  ),
  "Episode_Reward/pelvis_rising": (
    "Pelvis rising",
    "Reward for pelvis moving toward standing height (exp kernel). Shapes the early-to-mid rise so the agent learns a smooth upward trajectory rather than a head-down / butt-up collapse.",
  ),
  "Episode_Reward/com_vel_z": (
    "Vertical CoM velocity",
    "Reward for positive vertical CoM velocity (exp kernel). Directly shapes the rising motion — the agent gets immediate gradient when its centre of mass moves up, blocking head-down / butt-up shortcuts.",
  ),
  "Episode_Reward/supine_success": (
    "Supine success",
    "Bonus for ending a Reposition episode flat on the back in the neutral pose (max 3.0). "
    "This is the Stage-A money chart — when it climbs toward 3, the robot reliably untangles into the sit-up start pose.",
  ),
  "Episode_Reward/supine_pose": (
    "Supine pose",
    "How closely the joints match the flat-on-back neutral pose (max 1.5). "
    "Dense shaping behind the sparse supine bonus.",
  ),
  "Episode_Reward/torso_horizontal": (
    "Torso horizontal",
    "Reward for the torso lying flat (max 1.0). The opposite pole of the upright bonus — Reposition wants the robot flat and stable before SitUp starts.",
  ),
  "Episode_Reward/roll_success": (
    "Roll-over success",
    "Bonus for ending a Roll episode low and face-up (max 3.0). The v2 "
    "funnel money chart — when it climbs toward 3, any fall reliably "
    "becomes a supine lying pose.",
  ),
  "Episode_Reward/face_up": (
    "Facing up",
    "How closely the torso gravity matches flat face-up (max 1.5). Dense "
    "shaping behind the sparse roll bonus.",
  ),
  "Episode_Reward/brace_success": (
    "Brace success",
    "Bonus for ending a Brace episode safely: settled on the ground or "
    "stumbled back to standing (max 2.0). The independent brace money "
    "chart — climbs when doomed falls end without damage.",
  ),
  "Episode_Reward/torso_impact": (
    "Torso impact",
    "Per-step torso-ground force (max 1.0, priced at -3.0). The vulnerable "
    "core proxy (the head rides on the torso — no separate head link): "
    "faceplants and chest-slams spike it, distributed back landings do not.",
  ),
  "Episode_Reward/arm_impact": (
    "Arm impact",
    "Per-step arm-ground force (max 1.0, priced at only -0.3). Arms are the "
    "cheap absorbers — this is where protective contact should show up.",
  ),
  "Episode_Reward/leg_impact": (
    "Leg impact",
    "Per-step leg-ground force (max 1.0, priced at -0.8).",
  ),
  "Episode_Reward/landed_face_up": (
    "Landed face-up",
    "Shaping toward back-landings (max 0.5): the head-safe way to meet "
    "the ground.",
  ),
  "Episode_Reward/pelvis_impact": (
    "Pelvis impact",
    "Per-step pelvis-ground force (max 1.0, priced at -1.0).",
  ),
  "Episode_Reward/stand_still": (
    "Standing still",
    "Reward for a quiet root (low body velocity, max 1.0). Recovery steps are "
    "allowed — only root motion counts, not foot movement.",
  ),
}

LIVE_STALE_S = 90  # no new data for this long -> training considered stopped
MAX_POINTS = 400

# Chart groups, drawn as canvases c0/c1/c2 (missing tags are skipped, so one
# layout serves stand, roll and standup runs). Single source: extend here
# (and METRICS above) when rewards change — the page template below only
# carries a placeholder.
CHART_GROUPS = [
  ["Episode_Reward/stand_success", "Episode_Reward/roll_success",
   "Episode_Reward/brace_success"],
  ["Episode_Reward/stand_height", "Episode_Reward/upright",
   "Episode_Reward/stand_on_feet", "Episode_Reward/feet_force",
   "Episode_Reward/stand_still", "Episode_Reward/supine_pose",
   "Episode_Reward/torso_horizontal", "Episode_Reward/face_up",
   "Episode_Reward/pelvis_rising", "Episode_Reward/com_vel_z",
   "Episode_Reward/torso_impact", "Episode_Reward/arm_impact",
   "Episode_Reward/leg_impact", "Episode_Reward/pelvis_impact",
   "Episode_Reward/landed_face_up"],
  ["Train/mean_reward"],
]

# Same tag, different story: when a stand run is selected, these replace the
# get-up-flavoured names/explanations in the chart legends.
STAND_METRIC_OVERRIDES = {
  "Episode_Reward/stand_success": (
    "Balance success",
    "Bonus for staying tall, upright and supported by both feet through "
    "shoves (max 5.0). When it climbs toward 5 and stays there, the robot "
    "can stand through pushes.",
  ),
  "Episode_Reward/stand_height": (
    "Standing height",
    "How close the torso stays to full standing height. Dips mark crouches "
    "and dips during shoves.",
  ),
  "Episode_Reward/upright": (
    "Torso upright",
    "How level the torso stays. Dips when a shove tilts the body.",
  ),
  "Episode_Reward/stand_on_feet": (
    "Supported standing",
    "Tall + upright + weight on both feet at once. The stayed-standing "
    "detector.",
  ),
  "Episode_Reward/feet_force": (
    "Weight on feet",
    "Foot-ground force while standing. The dense signal that the load is on "
    "the feet.",
  ),
}


def _load_yaml(path):
  """Read a few scalar fields from a params file.

  The files contain !!python/tuple tags that SafeLoader rejects, so plain
  regex extraction instead of full YAML parsing.
  """
  import re
  out = {}
  try:
    with open(path) as f:
      text = f.read()
    for key in ("max_iterations", "num_steps_per_env", "num_envs",
                "experiment_name", "run_name"):
      m = re.search(rf"^\s*{key}:\s*([0-9]+)", text, re.M)
      if m:
        out[key] = int(m.group(1))
  except Exception:
    pass
  return out


class RunStore:
  """Cache of parsed event files, reloaded only when they change."""

  def __init__(self):
    self._lock = threading.Lock()
    self._cache: dict[str, dict] = {}

  def runs(self):
    found = []
    for root, _dirs, files in os.walk(LOG_ROOT):
      if any(f.startswith("events.out.tfevents.") for f in files):
        found.append(root)
    def newest(d):
      return max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)
                 if f.startswith("events.out.tfevents."))
    found.sort(key=newest, reverse=True)
    return found

  def describe(self, run_dir):
    with self._lock:
      mtime = max(os.path.getmtime(f)
                  for f in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
      cached = self._cache.get(run_dir)
      if cached and cached["mtime"] >= mtime:
        return cached["data"]
      data = self._parse(run_dir)
      self._cache[run_dir] = {"mtime": mtime, "data": data}
      # drop runs that no longer exist from cache
      for k in [k for k in self._cache if not os.path.isdir(k)]:
        del self._cache[k]
      return data

  def _parse(self, run_dir):
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(
      run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {"metrics": {}, "last_event_age_s": None,
           "elapsed_s": 0.0, "iters_per_s": 0.0}
    have_tags = set(ea.Tags().get("scalars", []))
    series = {}
    first_t = last_t = None
    for tag in have_tags:
      try:
        evs = ea.Scalars(tag)
      except Exception:
        continue
      if not evs:
        continue
      if first_t is None or evs[0].wall_time < first_t:
        first_t = evs[0].wall_time
      if last_t is None or evs[-1].wall_time > last_t:
        last_t = evs[-1].wall_time
      step = [e.step for e in evs]
      val = [e.value for e in evs]
      if len(step) > MAX_POINTS:
        stride = len(step) // MAX_POINTS + 1
        step, val = step[::stride], val[::stride]
      series[tag] = {"step": step, "value": val,
                     "last": val[-1], "last_step": step[-1]}
    out["metrics"] = series
    if last_t is not None:
      out["last_event_age_s"] = time.time() - last_t
      out["elapsed_s"] = last_t - (first_t or last_t)
      # speed from the most-logged series
      longest = max(series.values(), key=lambda s: len(s["step"]), default=None)
      if longest and len(longest["step"]) >= 3 and out["elapsed_s"] > 0:
        out["iters_per_s"] = (longest["last_step"] - longest["step"][0]) / out["elapsed_s"]
    # training config for totals + ETA
    agent = _load_yaml(os.path.join(run_dir, "params", "agent.yaml"))
    env = _load_yaml(os.path.join(run_dir, "params", "env.yaml"))
    num_envs = int(env.get("num_envs", 0) or 0)
    steps_per_env = int(agent.get("num_steps_per_env", 0) or 0)
    out["max_iterations"] = agent.get("max_iterations")
    out["num_envs"] = num_envs
    out["steps_per_second"] = out["iters_per_s"] * steps_per_env * num_envs
    return out


STORE = RunStore()


# ---------------------------------------------------------------- recordings
REC_JOB: dict = {"state": "idle", "started": 0.0, "error": ""}
REC_LOCK = threading.Lock()


def _read_json(path):
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return {}


def latest_video_info(run_id=None):
  """Newest recording for a run (or overall). Returns None when absent."""
  cands = []
  if not os.path.isdir(VIDEOS_DIR):
    return None
  want = os.path.basename(run_id) if run_id else None
  for jf in glob.glob(os.path.join(VIDEOS_DIR, "*.json")):
    info = _read_json(jf)
    mp4 = os.path.splitext(jf)[0] + ".mp4"
    if not os.path.isfile(mp4):
      continue
    tag = os.path.basename(jf)
    if want and want not in tag and want not in str(info.get("policy", "")):
      continue
    cands.append((os.path.getmtime(jf), jf, mp4, info))
  if not cands and want:
    return latest_video_info(None)
  if not cands:
    return None
  mtime, _jf, mp4, info = sorted(cands)[-1]
  eps = info.get("episodes", [])
  return {
    "url": "/videos/" + os.path.basename(mp4) + f"?v={int(mtime)}",
    "recorded_at": info.get("recorded_at", mtime),
    "episodes": len(eps),
    "stood_up": info.get("stood_up", "?"),
    "policy_mtime": info.get("checkpoint_mtime"),
  }


def start_recording(run_id=None):
  with REC_LOCK:
    if REC_JOB["state"] == "recording":
      return dict(REC_JOB)
    REC_JOB.update(state="recording", started=time.time(), error="")
  t = threading.Thread(target=_record_job, args=(run_id,), daemon=True)
  t.start()
  return dict(REC_JOB)


def _record_job(run_id):
  try:
    try:
        from lab import record as record_getup
    except ImportError:
        from lab import record as record_getup  # type: ignore
    if run_id and os.path.isfile(os.path.join(run_id, "policy.onnx")):
      chosen = run_id
    else:
      chosen = record_getup.find_latest_run()
    standing = "g1_stand" in chosen
    record_getup.record(record_getup.latest_policy_onnx(chosen),
                        episodes=3, seconds=8.0, standing=standing)
    with REC_LOCK:
      REC_JOB.update(state="done")
  except Exception as e:  # noqa: BLE001 - surfaced in the UI
    with REC_LOCK:
      REC_JOB.update(state="error", error=f"{type(e).__name__}: {e}")


def record_job_status():
  with REC_LOCK:
    d = dict(REC_JOB)
  d["elapsed_s"] = time.time() - d["started"] if d["state"] == "recording" else 0.0
  if d["state"] == "done" and time.time() - d["started"] > 30:
    with REC_LOCK:
      REC_JOB.update(state="idle")
    d["state"] = "idle"
  return d


def status_payload(want_run=None):
  runs = [{"id": d, "name": os.path.relpath(d, LOG_ROOT)} for d in STORE.runs()]
  if not runs:
    return {"runs": [], "run": None, "task": None, "video": latest_video_info(),
            "record_job": record_job_status()}
  run_id = want_run if any(r["id"] == want_run for r in runs) else runs[0]["id"]
  task = "stand" if "g1_stand" in run_id else ("getup" if "g1_getup" in run_id
                                              else "other")
  desc = STORE.describe(run_id)
  age = desc.get("last_event_age_s")
  live = age is not None and age < LIVE_STALE_S
  it = 0
  for m in desc["metrics"].values():
    it = max(it, m["last_step"])
  eta_s = None
  if desc.get("max_iterations") and desc.get("iters_per_s"):
    eta_s = max(0.0, (desc["max_iterations"] - it) / desc["iters_per_s"])
  metric_help = {t: {"name": n, "help": h} for t, (n, h) in METRICS.items()}
  if task == "stand":
    for t, (n, h) in STAND_METRIC_OVERRIDES.items():
      metric_help[t] = {"name": n, "help": h}
  return {
    "runs": runs,
    "run": {"id": run_id, "name": os.path.relpath(run_id, LOG_ROOT)},
    "task": task,
    "live": live,
    "age_s": age,
    "iteration": it,
    "elapsed_s": desc.get("elapsed_s", 0.0),
    "iters_per_s": desc.get("iters_per_s", 0.0),
    "steps_per_second": desc.get("steps_per_second", 0.0),
    "eta_s": eta_s,
    "max_iterations": desc.get("max_iterations"),
    "num_envs": desc.get("num_envs"),
    "metrics": desc["metrics"],
    "metric_help": metric_help,
    "video": latest_video_info(run_id),
    "record_job": record_job_status(),
  }


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>G1 training dashboard</title>
<style>
body{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222}
.pill{display:inline-block;padding:4px 14px;border-radius:999px;font-weight:700;color:#fff}
.live{background:#1a9e4b}.idle{background:#888}
.card{border:1px solid #ddd;border-radius:10px;padding:12px 16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{font-size:22px;font-weight:700}.lbl{font-size:12px;color:#666}
canvas{width:100%;height:180px;border:1px solid #eee;border-radius:8px}
video{width:100%;border-radius:8px;background:#000}
.help{font-size:13px;color:#555}select{font-size:14px;padding:4px}
button{font-size:14px;padding:6px 14px;cursor:pointer}
h1{font-size:24px}h2{font-size:17px;margin:6px 0}
</style></head><body>
<h1><span id="title">G1 training</span> <span id="pill" class="pill idle">…</span></h1>
<div class="card"><label>Run: <select id="runs"></select></label>
<span id="liveage" class="lbl"></span></div>
<div class="card"><div class="grid" id="stats"></div></div>
<div class="card"><h2>Watch latest policy</h2>
<video id="vid" controls preload="none"></video>
<div class="help" id="vstat">No recording yet.</div>
<div style="margin-top:8px"><button id="recbtn">Record fresh video</button>
<span id="recstat" class="lbl"></span></div>
<div class="help">Records the newest saved policy (3 episodes from random falls, CPU-only —
safe to run while training continues). A new snapshot lands every ~100 training iterations.</div></div>
<div class="card"><h2><span id="moneytitle">Stand-up success</span> <span class="lbl">(the money chart)</span></h2>
<canvas id="c0" width="900" height="180"></canvas><div class="help" id="h0"></div></div>
<div class="card"><h2>Shaping rewards</h2>
<canvas id="c1" width="900" height="180"></canvas><div class="help" id="h1"></div></div>
<div class="card"><h2>Total reward</h2>
<canvas id="c2" width="900" height="180"></canvas><div class="help" id="h2"></div></div>
<script>
const GROUPS=__GROUPS_JSON__;
const COLORS=["#1a9e4b","#2563eb","#d97706","#7c3aed","#dc2626"];
const HELP={};
function fmtT(s){if(s==null||!isFinite(s))return "–";s=Math.floor(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return (h?h+"h ":"")+m+"m "+(s%60)+"s";}
function fmtAgo(t){if(!t)return "–";return fmtT(Date.now()/1000-t)+" ago";}
async function tick(){
  const sel=document.getElementById("runs");
  const r=await (await fetch("/api/status?run="+(sel.value||""))).json();
  Object.assign(HELP,r.metric_help||{});
  if(!r.run){document.getElementById("pill").textContent="NO RUNS";return;}
  if(!sel.options.length||sel.options[0].value.startsWith("__")){
    sel.innerHTML=r.runs.map(x=>`<option value="${x.id}">${x.name}</option>`).join("");
    sel.value=r.run.id;
  }
  const pill=document.getElementById("pill");
  pill.textContent=r.live?"● TRAINING":"○ IDLE";
  pill.className="pill "+(r.live?"live":"idle");
  document.getElementById("liveage").textContent=r.live?"updated just now":"last data "+fmtT(r.age_s)+" ago — showing history";
  const S=[
    ["Iteration",r.iteration+(r.max_iterations?" / "+r.max_iterations:"")],
    ["Elapsed",fmtT(r.elapsed_s)],["Speed",(r.steps_per_second?Math.round(r.steps_per_second).toLocaleString()+" steps/s":"–")],
    ["ETA",r.eta_s!=null?fmtT(r.eta_s):"–"],
    ["Environments",r.num_envs||"–"],
    ["Success now",(r.metrics["Episode_Reward/stand_success"]||r.metrics["Episode_Reward/roll_success"]||r.metrics["Episode_Reward/brace_success"]||{last:"–"}).last?.toFixed?.(2)??"–"],
  ];
  document.getElementById("stats").innerHTML=S.map(([l,v])=>`<div><div class="stat">${v}</div><div class="lbl">${l}</div></div>`).join("");
  const v=r.video,vid=document.getElementById("vid");
  if(v&&v.url){
    if(vid.dataset.src!==v.url){vid.dataset.src=v.url;vid.src=v.url;}
    document.getElementById("vstat").textContent=(r.task==="stand")
      ?`${v.stood_up} episodes stayed standing · ${v.episodes} recorded · video from ${fmtAgo(v.recorded_at)}`
      :`${v.stood_up} episodes ended standing · ${v.episodes} falls recorded · video from ${fmtAgo(v.recorded_at)}`;
  } else {
    document.getElementById("vstat").textContent="No recording yet — press “Record fresh video”.";
  }
  const j=r.record_job||{state:"idle"};
  document.getElementById("recstat").textContent=j.state==="recording"?`recording… ${fmtT(j.elapsed_s)} (charts keep updating meanwhile)`:j.state==="error"?("failed: "+(j.error||"unknown")):"";
  draw("c0",GROUPS[0],r);draw("c1",GROUPS[1],r);draw("c2",GROUPS[2],r);
  const isStand=r.task==="stand";
  document.getElementById("title").textContent=isStand?"G1 stand-still training":"G1 get-up training";
  document.getElementById("moneytitle").textContent=isStand?"Balance success":"Stand-up success";
  document.getElementById("h0").textContent=isStand
    ?"Bonus for staying tall, upright and on both feet through shoves (max 5.0). When it climbs toward 5 and stays there, the robot can stand through pushes."
    :(HELP["Episode_Reward/stand_success"]||{}).help||"";
  document.getElementById("h1").textContent=isStand
    ?"Height, upright torso, weight on both feet, plus staying still. All rise toward their max as balance improves."
    :"Height → tall first, upright torso, feet taking the load, then matching standing pose. All rise toward their max as recovery improves.";
  document.getElementById("h2").textContent="Everything summed per episode — overall direction of learning.";
}
function draw(id,tags,r){
  const c=document.getElementById(id),x=c.getContext("2d");
  x.clearRect(0,0,c.width,c.height);
  x.strokeStyle="#eee";x.beginPath();x.moveTo(34,0);x.lineTo(34,c.height-18);x.lineTo(c.width, c.height-18);x.stroke();
  tags.forEach((t,i)=>{
    const s=r.metrics[t];if(!s||s.step.length<2)return;
    const n=s.step.length;
    let mn=Math.min(...s.value),mx=Math.max(...s.value);if(mx-mn<1e-9){mx=mn+1;}
    const X=j=>34+(c.width-44)*j/(n-1),Y=v=>10+(c.height-36)*(1-(v-mn)/(mx-mn));
    x.strokeStyle=COLORS[i%COLORS.length];x.lineWidth=2;x.beginPath();
    s.value.forEach((v,j)=>j?x.lineTo(X(j),Y(v)):x.moveTo(X(j),Y(v)));x.stroke();
    x.fillStyle=COLORS[i%COLORS.length];x.font="12px sans-serif";
    const nm=(HELP[t]||{}).name||t;
    x.fillText(`${nm}  last ${s.last.toFixed(3)}`,40+i*170,c.height-4);
  });
}
document.getElementById("runs").innerHTML='<option value="__">loading…</option>';
document.getElementById("runs").onchange=tick;
document.getElementById("recbtn").onclick=async()=>{
  document.getElementById("recstat").textContent="starting…";
  await fetch("/api/record?run="+document.getElementById("runs").value,{method:"POST"});
  tick();
};
tick();setInterval(tick,3000);
</script></body></html>
"""
PAGE = PAGE.replace("__GROUPS_JSON__", json.dumps(CHART_GROUPS))


class Handler(BaseHTTPRequestHandler):
  def log_message(self, *a):
    pass

  def _send(self, body: bytes, ctype: str):
    self.send_response(200)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    u = urlparse(self.path)
    if u.path in ("/", "/index.html"):
      self._send(PAGE.encode(), "text/html; charset=utf-8")
    elif u.path == "/api/status":
      q = parse_qs(u.query)
      payload = status_payload((q.get("run") or [None])[0])
      self._send(json.dumps(payload).encode(), "application/json")
    elif u.path.startswith("/videos/"):
      name = os.path.basename(u.path[len("/videos/"):])
      path = os.path.join(VIDEOS_DIR, name)
      if name.endswith(".mp4") and os.path.isfile(path):
        with open(path, "rb") as f:
          self._send(f.read(), "video/mp4")
      else:
        self.send_response(404)
        self.end_headers()
    else:
      self.send_response(404)
      self.end_headers()

  def do_POST(self):
    u = urlparse(self.path)
    if u.path == "/api/record":
      q = parse_qs(u.query)
      job = start_recording((q.get("run") or [None])[0])
      self._send(json.dumps(job).encode(), "application/json")
    else:
      self.send_response(404)
      self.end_headers()


def main() -> int:
  ap = argparse.ArgumentParser(description="Friendly live G1 training dashboard")
  ap.add_argument("--port", type=int, default=6006)
  args = ap.parse_args()
  srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
  print(f"Dashboard: http://localhost:{args.port}/  (auto-updates every 3 s)")
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
