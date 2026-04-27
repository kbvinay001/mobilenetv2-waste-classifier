"""ui_styles.py — GarbageSort AI Design System (adapted from SafeGuard AI)"""


def get_css() -> str:
    return """<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg:#060913; --bg2:#0A0F1E; --card:rgba(14,20,36,.62);
  --bdr:rgba(255,255,255,.06); --bdrh:rgba(255,255,255,.14);
  --green:#22C55E; --cyan:#22D3EE; --orange:#F97316;
  --purple:#A78BFA; --yellow:#FBBF24; --blue:#3B82F6; --red:#F87171;
  --txt:#F1F5F9; --txt2:#94A3B8; --muted:#64748B;
  --blur:blur(18px); --ease:.28s cubic-bezier(.4,0,.2,1);
}

html,body,[class*="css"]{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;
  -webkit-font-smoothing:antialiased;
}
.stApp{ background:var(--bg) !important; min-height:100vh; }
.main .block-container{ padding:0 !important; max-width:100% !important; }
section[data-testid="stSidebar"],
button[data-testid="stSidebarCollapsedControl"],
header[data-testid="stHeader"],
.stDeployButton,footer,#MainMenu{ display:none !important; }
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:rgba(34,197,94,.3);border-radius:4px}

/* ── Animations ── */
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes pulse-glow{0%,100%{box-shadow:0 0 4px #22C55E}50%{box-shadow:0 0 18px #22C55E,0 0 36px rgba(34,197,94,.3)}}
@keyframes pulse-glow-warn{0%,100%{box-shadow:0 0 4px #FBBF24}50%{box-shadow:0 0 14px #FBBF24,0 0 28px rgba(251,191,36,.3)}}
@keyframes pulse-glow-alert{0%,100%{box-shadow:0 0 6px #F87171}50%{box-shadow:0 0 20px #F87171,0 0 40px rgba(248,113,113,.4)}}
@keyframes ripple{to{transform:scale(4);opacity:0}}
@keyframes clickPop{0%{transform:scale(1)}50%{transform:scale(.97)}100%{transform:scale(1)}}
@keyframes shimmer{0%{background-position:200% center}100%{background-position:-200% center}}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes confBar{from{width:0}to{width:var(--bar-w,0%)}}

/* ── Navbar ── */
.gs-nav{
  position:sticky;top:0;z-index:999;
  background:rgba(6,9,19,.88);
  backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);
  border-bottom:1px solid var(--bdr);
}
.gs-nav-inner{
  max-width:1280px;margin:0 auto;padding:0 2.5rem;height:66px;
  display:flex;align-items:center;justify-content:space-between;
}
.gs-logo{display:flex;align-items:center;gap:11px;font-weight:800;font-size:1.2rem;color:#fff;letter-spacing:-.02em}
.gs-logo-icon{
  width:38px;height:38px;
  background:linear-gradient(135deg,var(--green),var(--cyan));
  border-radius:10px;display:flex;align-items:center;justify-content:center;
  color:#fff;font-size:1.1rem;
}
.gs-badge{font-size:.63rem;font-weight:700;letter-spacing:.07em;padding:3px 9px;border-radius:9999px;background:rgba(34,197,94,.15);color:#4ADE80}
.gs-nav-right{display:flex;align-items:center;gap:.8rem}
.pill-on{
  display:inline-flex;align-items:center;gap:5px;
  padding:.3rem .9rem;border-radius:9999px;font-size:.74rem;font-weight:600;
  background:rgba(34,197,94,.12);color:#4ADE80;border:1px solid rgba(34,197,94,.25);
}

/* ── Back button ── */
.gs-backrow{padding:.8rem 2.5rem 0;max-width:1280px;margin:0 auto}
.gs-backrow .stButton>button{
  background:rgba(14,20,36,.5) !important;border:1px solid var(--bdr) !important;
  color:var(--txt2) !important;border-radius:10px !important;font-size:.85rem !important;font-weight:500 !important;
}
.gs-backrow .stButton>button:hover{background:rgba(255,255,255,.06) !important;color:#fff !important}

/* ── Hero ── */
.gs-hero{
  max-width:1280px;margin:0 auto;
  padding:4.5rem 2.5rem 2rem;
}
.gs-welcome-badge{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.25);
  padding:8px 18px;border-radius:9999px;
  color:#4ADE80;font-size:.82rem;font-weight:600;margin-bottom:1.6rem;
  animation:fadeIn .6s ease both;
}
.pulse-dot{width:8px;height:8px;border-radius:50%;background:#4ADE80;flex-shrink:0;animation:pulse-glow 2s infinite}

.gs-hero-title{
  font-family:'Playfair Display',Georgia,serif !important;
  font-size:clamp(3rem,6vw,5rem) !important;
  font-weight:800 !important;line-height:1.05 !important;
  letter-spacing:-.03em !important;color:#fff !important;
  margin-bottom:1.1rem !important;
  animation:fadeSlideUp .6s .1s ease both;
}
.text-gradient{
  background:linear-gradient(135deg,var(--green),var(--cyan));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.gs-hero-sub{
  color:var(--muted);font-size:1.05rem;max-width:580px;line-height:1.7;
  animation:fadeSlideUp .6s .2s ease both;
}

/* ── Wrapper ── */
.gs-wrap{max-width:1280px;margin:0 auto;padding:0 2.5rem 4rem}

/* ── Stat row ── */
.gs-stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:3.5rem}
.gs-stat-card{
  background:var(--card);backdrop-filter:var(--blur);
  border:1px solid var(--bdr);border-radius:18px;
  padding:26px 22px;text-align:center;
  transition:border-color var(--ease),box-shadow var(--ease),transform var(--ease);
  animation:fadeSlideUp .5s ease both;
  position:relative;overflow:hidden;
}
.gs-stat-card::before{
  content:'';position:absolute;inset:-1px;border-radius:18px;
  background:linear-gradient(135deg,rgba(34,197,94,.4),rgba(34,211,238,.2),rgba(167,139,250,.3));
  opacity:0;z-index:-1;transition:opacity var(--ease);
}
.gs-stat-card:hover::before{opacity:1}
.gs-stat-card:hover{border-color:transparent;box-shadow:0 0 36px rgba(34,197,94,.12);transform:translateY(-3px)}
.gs-stat-icon{margin-bottom:12px;font-size:1.8rem}
.gs-stat-value{font-size:1.55rem;font-weight:800;letter-spacing:-.03em;color:#fff;margin-bottom:5px}
.gs-stat-label{font-size:.67rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.09em}

/* ── Section heading ── */
.gs-section-title{font-size:1.15rem;font-weight:700;color:var(--txt2);letter-spacing:.02em;margin-bottom:1.5rem}

/* ── Module cards ── */
.mod-card-visual{
  background:var(--card);backdrop-filter:var(--blur);
  border:1px solid var(--bdr);border-radius:20px 20px 0 0;
  padding:30px 28px 22px;
  transition:border-color var(--ease),box-shadow var(--ease),transform var(--ease);
  cursor:pointer;
  animation:fadeSlideUp .55s ease both;
  position:relative;overflow:hidden;
}
.mod-card-visual::before{
  content:'';position:absolute;top:0;left:15%;right:15%;height:2px;
  background:var(--accent,#22C55E);border-radius:0 0 6px 6px;
  opacity:0;filter:blur(3px);transition:opacity var(--ease);
}
.mod-card-visual:hover::before{opacity:.8}
.mod-card-visual:hover{transform:translateY(-5px);border-color:var(--accent,#22C55E)}
.mod-icon-wrap{
  width:52px;height:52px;border-radius:14px;
  display:flex;align-items:center;justify-content:center;
  margin-bottom:18px;
  background:rgba(255,255,255,.05);
  font-size:1.6rem;
}
.mod-card-title{font-size:1.15rem;font-weight:700;color:#fff;margin-bottom:10px;letter-spacing:-.01em}
.mod-card-desc{font-size:.88rem;color:var(--muted);line-height:1.65;min-height:52px;margin-bottom:0}

.mod-explore .stButton>button{
  border-radius:0 0 20px 20px !important;
  border:1px solid var(--bdr) !important;border-top:1px solid rgba(255,255,255,.04) !important;
  background:rgba(14,20,36,.75) !important;
  color:var(--accent,#22C55E) !important;
  font-size:.85rem !important;font-weight:600 !important;
  height:44px !important;
  transition:all var(--ease) !important;
  position:relative;overflow:hidden;
}
.mod-explore .stButton>button:hover{
  background:rgba(14,20,36,.95) !important;
  box-shadow:0 0 15px rgba(34,211,238,.2),0 0 30px rgba(34,211,238,.08) !important;
  text-shadow:0 0 8px rgba(34,211,238,.4);
  filter:brightness(1.15);
}
.mod-explore .stButton>button:active{animation:clickPop .2s ease !important}

/* Per-accent colours */
.acc-green{--accent:#22C55E} .acc-cyan{--accent:#22D3EE}
.acc-orange{--accent:#F97316} .acc-purple{--accent:#A78BFA}
.acc-blue{--accent:#3B82F6}  .acc-yellow{--accent:#FBBF24}

/* ── Info boxes ── */
.gs-box{background:var(--card);backdrop-filter:var(--blur);border:1px solid var(--bdr);border-radius:16px;padding:1.2rem 1.5rem;margin-bottom:.9rem;animation:fadeIn .4s ease}
.gs-box.info{border-left:3px solid var(--blue)}
.gs-box.good{border-left:3px solid var(--green)}
.gs-box.warn{border-left:3px solid var(--orange)}
.gs-box.eco{border-left:3px solid var(--green);background:rgba(34,197,94,.05)}

/* ── Confidence bars ── */
.conf-bar-wrap{margin-bottom:10px}
.conf-bar-label{display:flex;justify-content:space-between;font-size:.82rem;color:var(--txt2);margin-bottom:4px}
.conf-bar-label strong{color:#fff}
.conf-bar-track{background:rgba(255,255,255,.06);border-radius:9999px;height:8px;overflow:hidden}
.conf-bar-fill{height:100%;border-radius:9999px;background:linear-gradient(90deg,var(--green),var(--cyan));transition:width .6s cubic-bezier(.4,0,.2,1)}

/* ── Class badge ── */
.class-badge{
  display:inline-flex;align-items:center;gap:8px;
  padding:10px 22px;border-radius:9999px;font-size:1rem;font-weight:700;
  border:1px solid;margin-bottom:1rem;
}

/* ── Result card ── */
.result-card{
  background:var(--card);backdrop-filter:var(--blur);
  border:1px solid var(--bdr);border-radius:18px;padding:24px;
  animation:fadeSlideUp .4s ease both;
}

/* ── Recycling tip ── */
.recycle-card{
  background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.2);
  border-radius:16px;padding:20px 24px;
  animation:fadeSlideUp .5s .1s ease both;
}
.recycle-icon{font-size:2rem;margin-bottom:8px}
.recycle-title{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#4ADE80;margin-bottom:6px}
.recycle-text{font-size:.9rem;color:var(--txt2);line-height:1.6}

/* ── Live feed HUD ── */
.gs-hud-status{
  display:flex;align-items:center;gap:12px;
  padding:.4rem .8rem;margin-top:.4rem;
  background:rgba(6,9,19,.7);border:1px solid rgba(255,255,255,.06);
  border-radius:8px;
}
.gs-stream-live{
  font-size:.78rem;font-weight:700;letter-spacing:.08em;
  color:#4ADE80;animation:pulse-glow 2s infinite;
  padding:2px 8px;border-radius:9999px;
  background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.25);
}
.gs-stream-reconnecting{
  font-size:.78rem;font-weight:700;letter-spacing:.05em;
  color:#FBBF24;animation:pulse-glow-warn 1.2s infinite;
  padding:2px 8px;border-radius:9999px;
  background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.25);
}

/* ── Buttons ── */
.stButton>button{
  background:rgba(34,197,94,.12) !important;color:var(--green) !important;
  border:1px solid rgba(34,197,94,.25) !important;border-radius:10px !important;
  font-family:'Inter',sans-serif !important;font-size:.85rem !important;font-weight:600 !important;
  transition:all var(--ease) !important;
}
.stButton>button:hover{
  background:rgba(34,197,94,.22) !important;
  transform:translateY(-1px) !important;
  box-shadow:0 0 15px rgba(34,211,238,.25),0 0 30px rgba(34,211,238,.1) !important;
  border-color:rgba(34,211,238,.4) !important;
  text-shadow:0 0 8px rgba(34,211,238,.5);
}
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#16A34A,#059669) !important;color:#fff !important;
  border-color:transparent !important;
  box-shadow:0 4px 16px rgba(22,163,74,.35) !important;
}
.stButton>button[kind="primary"]:hover{
  box-shadow:0 0 20px rgba(34,197,94,.5),0 0 40px rgba(34,197,94,.2) !important;
  transform:translateY(-2px) !important;
}

div[data-testid="metric-container"]{background:var(--card) !important;border:1px solid var(--bdr) !important;border-radius:14px !important;padding:.8rem 1rem !important}
div[data-testid="metric-container"] [data-testid="metric-value"]{color:#fff !important;font-size:1.4rem !important;font-weight:800 !important}
.stTextInput input,.stTextArea textarea{background:rgba(14,20,36,.7) !important;border:1px solid var(--bdr) !important;color:var(--txt) !important;border-radius:10px !important}
div[data-baseweb="select"] div{background:rgba(14,20,36,.8) !important;border-color:var(--bdr) !important;color:var(--txt) !important;border-radius:10px !important}
.stFileUploader{background:rgba(14,20,36,.5) !important;border:2px dashed rgba(34,197,94,.3) !important;border-radius:14px !important}
.streamlit-expanderHeader{background:rgba(14,20,36,.7) !important;border:1px solid var(--bdr) !important;border-radius:12px !important;font-size:.85rem !important;font-weight:600 !important;color:var(--txt2) !important}
.stProgress>div>div>div{background:linear-gradient(90deg,var(--green),var(--cyan)) !important}
div[data-testid="stToggle"] span{background:rgba(34,197,94,.3) !important}

/* ── Log ── */
.gs-log{background:#020509;border:1px solid rgba(34,197,94,.15);border-radius:14px;padding:.9rem 1.1rem;font-family:'JetBrains Mono','Fira Code',monospace;font-size:.74rem;color:#4a7a6a;max-height:300px;overflow-y:auto;line-height:1.9}
.li{color:#60A5FA}.ls{color:#4ADE80}.lw{color:#FBBF24}.la{color:#F87171}.lt{color:var(--muted)}

/* ── Table ── */
.gs-table{width:100%;border-collapse:collapse;font-size:.85rem}
.gs-table th{background:rgba(14,20,36,.8);color:var(--muted);padding:.7rem 1rem;text-align:left;font-size:.67rem;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid var(--bdr)}
.gs-table td{padding:.6rem 1rem;border-bottom:1px solid rgba(255,255,255,.025);color:var(--txt2)}
.gs-table tr:hover td{background:rgba(34,197,94,.04)}

/* ── Footer ── */
.gs-footer{text-align:center;padding:2rem 1rem;font-size:.74rem;color:var(--muted);border-top:1px solid var(--bdr);margin-top:2rem}
.gs-footer span{color:var(--green)}

/* ── Responsive ── */
@media(max-width:900px){.gs-stats-row{grid-template-columns:repeat(2,1fr)}.gs-hero{padding:3rem 1.5rem 2rem}.gs-wrap{padding:0 1.5rem 3rem}}
@media(max-width:640px){.gs-hero-title{font-size:2.5rem !important}}
</style>"""


CONSTELLATION_JS = """
<script>
(function(){
  var pd=window.parent.document;
  var old=pd.getElementById('gs-cv'); if(old)old.remove();
  var c=pd.createElement('canvas'); c.id='gs-cv';
  c.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;opacity:0.35;';
  pd.body.appendChild(c);
  var ctx=c.getContext('2d'),anim,pts=[];
  function resize(){c.width=window.parent.innerWidth;c.height=window.parent.innerHeight;}
  resize(); window.parent.addEventListener('resize',resize);
  for(var i=0;i<60;i++) pts.push({x:Math.random()*c.width,y:Math.random()*c.height,vx:(Math.random()-.5)*.28,vy:(Math.random()-.5)*.28,r:Math.random()*1.2+.4});
  function draw(){
    ctx.clearRect(0,0,c.width,c.height);
    pts.forEach(function(p){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>c.width)p.vx*=-1;if(p.y<0||p.y>c.height)p.vy*=-1;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='rgba(34,197,94,.55)';ctx.fill();});
    for(var i=0;i<pts.length;i++)for(var j=i+1;j<pts.length;j++){var dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<125){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle='rgba(34,197,94,'+(0.07*(1-d/125))+')';ctx.lineWidth=1;ctx.stroke();}}
    anim=window.parent.requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""

CLASS_COLORS = {
    "Battery":   "#F87171",   # red — hazardous
    "Cardboard": "#FBBF24",   # yellow
    "Clothes":   "#A78BFA",   # purple
    "Glass":     "#22D3EE",   # cyan
    "Metal":     "#60A5FA",   # blue
    "Paper":     "#4ADE80",   # green
    "Plastic":   "#F97316",   # orange
}

CLASS_EMOJIS = {
    "Battery":   "🔋",
    "Cardboard": "📦",
    "Clothes":   "👕",
    "Glass":     "🥛",
    "Metal":     "🥫",
    "Paper":     "📄",
    "Plastic":   "🧴",
}
