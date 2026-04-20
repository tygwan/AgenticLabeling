// Home: activity feed + ingest CTA + recent sources
const HomeScreen = ({ onGoReview, onRunPipeline }) => {
  const [dragOver, setDragOver] = React.useState(false);
  const [projectId, setProjectId] = React.useState('urban-mobility-v3');
  const [classes, setClasses] = React.useState('person, car, traffic_light, bicycle');
  const [running, setRunning] = React.useState(false);
  const [progress, setProgress] = React.useState(0);

  const kick = () => {
    setRunning(true);
    setProgress(0);
    const t = setInterval(() => {
      setProgress(p => {
        if (p >= 1) { clearInterval(t); setRunning(false); onRunPipeline?.(); return 1; }
        return p + 0.04;
      });
    }, 80);
  };

  return (
    <div className="home-grid">
      <section className="ingest-card">
        <div className="card-head">
          <div>
            <div className="card-eyebrow">Ingest</div>
            <h2 className="card-title">Run auto-labeling pipeline</h2>
          </div>
          <div className="pipeline-tag">
            <span>Florence-2</span>
            <IconArrowRight size={12} />
            <span>SAM2</span>
          </div>
        </div>
        <div
          className={`dropzone ${dragOver ? 'over' : ''} ${running ? 'running' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); kick(); }}
        >
          {!running ? (
            <>
              <IconUpload size={28} />
              <div className="drop-title">Drop images here</div>
              <div className="drop-sub">or <button className="link-btn" onClick={kick}>browse files</button> · .jpg .png .webp up to 50MB</div>
            </>
          ) : (
            <div className="run-state">
              <div className="run-stages">
                <Stage active={progress > 0}   done={progress > 0.2} label="Upload" />
                <Stage active={progress > 0.2} done={progress > 0.55} label="Detection (Florence-2)" />
                <Stage active={progress > 0.55} done={progress > 0.9}  label="Segmentation (SAM2)" />
                <Stage active={progress > 0.9}  done={progress >= 1}   label="Registry write" />
              </div>
              <div className="progress-bar"><div className="progress-fill" style={{ width: `${progress * 100}%` }} /></div>
              <div className="run-hint">Processing 1 image · ~{((1 - progress) * 6).toFixed(1)}s remaining</div>
            </div>
          )}
        </div>
        <div className="ingest-form">
          <div className="form-field">
            <label>project_id</label>
            <input value={projectId} onChange={e => setProjectId(e.target.value)} disabled={running} />
          </div>
          <div className="form-field">
            <label>classes <span className="hint">comma separated</span></label>
            <input value={classes} onChange={e => setClasses(e.target.value)} disabled={running} />
          </div>
          <button className="btn primary" onClick={kick} disabled={running}>
            {running ? <><IconSpinner size={14} /> Running…</> : <>Run pipeline <Kbd>⏎</Kbd></>}
          </button>
        </div>
      </section>

      <section className="stats-card">
        <div className="card-eyebrow">Today</div>
        <div className="stat-grid">
          <BigStat v="384" k="Sources processed" />
          <BigStat v="2,147" k="Objects detected" />
          <BigStat v="1,402" k="Validated" trend="+12%" />
          <BigStat v="97.2%" k="Pipeline uptime" />
        </div>
        <div className="mini-chart">
          {Array.from({ length: 24 }).map((_, i) => {
            const h = 12 + Math.sin(i * 0.7) * 10 + Math.cos(i * 0.3) * 8 + Math.random() * 20;
            return <div key={i} className="bar" style={{ height: `${h}%` }} />;
          })}
        </div>
        <div className="mini-chart-axis">
          <span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>24:00</span>
        </div>
      </section>

      <section className="runs-card">
        <div className="card-head">
          <h3 className="card-title sm">Recent runs</h3>
          <button className="btn ghost sm">View all <IconChevRight size={12} /></button>
        </div>
        <div className="run-table">
          <div className="run-head">
            <span>Run</span><span>Project</span><span>Sources</span><span>Duration</span><span>Status</span>
          </div>
          {RECENT_RUNS.map(r => (
            <div key={r.id} className="run-row">
              <span className="mono">{r.id}</span>
              <span>{r.project}</span>
              <span>{r.source_count}</span>
              <span>{r.duration}</span>
              <span>
                {r.status === 'running'
                  ? <span className="running-pill"><IconSpinner size={10} /> running · {Math.round(r.progress*100)}%</span>
                  : <StatusPill status={r.status} />
                }
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="projects-card">
        <div className="card-head">
          <h3 className="card-title sm">Projects</h3>
          <button className="btn ghost sm">New project</button>
        </div>
        {PROJECTS.map(p => (
          <div key={p.id} className="project-row">
            <div>
              <div className="project-name">{p.name}</div>
              <div className="project-sub">{p.classes} classes · {p.sources.toLocaleString()} sources</div>
            </div>
            <div className="project-progress">
              <div className="project-bar"><div className="project-fill" style={{ width: `${(p.validated/p.sources)*100}%` }} /></div>
              <div className="project-pct">{Math.round((p.validated/p.sources)*100)}%</div>
            </div>
          </div>
        ))}
        <button className="resume-cta" onClick={onGoReview}>
          Resume review <IconArrowRight size={14} />
        </button>
      </section>
    </div>
  );
};

const Stage = ({ active, done, label }) => (
  <div className={`stage ${active ? 'active' : ''} ${done ? 'done' : ''}`}>
    <div className="stage-dot">
      {done ? <IconCheck size={10} stroke={3} /> : active ? <IconSpinner size={10} /> : null}
    </div>
    <div className="stage-label">{label}</div>
  </div>
);

const BigStat = ({ v, k, trend }) => (
  <div className="big-stat">
    <div className="big-stat-v">{v}</div>
    <div className="big-stat-k">{k}{trend && <span className="trend">{trend}</span>}</div>
  </div>
);

Object.assign(window, { HomeScreen });
