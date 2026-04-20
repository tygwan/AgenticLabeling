// Home: activity feed + ingest CTA + recent sources
const HomeScreen = ({ onGoReview, onRunPipeline, workspace }) => {
  const [dragOver, setDragOver] = React.useState(false);
  const [projectId, setProjectId] = React.useState('default-project');
  const [classes, setClasses] = React.useState('person, car, road');
  const [running, setRunning] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [lastError, setLastError] = React.useState(null);
  const fileInputRef = React.useRef(null);

  const stats = workspace?.stats || { sources: 0, objects: 0, validated_objects: 0 };
  const projects = workspace?.projects || [];

  const runPipeline = React.useCallback(async (files) => {
    if (!files || files.length === 0) return;
    setRunning(true); setProgress(0); setLastError(null);
    const ticker = setInterval(() => {
      setProgress(p => (p < 0.85 ? p + 0.02 : p));
    }, 80);
    try {
      const classList = classes.split(',').map(c => c.trim()).filter(Boolean);
      for (const file of files) {
        await API.uploadImage({ file, projectId, classes: classList });
      }
      setProgress(1);
      clearInterval(ticker);
      onRunPipeline?.();
    } catch (e) {
      clearInterval(ticker);
      setLastError(String(e));
    } finally {
      setTimeout(() => { setRunning(false); setProgress(0); }, 600);
    }
  }, [projectId, classes, onRunPipeline]);

  const kick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const onPickFiles = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) runPipeline(files);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length > 0) runPipeline(files);
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
            <span>SAM3</span>
          </div>
        </div>
        <div
          className={`dropzone ${dragOver ? 'over' : ''} ${running ? 'running' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <input ref={fileInputRef} type="file" accept="image/*" multiple hidden onChange={onPickFiles} />
          {!running ? (
            <>
              <IconUpload size={28} />
              <div className="drop-title">Drop images here</div>
              <div className="drop-sub">or <button className="link-btn" onClick={kick}>browse files</button> · .jpg .png .webp up to 50MB</div>
              {lastError && <div className="drop-sub" style={{ color: 'var(--c-err)' }}>Last upload failed: {lastError}</div>}
            </>
          ) : (
            <div className="run-state">
              <div className="run-stages">
                <Stage active={progress > 0}    done={progress > 0.2}  label="Upload" />
                <Stage active={progress > 0.2}  done={progress > 0.55} label="Detection (Florence-2)" />
                <Stage active={progress > 0.55} done={progress > 0.9}  label="Segmentation (SAM3)" />
                <Stage active={progress > 0.9}  done={progress >= 1}   label="Registry write" />
              </div>
              <div className="progress-bar"><div className="progress-fill" style={{ width: `${progress * 100}%` }} /></div>
              <div className="run-hint">Uploading · waiting for pipeline</div>
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
        <div className="card-eyebrow">Current workspace</div>
        <div className="stat-grid">
          <BigStat v={stats.sources?.toLocaleString() ?? '0'} k="Sources" />
          <BigStat v={stats.objects?.toLocaleString() ?? '0'} k="Objects" />
          <BigStat v={stats.validated_objects?.toLocaleString() ?? '0'} k="Validated" />
          <BigStat v={stats.objects > 0 ? `${Math.round((stats.validated_objects / stats.objects) * 100)}%` : '—'} k="Validation rate" />
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
        </div>
        <div className="run-table">
          <div className="run-head">
            <span>Run</span><span>Project</span><span>Detections</span><span>Duration</span><span>Status</span>
          </div>
          {(workspace?.recent_runs || []).length === 0 ? (
            <div className="run-row" style={{ color: 'var(--c-muted)' }}>
              <span colSpan={5} style={{ gridColumn: '1 / -1' }}>No runs yet — upload an image to run the pipeline.</span>
            </div>
          ) : (workspace.recent_runs || []).map(r => {
            const dur = r.duration_ms != null ? (r.duration_ms > 1000 ? `${(r.duration_ms / 1000).toFixed(1)}s` : `${r.duration_ms}ms`) : '—';
            return (
              <div key={r.run_id} className="run-row" title={r.error || ''}>
                <span className="mono">{r.run_id}</span>
                <span>{r.project_id || 'default'}</span>
                <span>{r.detections ?? '—'}</span>
                <span>{dur}</span>
                <span>
                  {r.status === 'running'
                    ? <span className="running-pill"><IconSpinner size={10} /> running</span>
                    : <StatusPill status={r.status === 'completed' ? 'validated' : (r.status === 'failed' ? 'failed' : r.status)} />
                  }
                </span>
              </div>
            );
          })}
        </div>
      </section>

      <section className="projects-card">
        <div className="card-head">
          <h3 className="card-title sm">Projects</h3>
        </div>
        {projects.length === 0 ? (
          <div className="project-row" style={{ color: 'var(--c-muted)' }}>No projects yet — upload an image to create one.</div>
        ) : projects.map(p => {
          const classCount = Array.isArray(p.classes) ? p.classes.length : 0;
          const pct = p.sources > 0 ? (p.validated / p.sources) * 100 : 0;
          return (
            <div key={p.id} className="project-row">
              <div>
                <div className="project-name">{p.name}</div>
                <div className="project-sub">{classCount} classes · {p.sources.toLocaleString()} sources</div>
              </div>
              <div className="project-progress">
                <div className="project-bar"><div className="project-fill" style={{ width: `${pct}%` }} /></div>
                <div className="project-pct">{Math.round(pct)}%</div>
              </div>
            </div>
          );
        })}
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
