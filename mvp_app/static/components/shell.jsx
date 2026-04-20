// App shell: left icon nav, top bar, main area. Theme + density + accent driven by context.

const AppShell = ({ route, setRoute, children, onOpenTweaks, tweaksOn, onOpenShortcuts }) => {
  const NAV = [
    { id: 'home',    label: 'Home',     Icon: IconHome },
    { id: 'review',  label: 'Review',   Icon: IconReview },
    { id: 'export',  label: 'Export',   Icon: IconExport },
    { id: 'settings', label: 'Settings', Icon: IconSettings },
  ];
  return (
    <div className="app-shell">
      <aside className="nav-rail">
        <div className="nav-brand" title="AgenticLabeling">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="3" width="8" height="8" rx="1" stroke="currentColor" strokeWidth="1.8"/>
            <rect x="13" y="3" width="8" height="8" rx="1" stroke="currentColor" strokeWidth="1.8" strokeDasharray="2 2"/>
            <rect x="3" y="13" width="8" height="8" rx="1" stroke="currentColor" strokeWidth="1.8" strokeDasharray="2 2"/>
            <rect x="13" y="13" width="8" height="8" rx="1" fill="currentColor"/>
          </svg>
        </div>
        <div className="nav-items">
          {NAV.map(({ id, label, Icon }) => (
            <button
              key={id}
              className={`nav-btn ${route === id ? 'active' : ''}`}
              onClick={() => setRoute(id)}
              title={label}
            >
              <Icon size={20} />
              <span className="nav-label">{label}</span>
            </button>
          ))}
        </div>
        <div className="nav-foot">
          <button className="nav-btn" title="Shortcuts (?)" onClick={onOpenShortcuts}>
            <IconKeyboard size={20} />
          </button>
          <button className={`nav-btn ${tweaksOn ? 'active' : ''}`} title="Tweaks" onClick={onOpenTweaks}>
            <IconTweak size={20} />
          </button>
        </div>
      </aside>
      <div className="app-main">
        {children}
      </div>
    </div>
  );
};

const TopBar = ({ title, subtitle, crumbs, right }) => (
  <header className="topbar">
    <div className="topbar-left">
      {crumbs && (
        <div className="crumbs">
          {crumbs.map((c, i) => (
            <React.Fragment key={i}>
              <span className="crumb">{c}</span>
              {i < crumbs.length - 1 && <span className="crumb-sep">/</span>}
            </React.Fragment>
          ))}
        </div>
      )}
      <h1 className="topbar-title">{title}</h1>
      {subtitle && <div className="topbar-sub">{subtitle}</div>}
    </div>
    <div className="topbar-right">{right}</div>
  </header>
);

const StatusPill = ({ status }) => {
  const map = {
    validated:  { label: 'Validated',   color: 'var(--c-ok)' },
    in_review:  { label: 'In review',   color: 'var(--c-accent)' },
    pending:    { label: 'Pending',     color: 'var(--c-warn)' },
    failed:     { label: 'Failed',      color: 'var(--c-err)' },
    running:    { label: 'Running',     color: 'var(--c-accent)' },
    completed:  { label: 'Completed',   color: 'var(--c-ok)' },
    approved:   { label: 'Approved',    color: 'var(--c-ok)' },
    deleted:    { label: 'Deleted',     color: 'var(--c-err)' },
  };
  const s = map[status] || { label: status, color: 'var(--c-muted)' };
  return (
    <span className="pill" style={{ '--pill-color': s.color }}>
      <IconDot size={6} color={s.color} />
      {s.label}
    </span>
  );
};

const Kbd = ({ children }) => <kbd className="kbd">{children}</kbd>;

Object.assign(window, { AppShell, TopBar, StatusPill, Kbd });
