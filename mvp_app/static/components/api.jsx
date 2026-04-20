// API helpers for the MVP backend. All calls return promises that resolve to
// the parsed JSON body (or throw on non-2xx).

const API = {
  async fetchWorkspace() {
    const r = await fetch('/api/review/workspace');
    if (!r.ok) throw new Error(`workspace fetch failed: ${r.status}`);
    return r.json();
  },
  async approveObject(objectId) {
    const r = await fetch(`/api/review/objects/${objectId}`, { method: 'PATCH' });
    if (!r.ok) throw new Error(`approve failed: ${r.status}`);
    return r.json();
  },
  async deleteObject(objectId) {
    const r = await fetch(`/api/review/objects/${objectId}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(`delete failed: ${r.status}`);
    return r.json();
  },
  async uploadImage({ file, projectId, classes }) {
    const form = new FormData();
    form.append('image', file);
    form.append('project_id', projectId || 'default-project');
    form.append('classes', (classes || []).join(','));
    const r = await fetch('/api/pipeline/auto-label', { method: 'POST', body: form });
    if (!r.ok) throw new Error(`upload failed: ${r.status}`);
    return r.json();
  },
  async exportDataset({ datasetName, exportFormat, onlyValidated, splits }) {
    const form = new FormData();
    form.append('dataset_name', datasetName || 'mvp-dataset');
    form.append('export_format', exportFormat || 'yolo');
    form.append('only_validated', onlyValidated ? 'True' : 'False');
    const s = splits || { train: 80, val: 15, test: 5 };
    form.append('split_train', String(s.train));
    form.append('split_val', String(s.val));
    form.append('split_test', String(s.test));
    const r = await fetch('/api/export', { method: 'POST', body: form });
    if (!r.ok) throw new Error(`export failed: ${r.status}`);
    return r.json();
  },
  async fetchHealth() {
    const r = await fetch('/health');
    if (!r.ok) throw new Error(`health failed: ${r.status}`);
    return r.json();
  },
};

// React hook: load workspace once on mount, expose reload + optimistic
// mutate helpers so components can update UI immediately while the API
// round-trip completes.
function useWorkspace() {
  const [state, setState] = React.useState({ loading: true, error: null, data: null });

  const reload = React.useCallback(async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const data = await API.fetchWorkspace();
      setState({ loading: false, error: null, data });
    } catch (e) {
      setState({ loading: false, error: String(e), data: null });
    }
  }, []);

  React.useEffect(() => { reload(); }, [reload]);

  const mutateObject = React.useCallback((sourceId, objectId, patch) => {
    setState(prev => {
      if (!prev.data) return prev;
      const sources = prev.data.sources.map(src => {
        if (src.id !== sourceId) return src;
        return {
          ...src,
          objects: src.objects.map(o => o.object_id === objectId ? { ...o, ...patch } : o),
        };
      });
      return { ...prev, data: { ...prev.data, sources } };
    });
  }, []);

  const removeObject = React.useCallback((sourceId, objectId) => {
    setState(prev => {
      if (!prev.data) return prev;
      const sources = prev.data.sources.map(src => {
        if (src.id !== sourceId) return src;
        return { ...src, objects: src.objects.filter(o => o.object_id !== objectId) };
      });
      return { ...prev, data: { ...prev.data, sources } };
    });
  }, []);

  return { ...state, reload, mutateObject, removeObject };
}
