// Mock data for the prototype.

const CATEGORY_COLORS = {
  person:     '#ef4444',
  car:        '#3b82f6',
  truck:      '#8b5cf6',
  bicycle:    '#f59e0b',
  motorcycle: '#ec4899',
  traffic_light: '#10b981',
  stop_sign:  '#f97316',
  dog:        '#14b8a6',
  backpack:   '#a855f7',
  handbag:    '#06b6d4',
  bench:      '#84cc16',
  pallet:     '#eab308',
  forklift:   '#22c55e',
  worker:     '#ef4444',
  shelf:      '#64748b',
  box:        '#f59e0b',
};

// Palette of real, credible street photos from Unsplash for domain realism.
// (Served via their image CDN; placeholder fallback gradient on err.)
const SOURCES = [
  {
    id: 'src_8f3a2c19b0e4',
    file_name: 'intersection_5th_ave.jpg',
    url: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=1600&auto=format&fit=crop',
    width: 1600, height: 1067,
    status: 'in_review',
    project: 'urban-mobility-v3',
    classes: ['person', 'car', 'traffic_light', 'bicycle'],
    uploaded_at: '2026-04-20T08:42:00Z',
    objects: [
      { object_id: 'obj_a1b2c3', category: 'car',           bbox: [0.12, 0.52, 0.22, 0.28], confidence: 0.94, validated: null },
      { object_id: 'obj_a1b2c4', category: 'car',           bbox: [0.38, 0.48, 0.18, 0.24], confidence: 0.91, validated: null },
      { object_id: 'obj_a1b2c5', category: 'car',           bbox: [0.62, 0.50, 0.20, 0.26], confidence: 0.88, validated: 'approved' },
      { object_id: 'obj_a1b2c6', category: 'person',        bbox: [0.28, 0.44, 0.05, 0.18], confidence: 0.82, validated: null },
      { object_id: 'obj_a1b2c7', category: 'person',        bbox: [0.46, 0.46, 0.04, 0.16], confidence: 0.79, validated: null },
      { object_id: 'obj_a1b2c8', category: 'traffic_light', bbox: [0.71, 0.18, 0.04, 0.12], confidence: 0.96, validated: 'approved' },
      { object_id: 'obj_a1b2c9', category: 'traffic_light', bbox: [0.18, 0.14, 0.04, 0.11], confidence: 0.93, validated: null },
      { object_id: 'obj_a1b2d0', category: 'bicycle',       bbox: [0.84, 0.62, 0.10, 0.20], confidence: 0.68, validated: null },
      { object_id: 'obj_a1b2d1', category: 'person',        bbox: [0.88, 0.46, 0.04, 0.22], confidence: 0.54, validated: 'deleted' },
    ],
  },
  {
    id: 'src_2d7e4f80a16c',
    file_name: 'warehouse_bay_12.jpg',
    url: 'https://images.unsplash.com/photo-1553413077-190dd305871c?w=1600&auto=format&fit=crop',
    width: 1600, height: 1067,
    status: 'pending',
    project: 'warehouse-ops',
    classes: ['pallet', 'forklift', 'worker', 'box'],
    uploaded_at: '2026-04-20T08:55:00Z',
    objects: [
      { object_id: 'obj_b2c3d4', category: 'pallet',   bbox: [0.08, 0.62, 0.16, 0.22], confidence: 0.89, validated: null },
      { object_id: 'obj_b2c3d5', category: 'pallet',   bbox: [0.30, 0.64, 0.18, 0.20], confidence: 0.86, validated: null },
      { object_id: 'obj_b2c3d6', category: 'forklift', bbox: [0.52, 0.40, 0.22, 0.38], confidence: 0.92, validated: null },
      { object_id: 'obj_b2c3d7', category: 'box',      bbox: [0.76, 0.50, 0.12, 0.18], confidence: 0.74, validated: null },
      { object_id: 'obj_b2c3d8', category: 'worker',   bbox: [0.58, 0.22, 0.06, 0.28], confidence: 0.81, validated: null },
    ],
  },
  {
    id: 'src_9c1f6b3a8e72',
    file_name: 'parking_lot_night.jpg',
    url: 'https://images.unsplash.com/photo-1506521781263-d8422e82f27a?w=1600&auto=format&fit=crop',
    width: 1600, height: 1067,
    status: 'validated',
    project: 'urban-mobility-v3',
    classes: ['car', 'person'],
    uploaded_at: '2026-04-20T07:30:00Z',
    objects: [
      { object_id: 'obj_c3d4e5', category: 'car', bbox: [0.10, 0.50, 0.25, 0.30], confidence: 0.76, validated: 'approved' },
      { object_id: 'obj_c3d4e6', category: 'car', bbox: [0.42, 0.52, 0.22, 0.28], confidence: 0.82, validated: 'approved' },
      { object_id: 'obj_c3d4e7', category: 'car', bbox: [0.68, 0.50, 0.24, 0.30], confidence: 0.71, validated: 'approved' },
    ],
  },
  {
    id: 'src_4b8d2a6f1c93',
    file_name: 'busy_crosswalk_dusk.jpg',
    url: 'https://images.unsplash.com/photo-1444723121867-7a241cacace9?w=1600&auto=format&fit=crop',
    width: 1600, height: 1067,
    status: 'pending',
    project: 'urban-mobility-v3',
    classes: ['person', 'car', 'bicycle'],
    uploaded_at: '2026-04-20T08:10:00Z',
    objects: [
      { object_id: 'obj_d4e5f6', category: 'person', bbox: [0.18, 0.42, 0.08, 0.40], confidence: 0.85, validated: null },
      { object_id: 'obj_d4e5f7', category: 'person', bbox: [0.32, 0.44, 0.07, 0.38], confidence: 0.80, validated: null },
      { object_id: 'obj_d4e5f8', category: 'person', bbox: [0.48, 0.42, 0.08, 0.42], confidence: 0.88, validated: null },
      { object_id: 'obj_d4e5f9', category: 'car',    bbox: [0.70, 0.54, 0.24, 0.26], confidence: 0.79, validated: null },
    ],
  },
  {
    id: 'src_5e9d3b7f2d04',
    file_name: 'low_light_alley.jpg',
    url: 'https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=1600&auto=format&fit=crop',
    width: 1600, height: 1067,
    status: 'failed',
    project: 'urban-mobility-v3',
    classes: ['person', 'dog'],
    uploaded_at: '2026-04-20T08:01:00Z',
    objects: [],
    error: 'No detections above confidence threshold (0.5)',
  },
  {
    id: 'src_7a2e8c4d1b65',
    file_name: 'loading_dock_03.jpg',
    url: 'https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=1600&auto=format&fit=crop',
    width: 1600, height: 1067,
    status: 'in_review',
    project: 'warehouse-ops',
    classes: ['pallet', 'worker', 'box'],
    uploaded_at: '2026-04-20T08:30:00Z',
    objects: [
      { object_id: 'obj_e5f6a7', category: 'pallet', bbox: [0.14, 0.58, 0.20, 0.28], confidence: 0.83, validated: 'approved' },
      { object_id: 'obj_e5f6a8', category: 'worker', bbox: [0.48, 0.28, 0.08, 0.44], confidence: 0.77, validated: null },
      { object_id: 'obj_e5f6a9', category: 'box',    bbox: [0.72, 0.56, 0.14, 0.20], confidence: 0.66, validated: null },
    ],
  },
];

const RECENT_RUNS = [
  { id: 'run_001', source_count: 142, status: 'completed', created: '09:42',  duration: '2m 14s', project: 'urban-mobility-v3' },
  { id: 'run_002', source_count: 38,  status: 'running',   created: '09:38',  duration: '—',      project: 'warehouse-ops',  progress: 0.62 },
  { id: 'run_003', source_count: 204, status: 'completed', created: '09:12',  duration: '4m 02s', project: 'urban-mobility-v3' },
  { id: 'run_004', source_count: 12,  status: 'failed',    created: '08:58',  duration: '0m 08s', project: 'warehouse-ops' },
  { id: 'run_005', source_count: 88,  status: 'completed', created: '08:40',  duration: '1m 44s', project: 'urban-mobility-v3' },
];

const PROJECTS = [
  { id: 'urban-mobility-v3', name: 'Urban Mobility v3', sources: 1284, validated: 892, classes: 14 },
  { id: 'warehouse-ops',     name: 'Warehouse Ops',     sources: 412,  validated: 287, classes: 8  },
  { id: 'retail-shelf-01',   name: 'Retail Shelf 01',   sources: 156,  validated: 156, classes: 22 },
];

Object.assign(window, { CATEGORY_COLORS, SOURCES, RECENT_RUNS, PROJECTS });
