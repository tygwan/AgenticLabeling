export interface DetectedObject {
  label: string;
  confidence: number;
  color?: string; // For UI visualization
}

export interface AgentAnalysis {
  caption: string;
  objects: DetectedObject[];
  tags: string[];
  reasoning: string;
  technicalDetails: {
    lighting: string;
    composition: string;
  };
}

export interface UploadedMedia {
  url: string;
  type: 'image' | 'video';
  file: File;
  base64?: string;
}

export enum AppState {
  IDLE = 'IDLE',
  ANALYZING = 'ANALYZING',
  COMPLETE = 'COMPLETE',
  ERROR = 'ERROR'
}