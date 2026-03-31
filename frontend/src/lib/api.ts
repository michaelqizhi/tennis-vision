/**
 * API client for the Tennis Vision backend.
 * Communicates with FastAPI at /api/* (proxied via Next.js rewrites).
 */

const API_BASE = "/api";

/* ---------- Types matching backend Pydantic schemas ---------- */

export type JobStatus = "queued" | "processing" | "complete" | "failed";

export interface UploadResponse {
  job_id: string;
  status: JobStatus;
  message: string;
}

export interface StepProgress {
  step: string;
  status: JobStatus;
  message: string;
}

export interface StatusResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  current_step: string;
  steps: StepProgress[];
  error: string | null;
}

export interface BallPosition {
  frame_number: number;
  x: number | null;
  y: number | null;
  confidence: number;
  interpolated: boolean;
  court_x: number | null;
  court_y: number | null;
}

export interface Rally {
  rally_id: number;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  duration: number;
  shot_count: number;
}

export interface RallyStats {
  rally_count: number;
  total_shots: number;
  avg_rally_length: number;
  avg_shots_per_rally: number;
  longest_rally_duration: number;
  longest_rally_shots: number;
  shortest_rally_duration: number;
  shortest_rally_shots: number;
  median_rally_duration: number;
  median_shots_per_rally: number;
  shots_per_rally: number[];
  shot_count_distribution: Record<number, number>;
}

export interface ServeStats {
  total_serves: number;
  first_serves: number;
  second_serves: number;
  first_serve_in: number;
  second_serve_in: number;
  first_serve_faults: number;
  double_faults: number;
  aces: number;
  first_serve_pct: number;
  second_serve_pct: number;
}

export interface VisualizationPaths {
  shot_heatmap: string | null;
  serve_heatmap_all: string | null;
  serve_heatmap_first: string | null;
  serve_heatmap_second: string | null;
}

export interface AnalysisResults {
  job_id: string;
  video_filename: string;
  fps: number;
  total_frames: number;
  ball_positions: BallPosition[];
  rallies: Rally[];
  rally_stats: RallyStats;
  serve_stats: ServeStats;
  visualizations: VisualizationPaths;
}

export interface ResultsResponse {
  job_id: string;
  status: JobStatus;
  results: AnalysisResults | null;
  error: string | null;
}

/* ---------- API functions ---------- */

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }

  return res.json();
}

export async function getStatus(jobId: string): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/status/${jobId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Status check failed: ${res.status}`);
  }

  return res.json();
}

export async function getResults(jobId: string): Promise<ResultsResponse> {
  const res = await fetch(`${API_BASE}/results/${jobId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Results fetch failed: ${res.status}`);
  }

  return res.json();
}

/**
 * Poll job status until complete or failed.
 * Calls onProgress with each status update.
 * Returns the final results.
 */
export async function pollUntilComplete(
  jobId: string,
  onProgress?: (status: StatusResponse) => void,
  intervalMs: number = 2000
): Promise<ResultsResponse> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getStatus(jobId);
        onProgress?.(status);

        if (status.status === "complete") {
          const results = await getResults(jobId);
          resolve(results);
        } else if (status.status === "failed") {
          reject(new Error(status.error || "Processing failed"));
        } else {
          setTimeout(poll, intervalMs);
        }
      } catch (err) {
        reject(err);
      }
    };
    poll();
  });
}
